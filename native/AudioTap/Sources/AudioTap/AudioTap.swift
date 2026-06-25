//
//  main.swift
//  AudioTap
//
//  Core Audio Process-Tap capture helper for meeting_asr.
//
//  Creates a Core Audio Process Tap (`AudioHardwareCreateProcessTap` +
//  `CATapDescription`, macOS 14.4+), wraps it in a private aggregate device,
//  attaches an IOProc, and streams the captured raw Float32 PCM (mono mixdown)
//  to stdout for the Python wrapper
//  `meeting_asr.audio.coreaudio_tap.CoreAudioTapCapture` to resample to 16 kHz.
//
//  On startup it prints the resolved tap stream format to stderr as
//  `rate=<N>` and `channels=<N>` so the wrapper resamples from the true
//  device-native rate (Process Taps run at the device native rate, ~48 kHz).
//  stdout carries raw PCM only.
//
//  The Python side reads this pipe; this helper never touches the network or
//  disk beyond stdout/stderr. Meeting-app-agnostic (FR-002). Permission is
//  requested via the standard macOS TCC capture prompt on first run; a denial
//  is reported on stderr (exit code != 0) so the wrapper raises
//  `CapturePermissionError` (FR-015).
//
//  Build: swift build -c release --package-path native/AudioTap
//

import CoreAudio
import Foundation

private func stderr(_ message: String) {
    let line = message + "\n"
    FileHandle.standardError.write(line.data(using: .utf8) ?? Data())
}

/// Translate a numeric pid to its Core Audio process AudioObjectID (for a
/// process-scoped tap). Returns nil if the pid has no audio object.
private func processObjectID(forPID pid: pid_t) -> AudioObjectID? {
    var pidVar = pid
    var address = AudioObjectPropertyAddress(
        mSelector: kAudioHardwarePropertyTranslatePIDToProcessObject,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain
    )
    var objectID: AudioObjectID = 0
    var size = UInt32(MemoryLayout<AudioObjectID>.size)
    let status = AudioObjectGetPropertyData(
        AudioObjectID(kAudioObjectSystemObject), &address,
        UInt32(MemoryLayout<pid_t>.size), &pidVar, &size, &objectID
    )
    return (status == noErr && objectID != 0) ? objectID : nil
}

/// Read the tap's resolved stream format (`kAudioTapPropertyFormat`).
private func tapStreamFormat(_ tapID: AudioObjectID) -> AudioStreamBasicDescription? {
    var address = AudioObjectPropertyAddress(
        mSelector: kAudioTapPropertyFormat,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain
    )
    var asbd = AudioStreamBasicDescription()
    var size = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
    let status = AudioObjectGetPropertyData(tapID, &address, 0, nil, &size, &asbd)
    return status == noErr ? asbd : nil
}

@main
struct AudioTap {
    static let semaphore = DispatchSemaphore(value: 0)

    /// Parse `--process <pid>` (default: system-wide tap) and run until
    /// stdin closes or the process is signalled.
    static func main() {
        let args = CommandLine.arguments.dropFirst()
        var processName: String?
        var idx = args.startIndex
        while idx < args.endIndex {
            if args[idx] == "--process", idx + 1 < args.endIndex {
                processName = args[args.index(after: idx)]
                idx = args.index(idx, offsetBy: 2)
            } else if args[idx] == "--help" {
                stderr("usage: AudioTap [--process <pid>]")
                return
            } else {
                idx = args.index(after: idx)
            }
        }

        runCapture(processName: processName)
    }

    /// Build the tap + aggregate device + IOProc and stream float32 PCM to
    /// stdout until terminated.
    static func runCapture(processName: String?) {
        // 1) Tap description: system-wide mono mixdown by default, or a
        //    process-scoped tap when `--process <pid>` resolves to an audio object.
        let tapDescription: CATapDescription
        if let name = processName, let pid = pid_t(name), let objectID = processObjectID(forPID: pid) {
            tapDescription = CATapDescription(__monoMixdownOfProcesses: [NSNumber(value: objectID)])
            stderr("note: process-scoped tap for pid \(pid) (object \(objectID))")
        } else {
            if processName != nil {
                stderr("note: could not resolve --process to an audio object; falling back to system-wide tap")
            }
            tapDescription = CATapDescription(__monoGlobalTapButExcludeProcesses: [])
        }
        tapDescription.name = "meeting_asr_tap"
        tapDescription.isPrivate = true
        tapDescription.muteBehavior = .unmuted

        // 2) Create the Process Tap. Returns a non-zero AudioObjectID on success;
        //    a TCC denial surfaces here as a non-noErr status.
        var tapObjectID: AudioObjectID = 0
        var status = AudioHardwareCreateProcessTap(tapDescription, &tapObjectID)
        guard status == noErr, tapObjectID != 0 else {
            stderr("permission denied: AudioHardwareCreateProcessTap failed (\(status))")
            exit(2)
        }
        defer { AudioHardwareDestroyProcessTap(tapObjectID) }

        // 3) Resolve the tap's native stream format and report it to stderr so the
        //    Python wrapper resamples from the true device rate (≈48 kHz), not 16 kHz.
        let asbd = tapStreamFormat(tapObjectID)
        let sampleRate = Int((asbd?.mSampleRate ?? 48000).rounded())
        let channels = Int(asbd?.mChannelsPerFrame ?? 1)
        stderr("rate=\(sampleRate)")
        stderr("channels=\(channels)")

        // 4) Wrap the tap in a private aggregate device — the IOProc reads the
        //    tapped audio from this device (the standard macOS 14.4+ tap pattern).
        let aggregateUID = UUID().uuidString
        let aggregateDescription: [String: Any] = [
            kAudioAggregateDeviceNameKey: "meeting_asr_aggregate",
            kAudioAggregateDeviceUIDKey: aggregateUID,
            kAudioAggregateDeviceIsPrivateKey: true,
            kAudioAggregateDeviceIsStackedKey: false,
            kAudioAggregateDeviceTapAutoStartKey: true,
            kAudioAggregateDeviceTapListKey: [
                [
                    kAudioSubTapUIDKey: tapDescription.uuid.uuidString,
                    kAudioSubTapDriftCompensationKey: true,
                ],
            ],
        ]
        var aggregateID: AudioObjectID = 0
        status = AudioHardwareCreateAggregateDevice(aggregateDescription as CFDictionary, &aggregateID)
        guard status == noErr, aggregateID != 0 else {
            stderr("error: AudioHardwareCreateAggregateDevice failed (\(status))")
            exit(3)
        }
        defer { AudioHardwareDestroyAggregateDevice(aggregateID) }

        // Watchdog: `AudioDeviceCreateIOProcIDWithBlock` issues a synchronous mach
        // RPC to coreaudiod that only completes once the audio-capture TCC grant is
        // resolved. When the host is an unsigned command-line binary (no
        // NSAudioCaptureUsageDescription / TCC identity), coreaudiod can't present
        // the prompt and the call blocks forever — which would freeze the Python
        // reader. Arm a watchdog so a stalled authorization exits cleanly as a
        // permission denial (FR-015) instead of hanging the app. The window is
        // generous because, from a signed bundle, this same RPC blocks while the
        // first-run TCC prompt is up — the user needs time to click Allow.
        let watchdog = DispatchWorkItem {
            stderr("permission denied: audio-capture authorization timed out "
                + "(run from the signed app bundle with NSAudioCaptureUsageDescription)")
            exit(2)
        }
        DispatchQueue.global().asyncAfter(deadline: .now() + 30.0, execute: watchdog)

        // 5) Attach an IOProc that copies each captured block's float32 samples to
        //    stdout. The block runs on the provided dispatch queue (not the HAL
        //    real-time thread), so writing to the pipe is safe and applies natural
        //    backpressure if the Python reader falls behind.
        let stdoutHandle = FileHandle.standardOutput
        let ioQueue = DispatchQueue(label: "meeting_asr.audiotap.io")
        var ioProcID: AudioDeviceIOProcID?
        status = AudioDeviceCreateIOProcIDWithBlock(&ioProcID, aggregateID, ioQueue) {
            _, inInputData, _, _, _ in
            let bufferList = inInputData.pointee
            let count = Int(bufferList.mNumberBuffers)
            withUnsafePointer(to: bufferList.mBuffers) { headPtr in
                headPtr.withMemoryRebound(to: AudioBuffer.self, capacity: count) { buffers in
                    for i in 0..<count {
                        let buffer = buffers[i]
                        if let data = buffer.mData, buffer.mDataByteSize > 0 {
                            stdoutHandle.write(Data(bytes: data, count: Int(buffer.mDataByteSize)))
                        }
                    }
                }
            }
        }
        // RPC returned → authorization resolved; disarm the watchdog.
        watchdog.cancel()
        guard status == noErr, let ioProcID else {
            stderr("error: AudioDeviceCreateIOProcIDWithBlock failed (\(status))")
            exit(4)
        }
        defer { AudioDeviceDestroyIOProcID(aggregateID, ioProcID) }

        status = AudioDeviceStart(aggregateID, ioProcID)
        guard status == noErr else {
            stderr("error: AudioDeviceStart failed (\(status))")
            exit(5)
        }
        defer { AudioDeviceStop(aggregateID, ioProcID) }

        stderr("AudioTap: streaming process-tap PCM to stdout "
            + "(pid=\(ProcessInfo.processInfo.processIdentifier), rate=\(sampleRate), channels=\(channels))")

        // Run until SIGTERM/SIGINT, then the deferred teardown runs. Signal
        // handlers can't capture context, so use a global semaphore.
        installSignalHandlers()
        AudioTap.semaphore.wait()
    }
}

private func installSignalHandlers() {
    signal(SIGINT) { _ in AudioTap.semaphore.signal() }
    signal(SIGTERM) { _ in AudioTap.semaphore.signal() }
}
