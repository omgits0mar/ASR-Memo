// swift-tools-version:5.9
//
// AudioTap — Core Audio Process-Tap capture helper for meeting_asr.
//
// A tiny macOS 14.4+ executable that creates a Core Audio Process Tap
// (`AudioHardwareCreateProcessTap` + `CATapDescription`) and streams raw PCM
// (Float32, mono, device-native rate) to stdout for the Python wrapper
// `meeting_asr.audio.coreaudio_tap.CoreAudioTapCapture` to resample to 16 kHz.
//
// This keeps all platform-native capture code isolated behind the
// `AudioCapture` protocol (Constitution III), meeting-app-agnostic (FR-002).
import PackageDescription

let package = Package(
    name: "AudioTap",
    platforms: [
        // Process Taps require macOS 14.4 (CATapDescription + AudioHardwareCreateProcessTap).
        .macOS("14.4")
    ],
    targets: [
        .executableTarget(
            name: "AudioTap",
            path: "Sources/AudioTap"
        )
    ]
)
