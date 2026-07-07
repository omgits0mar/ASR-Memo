use serde::{Deserialize, Serialize};

/// Audio origin. Serializes as "microphone" / "system" (app/dto.py contract).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AudioSource {
    Microphone,
    System,
}

/// SegmentDTO — the atomic rendered/exported unit (app/dto.py::segment_dto).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SegmentDto {
    pub segment_id: String,
    pub speaker_label: String,
    pub start: f64,
    pub end: f64,
    pub text: String,
    pub language: Option<String>,
    pub confidence: f64,
    pub confidence_band: Option<String>,
    pub source: Option<AudioSource>,
    pub is_final: bool,
}

/// SpeakerDTO (app/dto.py::speaker_dto).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeakerDto {
    pub label: String,
    pub color: String,
    pub total_speech_seconds: f64,
    pub segment_count: u32,
}

/// One model row inside ReadinessDTO (app/dto.py::_model_dto).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelDto {
    pub name: String,
    pub kind: String,
    pub state: String,
    pub is_cached: bool,
}

/// ReadinessDTO — drives the setup screen (app/dto.py::readiness_dto).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReadinessDto {
    pub ready: bool,
    pub compute_backend: String,
    pub os_supports_process_tap: bool,
    pub mic_permission: bool,
    pub system_audio_permission: bool,
    pub models: Vec<ModelDto>,
    pub missing: Vec<String>,
}

/// ErrorInfo (app/dto.py::error_dto).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ErrorInfoDto {
    pub code: String,
    pub message: String,
    pub recoverable: bool,
    pub hint: Option<String>,
}

/// prepare_progress event payload (app/dto.py::prepare_progress_dto).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PrepareProgressDto {
    pub asset: String,
    pub downloaded: u64,
    pub total: u64,
    pub fraction: f64,
    pub state: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Golden contract: must match app/dto.py::segment_dto exactly.
    #[test]
    fn segment_dto_matches_python_bridge_contract() {
        let seg = SegmentDto {
            segment_id: "seg-1".into(),
            speaker_label: "S1".into(),
            start: 0.0,
            end: 1.25,
            text: "hello".into(),
            language: Some("en".into()),
            confidence: 0.9,
            confidence_band: Some("high".into()),
            source: Some(AudioSource::System),
            is_final: true,
        };
        let v = serde_json::to_value(&seg).unwrap();
        assert_eq!(
            v,
            serde_json::json!({
                "segment_id": "seg-1", "speaker_label": "S1",
                "start": 0.0, "end": 1.25, "text": "hello",
                "language": "en", "confidence": 0.9,
                "confidence_band": "high", "source": "system",
                "is_final": true
            })
        );
    }

    /// Golden contract: must match app/dto.py::readiness_dto exactly.
    #[test]
    fn readiness_dto_matches_python_bridge_contract() {
        let r = ReadinessDto {
            ready: false,
            compute_backend: "fake".into(),
            os_supports_process_tap: true,
            mic_permission: true,
            system_audio_permission: false,
            models: vec![ModelDto {
                name: "asr".into(),
                kind: "asr".into(),
                state: "missing".into(),
                is_cached: false,
            }],
            missing: vec!["system audio permission".into()],
        };
        let v = serde_json::to_value(&r).unwrap();
        assert_eq!(
            v,
            serde_json::json!({
                "ready": false, "compute_backend": "fake",
                "os_supports_process_tap": true, "mic_permission": true,
                "system_audio_permission": false,
                "models": [{"name": "asr", "kind": "asr", "state": "missing", "is_cached": false}],
                "missing": ["system audio permission"]
            })
        );
    }
}
