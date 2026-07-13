//! System-audio capture, per-OS. macOS uses a cidre Core Audio global tap;
//! Windows/Linux land in Phase 2b.

#[cfg(target_os = "macos")]
pub mod macos;
