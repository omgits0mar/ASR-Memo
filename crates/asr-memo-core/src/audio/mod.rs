//! Real audio engine (Phase 2a): capture → 50 ms mix → record + health.
//! Offline-testable core first; cpal/cidre captures are hardware-gated.

pub mod health;
pub mod mic;
pub mod mixer;
pub mod record;
pub mod resample;
pub mod system;
