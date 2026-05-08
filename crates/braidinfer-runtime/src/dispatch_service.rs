//! `DispatchService` — abstracts the inference loop so Phase 3's daemon can
//! plug in over Unix-socket RPC without changing the consumers.
//!
//! Plan: ~/.claude/plans/PLAN-dispatch-daemon.md (epic braidinfer-wks).
//!
//! Two implementations land:
//!
//! - [`InProcessDispatch`] (this file) — wraps a single in-process [`Model`].
//!   Used today by `bin/generate` and the existing test surface.
//! - `RemoteDispatch` (Phase 3+) — opens the daemon's Unix socket and
//!   serializes calls. Lives in `crates/braidinfer-client` once the daemon
//!   binary exists.
//!
//! ## Trait shape
//!
//! `decode_step_batch` accepts a batch of `(session, input_token, position)`
//! requests and returns one output token per request. With a single-element
//! batch this is today's per-token decode path. With multiple elements it
//! lets a Phase-6 continuous-batching scheduler pack tokens from multiple
//! sessions into one decode pass without changing the trait surface
//! (review feedback P2-6 on PLAN-dispatch-daemon.md).
//!
//! ## Session model
//!
//! `SessionId` is opaque to the consumer. The in-process implementation
//! treats it as a noop label (today's `Model` owns exactly one logical
//! session); the daemon will use it as the key into a per-session KV
//! cache pool.

use crate::model::{Model, ModelError};

/// Opaque session identifier. `InProcessDispatch` accepts any value but
/// only one session can be alive at a time. The daemon maps this to a
/// per-user KV cache slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SessionId(pub u32);

#[derive(Debug)]
pub enum DispatchServiceError {
    /// Underlying model error (in-process path).
    Model(ModelError),
    /// Multi-session not supported by this implementation.
    MultiSessionUnsupported,
    /// SIGINT/SIGTERM received during a dispatch.
    Shutdown,
}

impl From<ModelError> for DispatchServiceError {
    fn from(e: ModelError) -> Self {
        DispatchServiceError::Model(e)
    }
}

impl std::fmt::Display for DispatchServiceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DispatchServiceError::Model(e) => write!(f, "model error: {e:?}"),
            DispatchServiceError::MultiSessionUnsupported => {
                write!(f, "multi-session not supported by this dispatch implementation")
            }
            DispatchServiceError::Shutdown => write!(f, "dispatch interrupted by shutdown signal"),
        }
    }
}

impl std::error::Error for DispatchServiceError {}

/// Inference dispatch surface. Generic over local-process and
/// daemon-RPC implementations.
pub trait DispatchService {
    /// Allocate a fresh inference session. The session owns its KV cache
    /// region; reset to position 0. In-process implementations may only
    /// support one session at a time.
    fn create_session(&mut self) -> Result<SessionId, DispatchServiceError>;

    /// Free a session's KV cache. The id is invalid after this call.
    fn drop_session(&mut self, session: SessionId);

    /// Run prefill on a session's prompt and return the argmax of the
    /// last-position logits (the first generated token).
    fn prefill(
        &mut self,
        session: SessionId,
        prompt: &[u32],
    ) -> Result<u32, DispatchServiceError>;

    /// Run one decode step per request and return one next-token argmax
    /// per request, in the same order. Single-element batch = today's
    /// per-token decode. Multi-element batch = Phase-6 continuous batching.
    fn decode_step_batch(
        &mut self,
        requests: &[DecodeRequest],
    ) -> Result<Vec<u32>, DispatchServiceError>;
}

/// One decode-step request inside a batch.
#[derive(Debug, Clone, Copy)]
pub struct DecodeRequest {
    pub session: SessionId,
    pub input_token: u32,
    pub position: u32,
}

/// In-process implementation backed by a single [`Model`]. One active
/// session at a time. Used by `bin/generate` and existing tests; will
/// remain the default until the daemon binary lands in Phase 3.
pub struct InProcessDispatch {
    model: Model,
    /// Currently-allocated session id, or `None` if no session is active.
    /// The model owns exactly one KV-cache slot; multiple concurrent
    /// sessions are not representable yet.
    active: Option<SessionId>,
    next_session_id: u32,
}

impl InProcessDispatch {
    pub fn new(model: Model) -> Self {
        Self {
            model,
            active: None,
            next_session_id: 1,
        }
    }

    /// Borrow the underlying model. Useful for callers that still need
    /// model-specific knobs that haven't been promoted onto the trait
    /// (e.g. the trace-mode methods, debug accessors).
    pub fn model(&self) -> &Model {
        &self.model
    }

    pub fn model_mut(&mut self) -> &mut Model {
        &mut self.model
    }

    /// Consume and return the inner model — required for the existing
    /// `op_profile_dump` shutdown sequence which drops the model to
    /// flush the persistent worker before reading counters.
    pub fn into_model(self) -> Model {
        self.model
    }
}

impl DispatchService for InProcessDispatch {
    fn create_session(&mut self) -> Result<SessionId, DispatchServiceError> {
        if self.active.is_some() {
            return Err(DispatchServiceError::MultiSessionUnsupported);
        }
        // No-op on the model side — `Model` is reset on demand by callers
        // that need it (e.g. bench_coherence). For a fresh `generate`
        // invocation the model is already at position 0.
        let id = SessionId(self.next_session_id);
        self.next_session_id += 1;
        self.active = Some(id);
        Ok(id)
    }

    fn drop_session(&mut self, session: SessionId) {
        if self.active == Some(session) {
            self.active = None;
        }
    }

    fn prefill(
        &mut self,
        session: SessionId,
        prompt: &[u32],
    ) -> Result<u32, DispatchServiceError> {
        if self.active != Some(session) {
            return Err(DispatchServiceError::MultiSessionUnsupported);
        }
        let logits = self.model.prefill(prompt)?;
        Ok(argmax(&logits))
    }

    fn decode_step_batch(
        &mut self,
        requests: &[DecodeRequest],
    ) -> Result<Vec<u32>, DispatchServiceError> {
        if requests.len() != 1 {
            // Continuous batching is Phase 6. Until then, in-process is
            // strictly one decode at a time.
            return Err(DispatchServiceError::MultiSessionUnsupported);
        }
        let r = requests[0];
        if self.active != Some(r.session) {
            return Err(DispatchServiceError::MultiSessionUnsupported);
        }
        let token = self.model.decode_step_token(r.input_token, r.position)?;
        Ok(vec![token])
    }
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best_id = 0u32;
    let mut best_val = f32::MIN;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_id = i as u32;
        }
    }
    best_id
}
