use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
use braidinfer_core::types::DeviceId;
use braidinfer_hip::{HipResult, memory::MappedHostBuffer};

// Mirror of WatchdogState from kernels/watchdog.h.
// C layout with natural alignment:
//   force_exit       u32 @ 0
//   [implicit pad]   u32 @ 4   (compiler inserts to align progress_counter to 8)
//   progress_counter u64 @ 8
//   last_op_id       u32 @ 16
//   _pad             u32 @ 20  (explicit in header to align last_beat_us to 8)
//   last_beat_us     u64 @ 24
#[repr(C)]
pub struct WatchdogState {
    pub force_exit: u32,
    pub _pad0: u32,
    pub progress_counter: u64,
    pub last_op_id: u32,
    pub _pad1: u32,
    pub last_beat_us: u64,
}

// Watchdog configuration (env-configurable).
struct WatchdogConfig {
    poll_interval_ms: u64,
    no_progress_ms: u64,
    grace_ms: u64,
}

impl WatchdogConfig {
    fn from_env() -> Self {
        fn env_u64(name: &str, default: u64) -> u64 {
            std::env::var(name)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(default)
        }
        Self {
            poll_interval_ms: env_u64("BRAIDINFER_WATCHDOG_POLL_MS", 100),
            no_progress_ms: env_u64("BRAIDINFER_WATCHDOG_NO_PROGRESS_MS", 2000),
            grace_ms: env_u64("BRAIDINFER_WATCHDOG_GRACE_MS", 1000),
        }
    }

    fn disabled(&self) -> bool {
        self.no_progress_ms == 0
    }
}

// Per-GPU watchdog registration.
struct WatchdogEntry {
    device: DeviceId,
    state: MappedHostBuffer<WatchdogState>,
    last_progress: u64,
    last_progress_at: Instant,
    force_exit_sent_at: Option<Instant>,
}

// Host-side watchdog thread. Monitors all registered WatchdogState pages.
// Spawned at worker launch; joined at shutdown.
pub struct WatchdogThread {
    entries: Arc<Mutex<Vec<WatchdogEntry>>>,
    stop: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl WatchdogThread {
    pub fn spawn() -> Self {
        let entries: Arc<Mutex<Vec<WatchdogEntry>>> = Arc::new(Mutex::new(Vec::new()));
        let stop = Arc::new(AtomicBool::new(false));

        let entries_clone = Arc::clone(&entries);
        let stop_clone = Arc::clone(&stop);

        let handle = std::thread::spawn(move || {
            watchdog_thread_main(entries_clone, stop_clone);
        });

        WatchdogThread { entries, stop, handle: Some(handle) }
    }

    // Register a GPU's WatchdogState. Returns the device pointer to pass to the kernel.
    pub fn register(&self, device: DeviceId) -> HipResult<*mut WatchdogState> {
        let state = MappedHostBuffer::<WatchdogState>::alloc(1)?;
        // Zero-init: force_exit=0, progress_counter=0.
        unsafe {
            let ptr = state.host_ptr();
            std::ptr::write_bytes(ptr, 0, 1);
        }
        let dev_ptr = state.device_ptr() as *mut WatchdogState;

        let entry = WatchdogEntry {
            device,
            state,
            last_progress: 0,
            last_progress_at: Instant::now(),
            force_exit_sent_at: None,
        };

        self.entries.lock().unwrap().push(entry);
        Ok(dev_ptr)
    }

    pub fn stop_and_join(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

impl Drop for WatchdogThread {
    fn drop(&mut self) {
        self.stop_and_join();
    }
}

fn watchdog_thread_main(entries: Arc<Mutex<Vec<WatchdogEntry>>>, stop: Arc<AtomicBool>) {
    let cfg = WatchdogConfig::from_env();
    if cfg.disabled() {
        eprintln!("[watchdog] BRAIDINFER_WATCHDOG_NO_PROGRESS_MS=0: disabled");
        return;
    }

    let poll = Duration::from_millis(cfg.poll_interval_ms);
    let no_progress = Duration::from_millis(cfg.no_progress_ms);
    let grace = Duration::from_millis(cfg.grace_ms);

    eprintln!(
        "[watchdog] started: poll={}ms no_progress={}ms grace={}ms",
        cfg.poll_interval_ms, cfg.no_progress_ms, cfg.grace_ms
    );

    loop {
        std::thread::sleep(poll);
        if stop.load(Ordering::Relaxed) {
            break;
        }

        let mut entries = entries.lock().unwrap();
        for entry in entries.iter_mut() {
            let now = Instant::now();
            let state = unsafe { &*entry.state.host_ptr() };
            let counter = unsafe { std::ptr::read_volatile(&state.progress_counter) };
            let op_id  = unsafe { std::ptr::read_volatile(&state.last_op_id) };

            if counter != entry.last_progress {
                // Kernel is making progress.
                entry.last_progress = counter;
                entry.last_progress_at = now;
                entry.force_exit_sent_at = None;
                continue;
            }

            let stall = now.duration_since(entry.last_progress_at);

            if stall < no_progress {
                continue;
            }

            // No progress for no_progress_ms. Set force_exit.
            if entry.force_exit_sent_at.is_none() {
                eprintln!(
                    "[watchdog] GPU {}: no progress for {:?} (op_id={}, counter={}). Setting force_exit.",
                    entry.device.0, stall, op_id, counter
                );
                unsafe {
                    std::ptr::write_volatile(
                        &mut (*entry.state.host_ptr()).force_exit as *mut u32,
                        1u32,
                    );
                }
                entry.force_exit_sent_at = Some(now);
                continue;
            }

            // force_exit was sent — check if grace period expired.
            let grace_elapsed = now.duration_since(entry.force_exit_sent_at.unwrap());
            if grace_elapsed < grace {
                continue;
            }

            // Grace expired. Escalate: quiesce all GPUs then abort.
            // NOTE: hipDeviceReset blocks indefinitely on RDNA3/gfx1100 when the kernel
            // is still running (ROCm has no GPU TDR preemption for compute). The only
            // safe last-resort is process abort, which triggers amdgpu driver context
            // teardown and releases the GPU. Verified by watchdog_recovery_test --buggy.
            let wedged_device = entry.device;
            eprintln!(
                "[watchdog] GPU {}: kernel did not honor force_exit after {:?}. Escalating to abort.",
                wedged_device.0, grace_elapsed
            );

            // Collect all force_exit pointers for quiesce (cannot borrow entries mutably twice).
            let force_exit_ptrs: Vec<*mut u32> = entries.iter().map(|e| {
                unsafe { &mut (*e.state.host_ptr()).force_exit as *mut u32 }
            }).collect();

            // Quiesce all registered GPUs first to avoid poisoning P2P TLPs in flight.
            for ptr in &force_exit_ptrs {
                unsafe { std::ptr::write_volatile(*ptr, 1u32); }
            }
            std::thread::sleep(Duration::from_millis(200));

            dump_telemetry_and_abort(wedged_device, op_id, counter);
        }
    }

    eprintln!("[watchdog] stopped.");
}

fn dump_telemetry_and_abort(device: DeviceId, last_op_id: u32, last_counter: u64) -> ! {
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let pid = std::process::id();
    let path = format!("/tmp/watchdog_crash_{pid}_{ts}.log");

    let mut lines = Vec::new();
    lines.push(format!("WATCHDOG CRASH DUMP — pid={pid} ts={ts}"));
    lines.push(format!("GPU: {}", device.0));
    lines.push(format!("last_op_id: {last_op_id}"));
    lines.push(format!("last_progress_counter: {last_counter}"));

    // Capture dmesg tail.
    if let Ok(out) = std::process::Command::new("dmesg").output() {
        let text = String::from_utf8_lossy(&out.stdout);
        let tail: Vec<&str> = text.lines().rev().take(200).collect();
        lines.push("\n=== dmesg tail ===".to_string());
        for l in tail.into_iter().rev() {
            lines.push(l.to_string());
        }
    }

    // Capture gpu_reset counter.
    for entry in std::fs::read_dir("/sys/class/drm").into_iter().flatten().flatten() {
        let p = entry.path().join("device/gpu_reset");
        if let Ok(v) = std::fs::read_to_string(&p) {
            lines.push(format!("{}: {}", p.display(), v.trim()));
        }
    }

    let content = lines.join("\n");
    if let Err(e) = std::fs::write(&path, &content) {
        eprintln!("[watchdog] failed to write crash dump to {path}: {e}");
    } else {
        eprintln!("[watchdog] crash dump written to {path}");
    }

    eprintln!("[watchdog] aborting process.");
    std::process::abort();
}
