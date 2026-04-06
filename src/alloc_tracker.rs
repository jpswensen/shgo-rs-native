//! Tracking allocator for memory profiling.
//!
//! When the `track-alloc` feature is enabled, this module provides a global
//! allocator wrapper that tracks allocation statistics: peak heap usage,
//! total bytes allocated/freed, and allocation count.
//!
//! # Usage
//!
//! In your binary (example or bench), add at the top:
//! ```ignore
//! #[cfg(feature = "track-alloc")]
//! #[global_allocator]
//! static ALLOC: shgo::alloc_tracker::TrackingAllocator = shgo::alloc_tracker::TrackingAllocator::new();
//! ```
//!
//! Then query stats with:
//! ```ignore
//! #[cfg(feature = "track-alloc")]
//! {
//!     let snap = shgo::alloc_tracker::snapshot();
//!     println!("Peak heap: {} bytes", snap.peak_bytes);
//! }
//! ```

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Global counters for allocation tracking.
static CURRENT_BYTES: AtomicUsize = AtomicUsize::new(0);
static PEAK_BYTES: AtomicUsize = AtomicUsize::new(0);
static TOTAL_ALLOCATED: AtomicUsize = AtomicUsize::new(0);
static TOTAL_FREED: AtomicUsize = AtomicUsize::new(0);
static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
static FREE_COUNT: AtomicUsize = AtomicUsize::new(0);

/// A thin wrapper around the system allocator that tracks memory usage.
pub struct TrackingAllocator;

impl TrackingAllocator {
    pub const fn new() -> Self {
        TrackingAllocator
    }
}

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            let size = layout.size();
            TOTAL_ALLOCATED.fetch_add(size, Ordering::Relaxed);
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
            let current = CURRENT_BYTES.fetch_add(size, Ordering::Relaxed) + size;
            // Update peak using a CAS loop
            let mut peak = PEAK_BYTES.load(Ordering::Relaxed);
            while current > peak {
                match PEAK_BYTES.compare_exchange_weak(
                    peak,
                    current,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => break,
                    Err(actual) => peak = actual,
                }
            }
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        let size = layout.size();
        TOTAL_FREED.fetch_add(size, Ordering::Relaxed);
        FREE_COUNT.fetch_add(1, Ordering::Relaxed);
        CURRENT_BYTES.fetch_sub(size, Ordering::Relaxed);
        unsafe { System.dealloc(ptr, layout) };
    }
}

/// A point-in-time snapshot of allocation statistics.
#[derive(Debug, Clone, Copy)]
pub struct AllocSnapshot {
    /// Current heap usage in bytes.
    pub current_bytes: usize,
    /// Peak heap usage observed since last reset.
    pub peak_bytes: usize,
    /// Total bytes allocated since last reset.
    pub total_allocated: usize,
    /// Total bytes freed since last reset.
    pub total_freed: usize,
    /// Number of allocations since last reset.
    pub alloc_count: usize,
    /// Number of frees since last reset.
    pub free_count: usize,
}

impl AllocSnapshot {
    /// Format bytes in a human-readable way.
    pub fn fmt_bytes(bytes: usize) -> String {
        if bytes >= 1_073_741_824 {
            format!("{:.2} GiB", bytes as f64 / 1_073_741_824.0)
        } else if bytes >= 1_048_576 {
            format!("{:.2} MiB", bytes as f64 / 1_048_576.0)
        } else if bytes >= 1024 {
            format!("{:.2} KiB", bytes as f64 / 1024.0)
        } else {
            format!("{} B", bytes)
        }
    }
}

impl std::fmt::Display for AllocSnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "current={}, peak={}, total_alloc={}, allocs={}, frees={}",
            Self::fmt_bytes(self.current_bytes),
            Self::fmt_bytes(self.peak_bytes),
            Self::fmt_bytes(self.total_allocated),
            self.alloc_count,
            self.free_count,
        )
    }
}

/// Take a snapshot of current allocation statistics.
pub fn snapshot() -> AllocSnapshot {
    AllocSnapshot {
        current_bytes: CURRENT_BYTES.load(Ordering::Relaxed),
        peak_bytes: PEAK_BYTES.load(Ordering::Relaxed),
        total_allocated: TOTAL_ALLOCATED.load(Ordering::Relaxed),
        total_freed: TOTAL_FREED.load(Ordering::Relaxed),
        alloc_count: ALLOC_COUNT.load(Ordering::Relaxed),
        free_count: FREE_COUNT.load(Ordering::Relaxed),
    }
}

/// Reset all counters to zero. Useful for isolating measurements.
pub fn reset() {
    CURRENT_BYTES.store(0, Ordering::Relaxed);
    PEAK_BYTES.store(0, Ordering::Relaxed);
    TOTAL_ALLOCATED.store(0, Ordering::Relaxed);
    TOTAL_FREED.store(0, Ordering::Relaxed);
    ALLOC_COUNT.store(0, Ordering::Relaxed);
    FREE_COUNT.store(0, Ordering::Relaxed);
}

/// Reset only the peak counter (keeps current_bytes accurate).
/// Useful for measuring peak of a specific section.
pub fn reset_peak() {
    let current = CURRENT_BYTES.load(Ordering::Relaxed);
    PEAK_BYTES.store(current, Ordering::Relaxed);
    TOTAL_ALLOCATED.store(0, Ordering::Relaxed);
    TOTAL_FREED.store(0, Ordering::Relaxed);
    ALLOC_COUNT.store(0, Ordering::Relaxed);
    FREE_COUNT.store(0, Ordering::Relaxed);
}
