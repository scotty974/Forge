//! Allocateur global qui compte les octets alloués sur le tas.
//!
//! Il délègue tout à l'allocateur système et tient deux compteurs :
//! la mémoire actuellement allouée et le pic atteint depuis le dernier `reset_peak`.
//! Pour l'activer, le binaire doit le déclarer :
//!
//! ```ignore
//! #[global_allocator]
//! static ALLOC: ml::bench::allocator::TrackingAllocator = ml::bench::allocator::TrackingAllocator;
//! ```
//!
//! Sans cette déclaration, les compteurs restent à 0.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

pub struct TrackingAllocator;

static CURRENT: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

fn add(n: usize) {
    let now = CURRENT.fetch_add(n, Ordering::Relaxed) + n;
    PEAK.fetch_max(now, Ordering::Relaxed);
}

fn sub(n: usize) {
    CURRENT.fetch_sub(n, Ordering::Relaxed);
}

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            add(layout.size());
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            add(layout.size());
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        sub(layout.size());
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_ptr = unsafe { System.realloc(ptr, layout, new_size) };
        if !new_ptr.is_null() {
            let old_size = layout.size();
            if new_size > old_size {
                add(new_size - old_size);
            } else {
                sub(old_size - new_size);
            }
        }
        new_ptr
    }
}

/// Octets actuellement alloués sur le tas.
pub fn current_bytes() -> usize {
    CURRENT.load(Ordering::Relaxed)
}

/// Pic d'octets alloués depuis le dernier `reset_peak`.
pub fn peak_bytes() -> usize {
    PEAK.load(Ordering::Relaxed)
}

/// Ramène le pic à la valeur courante, pour mesurer une nouvelle phase.
pub fn reset_peak() {
    PEAK.store(CURRENT.load(Ordering::Relaxed), Ordering::Relaxed);
}

#[cfg(test)]
mod test {
    use super::*;

    // Les tests de la lib n'utilisent pas TrackingAllocator comme allocateur global :
    // on appelle ses méthodes directement, seul ce test touche aux compteurs.
    #[test]
    fn compte_alloc_realloc_dealloc() {
        let alloc = TrackingAllocator;
        let layout = Layout::from_size_align(1024, 8).unwrap();
        let base = current_bytes();
        reset_peak();

        unsafe {
            let ptr = alloc.alloc(layout);
            assert!(!ptr.is_null());
            assert_eq!(current_bytes(), base + 1024);

            let ptr = alloc.realloc(ptr, layout, 4096);
            assert!(!ptr.is_null());
            assert_eq!(current_bytes(), base + 4096);

            let big = Layout::from_size_align(4096, 8).unwrap();
            let ptr = alloc.realloc(ptr, big, 512);
            assert_eq!(current_bytes(), base + 512);

            alloc.dealloc(ptr, Layout::from_size_align(512, 8).unwrap());
        }

        assert_eq!(current_bytes(), base);
        assert_eq!(peak_bytes(), base + 4096, "le pic doit garder le maximum atteint");

        reset_peak();
        assert_eq!(peak_bytes(), base);
    }
}
