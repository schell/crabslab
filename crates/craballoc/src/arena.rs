//! An arena allocator built on a `u32` slab.
//!
//! The [`Arena`] type provided by this module is the successor to [`SlabAllocator`].
//! Much like [`SlabAllocator`], [`Arena`] provides an API to dynamically allocate types that
//! implement [`SlabItem`] from the CPU, and then synchronize those changes to the GPU.
//! Unlike [`SlabAllocator`], [`Arena`] also provides synchronization _back_ from the GPU.
//! Only values that are explicitly set to sync will be synchronized.
use std::{
    borrow::Cow,
    sync::{atomic::AtomicU32, Arc, RwLock},
};

#[cfg(doc)]
use crate::slab::SlabAllocator;
use crate::{
    buffer::{manager::BufferManager, SlabBuffer},
    range::RangeManager,
    runtime::IsRuntime,
};

/// Value is synchronized on both CPU and GPU.
pub struct Bidirectional;
// Value is synchronized from CPU to the GPU, but any changes on the GPU will
// leave the value out of sync.
pub struct OneWayFromCpu;
// Value is synchronized from GPU to CPU but cannot be changed from the CPU.
pub struct OneWayFromGpu;

pub struct Value<T, const COUNT: u32 = 1, Sync = Bidirectional> {}

/// An arena allocator backed by a `u32` slab.
///
/// TODO: write about how values allocated from the arena can conditionally be
/// synchronized from the CPU to the GPU and back.
#[derive(Debug)]
pub struct Arena<R: IsRuntime> {
    /// Manages allocation on the slab
    buffer_manager: BufferManager<R>,
    /// Manages the ranges that will be synchronized from the CPU to the GPU.
    cpu_to_gpu_synchronization_ranges: RangeManager<u32>,
    /// Manages the ranges that will be synchronized from the GPU to the CPU.
    gpu_to_cpu_synchronization_ranges: RangeManager<u32>,
}

impl<Runtime: IsRuntime> Clone for Arena<Runtime> {
    fn clone(&self) -> Self {
        Self {
            buffer_manager: self.buffer_manager.clone(),
        }
    }
}

impl<R: IsRuntime> Arena<R> {
    pub fn new(
        runtime: impl AsRef<R>,
        label: impl Into<Cow<'static, str>>,
        default_buffer_usages: R::BufferUsages,
    ) -> Self {
        Self {
            buffer_manager: BufferManager::new(runtime, label, default_buffer_usages),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn sanity() {
        let arena = Arena::new(crate::wgpu_runtime(), "tests", wgpu::BufferUsages::empty());
    }
}
