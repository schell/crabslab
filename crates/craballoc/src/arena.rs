//! An arena allocator built on a `u32` slab.
//!
//! The [`Arena`] type provided by this module is the successor to [`SlabAllocator`].
//! Much like [`SlabAllocator`], [`Arena`] provides an API to dynamically allocate types that
//! implement [`SlabItem`] from the CPU, and then synchronize those changes to the GPU.
//! Unlike [`SlabAllocator`], [`Arena`] also provides synchronization _back_ from the GPU.
//! Only values that are explicitly set to sync will be synchronized.
use std::{
    borrow::Cow,
    marker::PhantomData,
    sync::{Arc, RwLock},
};

use crabslab::SlabItem;

#[cfg(doc)]
use crate::slab::SlabAllocator;
use crate::{
    buffer::{manager::BufferManager, SlabBuffer},
    range::{Range, RangeManager},
    runtime::IsRuntime,
    update::{CpuUpdateSource, SourceId},
};

/// Value is synchronized on both CPU and GPU.
pub struct Bidirectional {
    cpu_update_source: CpuUpdateSource,
}

// Value is synchronized from CPU to the GPU, but any changes on the GPU will
// leave the value out of sync.
pub struct OneWayFromCpu {
    cpu_update_source: CpuUpdateSource,
}

// Value is synchronized from GPU to CPU but cannot be changed from the CPU.
pub struct OneWayFromGpu;

pub struct Value<T: SlabItem, Sync = Bidirectional> {
    sync_source: Sync,
    /// A channell to send updates into.
    _phantom: PhantomData<(T, Sync)>,
}

/// An arena allocator backed by a `u32` slab.
///
/// TODO: write about how values allocated from the arena can conditionally be
/// synchronized from the CPU to the GPU and back.
#[derive(Debug)]
pub struct Arena<R: IsRuntime> {
    /// Manages allocation on the slab
    buffer_manager: BufferManager<R>,
    /// Manages the ranges that will be synchronized from the CPU to the GPU.
    cpu_to_gpu_synchronization_ranges: Arc<RwLock<RangeManager<Range>>>,
    /// Manages the ranges that will be synchronized from the GPU to the CPU.
    gpu_to_cpu_synchronization_ranges: Arc<RwLock<RangeManager<Range>>>,
}

impl<Runtime: IsRuntime> Clone for Arena<Runtime> {
    fn clone(&self) -> Self {
        Self {
            buffer_manager: self.buffer_manager.clone(),
            cpu_to_gpu_synchronization_ranges: self.gpu_to_cpu_synchronization_ranges.clone(),
            gpu_to_cpu_synchronization_ranges: self.cpu_to_gpu_synchronization_ranges.clone(),
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
            cpu_to_gpu_synchronization_ranges: Default::default(),
            gpu_to_cpu_synchronization_ranges: Default::default(),
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
