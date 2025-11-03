//! Slab allocators that run on the CPU.
use crabslab::{Array, Id, SlabItem};
use rustc_hash::FxHashSet;
use snafu::prelude::*;
use std::{
    borrow::Cow,
    num::NonZeroU32,
    ops::Deref,
    sync::{Arc, RwLock},
};

use crate::{
    buffer::{manager::BumpAllocator, SlabBuffer},
    range::{Range, RangeManager},
    runtime::IsRuntime,
    update::{SourceId, Update, UpdateManager},
    value::{Hybrid, HybridArray},
};

#[cfg(feature = "wgpu")]
mod wgpu_slab;

#[derive(Debug, Snafu)]
#[snafu(visibility(pub(crate)))]
pub enum SlabAllocatorError {
    #[snafu(display(
        "Slab has no internal buffer. Please call SlabAllocator::commit or \
         SlabAllocator::get_updated_buffer first."
    ))]
    NoInternalBuffer,

    #[snafu(display("Async recv error: {source}"))]
    AsyncRecv { source: async_channel::RecvError },

    #[cfg(feature = "wgpu")]
    #[snafu(display("Async error: {source}"))]
    Async { source: wgpu::BufferAsyncError },

    #[cfg(feature = "wgpu")]
    #[snafu(display("Poll error: {source}"))]
    Poll { source: wgpu::PollError },

    #[snafu(display("{source}"))]
    Other { source: Box<dyn std::error::Error> },
}

/// Manages slab allocations and updates over a parameterised buffer.
///
/// Create a new instance using [`SlabAllocator::new`].
///
/// Upon creation you will need to call [`SlabAllocator::get_buffer`] or
/// [`SlabAllocator::commit`] at least once before any data is written to the
/// internal buffer.
pub struct SlabAllocator<Runtime: IsRuntime> {
    pub(crate) notifier: (
        async_channel::Sender<SourceId>,
        async_channel::Receiver<SourceId>,
    ),
    bump_allocator: BumpAllocator<Runtime>,
    update_manager: UpdateManager,

    // Recycled memory ranges
    pub(crate) recycles: Arc<RwLock<RangeManager<Range>>>,
}

impl<R: IsRuntime> Clone for SlabAllocator<R> {
    fn clone(&self) -> Self {
        SlabAllocator {
            notifier: self.notifier.clone(),
            bump_allocator: self.bump_allocator.clone(),
            update_manager: self.update_manager.clone(),
            recycles: self.recycles.clone(),
        }
    }
}

impl<R: IsRuntime> Deref for SlabAllocator<R> {
    type Target = BumpAllocator<R>;

    fn deref(&self) -> &Self::Target {
        &self.bump_allocator
    }
}

impl<R: IsRuntime> SlabAllocator<R> {
    pub fn new(
        runtime: impl AsRef<R>,
        name: impl Into<Cow<'static, str>>,
        default_buffer_usages: R::BufferUsages,
    ) -> Self {
        log::debug!("new slab allocator");
        Self {
            notifier: async_channel::unbounded(),
            recycles: Default::default(),
            bump_allocator: BumpAllocator::new(runtime, name, default_buffer_usages),
            update_manager: UpdateManager::default(),
        }
    }

    /// Whether the underlying buffer is empty.
    ///
    /// This does not include data that has not yet been committed.
    pub fn is_empty(&self) -> bool {
        self.bump_allocator.is_empty()
    }

    pub(crate) fn allocate<T: SlabItem>(&self) -> Id<T> {
        // UNWRAP: we want to panic
        let may_range = self.recycles.write().unwrap().remove(T::SLAB_SIZE as u32);
        if let Some(range) = may_range {
            let id = Id::<T>::new(range.first_index);
            log::trace!(
                "slab allocate {}: dequeued {range:?} to {id:?}",
                std::any::type_name::<T>()
            );
            debug_assert_eq!(
                range.last_index,
                range.first_index + T::SLAB_SIZE as u32 - 1
            );
            id
        } else if let Some(spaces) = NonZeroU32::new(T::SLAB_SIZE as u32) {
            let range = self.bump_allocator.alloc(spaces);
            Id::new(range.first_index)
        } else {
            Id::NONE
        }
    }

    pub(crate) fn allocate_array<T: SlabItem>(&self, len: usize) -> Array<T> {
        if len == 0 {
            return Array::default();
        }

        // UNWRAP: we want to panic
        let may_range = self
            .recycles
            .write()
            .unwrap()
            .remove((T::SLAB_SIZE * len) as u32);
        if let Some(range) = may_range {
            let array = Array::<T>::new(Id::new(range.first_index), len as u32);
            log::trace!(
                "slab allocate_array {len}x{}: dequeued {range:?} to {array:?}",
                std::any::type_name::<T>()
            );
            debug_assert_eq!(
                range.last_index,
                range.first_index + (T::SLAB_SIZE * len) as u32 - 1
            );
            array
        } else if let Some(spaces) = NonZeroU32::new(T::SLAB_SIZE as u32 * len as u32) {
            let range = self.bump_allocator.alloc(spaces);
            Array::new(Id::new(range.first_index), len as u32)
        } else {
            Array::NONE
        }
    }

    /// Stage a new value that lives on the GPU _and_ CPU.
    pub fn new_value<T: SlabItem + Clone + Send + Sync + 'static>(&self, value: T) -> Hybrid<T> {
        Hybrid::new(self, value)
    }

    /// Stage a contiguous array of new values that live on the GPU _and_ CPU.
    pub fn new_array<T: SlabItem + Clone + Send + Sync + 'static>(
        &self,
        values: impl IntoIterator<Item = T>,
    ) -> HybridArray<T> {
        HybridArray::new(self, values)
    }

    /// Return the `SourceId`s of all live values allocated by this slab.
    pub fn get_live_source_ids(&self) -> FxHashSet<SourceId> {
        self.update_manager.get_managed_source_ids()
    }

    /// Return the ids of all sources that require updating.
    pub fn get_updated_source_ids(&self) -> FxHashSet<SourceId> {
        self.update_manager.get_updated_source_ids()
    }

    /// Returns whether any update sources, most likely from [`Hybrid`] or [`Gpu`](crate::value::Gpu) values,
    /// have queued updates waiting to be committed.
    pub fn has_queued_updates(&self) -> bool {
        self.update_manager.has_queued_updates()
    }

    /// Defragments the internal "recycle" buffer.
    pub fn defrag(&self) {
        // UNWRAP: panic on purpose
        let mut recycle_guard = self.recycles.write().unwrap();
        for range in std::mem::take(&mut recycle_guard.ranges) {
            recycle_guard.add_range(range);
        }
    }

    /// Perform upkeep on the slab, synchronizing changes to the internal buffer.
    ///
    /// Changes made to allocated values stored on this slab are not committed
    /// until this function has been called, and sometimes not until the runtime
    /// has finished synchronizing the CPU to the GPU (eg with `wgpu::Device::poll`).
    ///
    /// The internal buffer is not created until the after the first time this
    /// function is called.
    ///
    /// Returns a [`SlabBuffer`] wrapping the internal buffer that is currently
    /// in use by the allocator.
    pub fn commit(&self) -> SlabBuffer<R::Buffer> {
        let buffer = self.bump_allocator.commit();
        let summary = self.update_manager.clear_updated_sources();

        {
            // Add the dropped sources bay to the recycle pool
            let mut guard = self.recycles.write().unwrap();
            for source_id in summary.recycle_ranges.ranges {
                guard.add_range(source_id.range);
            }
        }

        for Update { range, data } in summary.cpu_update_ranges.ranges {
            self.bump_allocator
                .runtime()
                .buffer_write(&buffer, range.into(), &data);
        }

        buffer
    }
}
