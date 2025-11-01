//! Slab allocators that run on the CPU.
use core::sync::atomic::{AtomicUsize, Ordering};
use crabslab::{Array, Id, SlabItem};
use rustc_hash::{FxHashMap, FxHashSet};
use snafu::prelude::*;
use std::{
    borrow::Cow,
    hash::Hash,
    num::NonZeroU32,
    ops::Deref,
    sync::{Arc, RwLock},
};

use crate::{
    buffer::{manager::BufferManager, SlabBuffer},
    range::{Range, RangeManager},
    runtime::{IsRuntime, SlabUpdate},
    value::{Hybrid, HybridArray, WeakGpuRef},
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

/// An identifier for a unique source of updates.
#[derive(Clone, Copy, Debug)]
pub struct SourceId {
    pub key: usize,
    /// This field is just for debugging.
    pub type_is: &'static str,
}

impl core::fmt::Display for SourceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("{}({})", self.type_is, self.key))
    }
}

impl PartialEq for SourceId {
    fn eq(&self, other: &Self) -> bool {
        self.key.eq(&other.key)
    }
}

impl Eq for SourceId {}

#[allow(clippy::non_canonical_partial_ord_impl)]
impl PartialOrd for SourceId {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.key.cmp(&other.key))
    }
}

impl Ord for SourceId {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.key.cmp(&other.key)
    }
}

impl Hash for SourceId {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.key.hash(state)
    }
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
    buffer_manager: BufferManager<Runtime>,

    // The next monotonically increasing update identifier
    pub(crate) update_k: Arc<AtomicUsize>,
    // Weak references to all values that can write updates into this slab
    pub(crate) update_sources: Arc<RwLock<FxHashMap<SourceId, WeakGpuRef>>>,
    // Set of ids of the update sources that have updates queued
    update_queue: Arc<RwLock<FxHashSet<SourceId>>>,
    // Recycled memory ranges
    pub(crate) recycles: Arc<RwLock<RangeManager<Range>>>,
}

impl<R: IsRuntime> Clone for SlabAllocator<R> {
    fn clone(&self) -> Self {
        SlabAllocator {
            notifier: self.notifier.clone(),
            buffer_manager: self.buffer_manager.clone(),
            update_k: self.update_k.clone(),
            update_sources: self.update_sources.clone(),
            update_queue: self.update_queue.clone(),
            recycles: self.recycles.clone(),
        }
    }
}

impl<R: IsRuntime> Deref for SlabAllocator<R> {
    type Target = BufferManager<R>;

    fn deref(&self) -> &Self::Target {
        &self.buffer_manager
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
            update_k: Default::default(),
            update_sources: Default::default(),
            update_queue: Default::default(),
            recycles: Default::default(),
            buffer_manager: BufferManager::new(runtime, name, default_buffer_usages),
        }
    }

    pub(crate) fn next_update_k(&self) -> usize {
        self.update_k.fetch_add(1, Ordering::Relaxed)
    }

    pub(crate) fn insert_update_source(&self, id: SourceId, source: WeakGpuRef) {
        log::trace!("{} insert_update_source {id}", self.buffer_manager.label());
        let _ = self.notifier.0.try_send(id);
        // UNWRAP: panic on purpose
        self.update_sources.write().unwrap().insert(id, source);
    }

    /// Whether the underlying buffer is empty.
    ///
    /// This does not include data that has not yet been committed.
    pub fn is_empty(&self) -> bool {
        self.buffer_manager.is_empty()
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
            let range = self.buffer_manager.alloc(spaces);
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
            let range = self.buffer_manager.alloc(spaces);
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

    /// Return the ids of all sources that require updating.
    pub fn get_updated_source_ids(&self) -> FxHashSet<SourceId> {
        // UNWRAP: panic on purpose
        let mut update_set = self.update_queue.write().unwrap();
        while let Ok(source_id) = self.notifier.1.try_recv() {
            update_set.insert(source_id);
        }
        update_set.clone()
    }

    /// Build the set of sources that require updates, draining the source
    /// notifier and resetting the stored `update_queue`.
    ///
    /// This also places recycled items into the recycle bin.
    fn drain_updated_sources(&self) -> RangeManager<SlabUpdate> {
        let update_set = self.get_updated_source_ids();
        // UNWRAP: panic on purpose
        *self.update_queue.write().unwrap() = Default::default();
        // Prepare all of our GPU buffer writes
        let mut writes = RangeManager::<SlabUpdate>::default();
        {
            // Recycle any update sources that are no longer needed, and collect the active
            // sources' updates into `writes`.
            let mut updates_guard = self.update_sources.write().unwrap();
            let mut recycles_guard = self.recycles.write().unwrap();
            for id in update_set {
                let delete = if let Some(gpu_ref) = updates_guard.get_mut(&id) {
                    let count = gpu_ref.weak.strong_count();
                    if count == 0 {
                        // recycle this allocation
                        let array = gpu_ref.u32_array;
                        log::debug!(
                            "{} drain_updated_sources: recycling {id} {array:?}",
                            self.buffer_manager.label()
                        );
                        if array.is_null() {
                            log::debug!("  cannot recycle, null");
                        } else if array.is_empty() {
                            log::debug!("  cannot recycle, empty");
                        } else {
                            recycles_guard.add_range(gpu_ref.u32_array.into());
                        }
                        true
                    } else {
                        gpu_ref.get_update().into_iter().flatten().for_each(|u| {
                            log::trace!("updating {id} {:?}", u.array);
                            writes.add_range(u)
                        });
                        false
                    }
                } else {
                    log::debug!("could not find {id}");
                    false
                };
                if delete {
                    let _ = updates_guard.remove(&id);
                }
            }
            // Defrag the recycle ranges
            let ranges = std::mem::take(&mut recycles_guard.ranges);
            let num_ranges_to_defrag = ranges.len();
            for range in ranges.into_iter() {
                recycles_guard.add_range(range);
            }
            let num_ranges = recycles_guard.ranges.len();
            if num_ranges < num_ranges_to_defrag {
                log::trace!("{num_ranges_to_defrag} ranges before, {num_ranges} after");
            }
        }

        writes
    }

    /// Returns whether any update sources, most likely from [`Hybrid`] or [`Gpu`](crate::value::Gpu) values,
    /// have queued updates waiting to be committed.
    pub fn has_queued_updates(&self) -> bool {
        !self.notifier.1.is_empty() || !self.update_queue.read().unwrap().is_empty()
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
        let buffer = self.buffer_manager.commit();

        let writes = self.drain_updated_sources();
        if !writes.is_empty() {
            self.buffer_manager
                .runtime()
                .buffer_write(writes.ranges.into_iter(), &buffer);
        }

        buffer
    }
}
