//! Slab allocators that run on the CPU.
use core::sync::atomic::{AtomicUsize, Ordering};
use crabslab::{Array, Id, SlabItem};
use rustc_hash::{FxHashMap, FxHashSet};
use snafu::prelude::*;
use std::{
    hash::Hash,
    ops::Deref,
    sync::{atomic::AtomicBool, Arc, RwLock},
};

use crate::{
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

/// A thin wrapper around a buffer `T` that provides the ability to tell
/// if the buffer has been invalidated by the [`SlabAllocator`] that it
/// originated from.
///
/// Invalidation happens when the slab resizes. For this reason it is
/// important to create as many values as necessary _before_ calling
/// [`SlabAllocator::commit`] to avoid unnecessary invalidation.
pub struct SlabBuffer<T> {
    // Id of the slab's last `commit` invocation.
    slab_commit_invocation_k: Arc<AtomicUsize>,
    // Id of the slab's last buffer invalidation.
    slab_invalidation_k: Arc<AtomicUsize>,
    // The slab's `slab_update_k` at the time of this buffer's creation.
    buffer_creation_k: usize,
    // The buffer created at `buffer_creation_k`
    buffer: Arc<T>,
    // The buffer the source slab is currently working with
    source_slab_buffer: Arc<RwLock<Option<SlabBuffer<T>>>>,
}

impl<T> Clone for SlabBuffer<T> {
    fn clone(&self) -> Self {
        Self {
            slab_commit_invocation_k: self.slab_commit_invocation_k.clone(),
            slab_invalidation_k: self.slab_invalidation_k.clone(),
            buffer_creation_k: self.buffer_creation_k,
            buffer: self.buffer.clone(),
            source_slab_buffer: self.source_slab_buffer.clone(),
        }
    }
}

impl<T> Deref for SlabBuffer<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.buffer
    }
}

impl<T> SlabBuffer<T> {
    fn new(
        invalidation_k: Arc<AtomicUsize>,
        invocation_k: Arc<AtomicUsize>,
        buffer: T,
        source_slab_buffer: Arc<RwLock<Option<SlabBuffer<T>>>>,
    ) -> Self {
        SlabBuffer {
            buffer: buffer.into(),
            buffer_creation_k: invalidation_k.load(std::sync::atomic::Ordering::Relaxed),
            slab_invalidation_k: invalidation_k,
            slab_commit_invocation_k: invocation_k,
            source_slab_buffer,
        }
    }

    pub(crate) fn invalidation_k(&self) -> usize {
        self.slab_invalidation_k
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    pub(crate) fn invocation_k(&self) -> usize {
        self.slab_commit_invocation_k
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Returns the timestamp at which the internal buffer was created.
    ///
    /// The returned timestamp is not a unix timestamp. It is a
    /// monotonically increasing count of buffer invalidations.
    pub fn creation_time(&self) -> usize {
        self.buffer_creation_k
    }

    /// Determines whether this buffer has been invalidated by the slab
    /// it originated from.
    pub fn is_invalid(&self) -> bool {
        self.creation_time() < self.invalidation_k()
    }

    /// Determines whether this buffer has been invalidated by the slab
    /// it originated from.
    pub fn is_valid(&self) -> bool {
        !self.is_invalid()
    }

    /// Returns `true` when the slab's internal buffer has been recreated, and this is that
    /// newly created buffer.
    ///
    /// This will return false if [`SlabAllocator::commit`] has been called since the creation
    /// of this buffer.
    ///
    /// Typically this function is used by structs that own the [`SlabAllocator`]. These owning
    /// structs will call [`SlabAllocator::commit`] which returns a [`SlabBuffer`]. The callsite
    /// can then call [`SlabBuffer::is_new_this_commit`] to determine if any
    /// downstream resources (like bindgroups) need to be recreated.
    ///
    /// This pattern keeps the owning struct from having to also store the `SlabBuffer`.
    pub fn is_new_this_commit(&self) -> bool {
        self.invocation_k() == self.buffer_creation_k
    }

    #[deprecated(since = "0.1.5", note = "please use `is_new_this_commit` instead")]
    pub fn is_new_this_upkeep(&self) -> bool {
        self.is_new_this_commit()
    }

    /// Syncronize the buffer with the slab's internal buffer.
    ///
    /// This checks to ensure that the internal buffer is the one the slab is working with,
    /// and updates it if the slab is working with a newer buffer.
    ///
    /// Returns `true` if the buffer was updated.
    /// Returns `false` if the buffer remains the same.
    ///
    /// Use the result of this function to invalidate any bind groups or other downstream
    /// resources.
    ///
    /// ## Note
    /// Be cautious when using this function with multiple buffers to invalidate downstream
    /// resources. Keep in mind that using the
    /// [lazy boolean operators](https://doc.rust-lang.org/reference/expressions/operator-expr.html#lazy-boolean-operators)
    /// might not have the effect you are expecting!
    ///
    /// For example:
    ///
    /// ```rust,no_run
    /// use craballoc::prelude::*;
    ///
    /// let buffer_a: SlabBuffer<wgpu::Buffer> = todo!();
    /// let buffer_b: SlabBuffer<wgpu::Buffer> = todo!();
    /// let buffer_c: SlabBuffer<wgpu::Buffer> = todo!();
    ///
    /// let should_invalidate = buffer_a.update_if_invalid()
    ///     || buffer_b.update_if_invalid()
    ///     || buffer_c.update_if_invalid();
    /// ```
    ///
    /// If `buffer_a` is invalid, neither `buffer_b` nor `buffer_c` will be synchronized, because
    /// `||` is lazy in its parameter evaluation.
    ///
    /// Instead, we should write the following:
    ///
    /// ```rust,no_run
    /// use craballoc::prelude::*;
    ///
    /// let buffer_a: SlabBuffer<wgpu::Buffer> = todo!();
    /// let buffer_b: SlabBuffer<wgpu::Buffer> = todo!();
    /// let buffer_c: SlabBuffer<wgpu::Buffer> = todo!();
    ///
    /// let buffer_a_invalid = buffer_a.update_if_invalid();
    /// let buffer_b_invalid = buffer_b.update_if_invalid();
    /// let buffer_c_invalid = buffer_c.update_if_invalid();
    ///
    /// let should_invalidate = buffer_a_invalid || buffer_b_invalid || buffer_c_invalid;
    /// ```
    pub fn update_if_invalid(&mut self) -> bool {
        if self.is_invalid() {
            // UNWRAP: Safe because it is an invariant of the system. Once the `SlabBuffer`
            // is created, source_slab_buffer will always be Some.
            let updated_buffer = {
                let guard = self.source_slab_buffer.read().unwrap();
                guard.as_ref().unwrap().clone()
            };
            debug_assert!(updated_buffer.is_valid());
            *self = updated_buffer;
            true
        } else {
            false
        }
    }

    #[deprecated(since = "0.1.5", note = "please use `update_if_invalid` instead")]
    pub fn synchronize(&mut self) -> bool {
        self.update_if_invalid()
    }
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
    runtime: Runtime,
    label: Arc<String>,
    len: Arc<AtomicUsize>,
    capacity: Arc<AtomicUsize>,
    needs_expansion: Arc<AtomicBool>,
    buffer: Arc<RwLock<Option<SlabBuffer<Runtime::Buffer>>>>,
    buffer_usages: Runtime::BufferUsages,
    // The value of invocation_k when the last buffer invalidation happened
    invalidation_k: Arc<AtomicUsize>,
    // The next monotonically increasing commit invocation identifier
    invocation_k: Arc<AtomicUsize>,
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
            runtime: self.runtime.clone(),
            notifier: self.notifier.clone(),
            label: self.label.clone(),
            len: self.len.clone(),
            capacity: self.capacity.clone(),
            needs_expansion: self.needs_expansion.clone(),
            buffer: self.buffer.clone(),
            buffer_usages: self.buffer_usages.clone(),
            invalidation_k: self.invalidation_k.clone(),
            invocation_k: self.invocation_k.clone(),
            update_k: self.update_k.clone(),
            update_sources: self.update_sources.clone(),
            update_queue: self.update_queue.clone(),
            recycles: self.recycles.clone(),
        }
    }
}

impl<R: IsRuntime> SlabAllocator<R> {
    pub fn new(
        runtime: impl AsRef<R>,
        name: impl AsRef<str>,
        default_buffer_usages: R::BufferUsages,
    ) -> Self {
        let label = Arc::new(name.as_ref().to_owned());
        Self {
            runtime: runtime.as_ref().clone(),
            label,
            notifier: async_channel::unbounded(),
            update_k: Default::default(),
            update_sources: Default::default(),
            update_queue: Default::default(),
            recycles: Default::default(),
            len: Default::default(),
            // Start with size 1, because some of `wgpu`'s validation depends on it.
            // See <https://github.com/gfx-rs/wgpu/issues/6414> for more info.
            capacity: Arc::new(AtomicUsize::new(1)),
            needs_expansion: Arc::new(true.into()),
            buffer: Default::default(),
            buffer_usages: default_buffer_usages,
            invalidation_k: Default::default(),
            invocation_k: Default::default(),
        }
    }

    pub(crate) fn next_update_k(&self) -> usize {
        self.update_k.fetch_add(1, Ordering::Relaxed)
    }

    pub(crate) fn insert_update_source(&self, id: SourceId, source: WeakGpuRef) {
        log::trace!("{} insert_update_source {id}", self.label);
        let _ = self.notifier.0.try_send(id);
        // UNWRAP: panic on purpose
        self.update_sources.write().unwrap().insert(id, source);
    }

    /// The length of the underlying buffer, in u32 slots.
    ///
    /// This does not include data that has not yet been committed.
    pub fn len(&self) -> usize {
        self.len.load(Ordering::Relaxed)
    }

    /// Whether the underlying buffer is empty.
    ///
    /// This does not include data that has not yet been committed.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
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
        } else {
            self.maybe_expand_to_fit::<T>(1);
            let index = self.increment_len(T::SLAB_SIZE);
            Id::from(index)
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
            let array = Array::<T>::new(range.first_index, len as u32);
            log::trace!(
                "slab allocate_array {len}x{}: dequeued {range:?} to {array:?}",
                std::any::type_name::<T>()
            );
            debug_assert_eq!(
                range.last_index,
                range.first_index + (T::SLAB_SIZE * len) as u32 - 1
            );
            array
        } else {
            self.maybe_expand_to_fit::<T>(len);
            let index = self.increment_len(T::SLAB_SIZE * len);
            Array::new(index as u32, len as u32)
        }
    }

    fn capacity(&self) -> usize {
        self.capacity.load(Ordering::Relaxed)
    }

    fn reserve_capacity(&self, capacity: usize) {
        self.capacity.store(capacity, Ordering::Relaxed);
        self.needs_expansion.store(true, Ordering::Relaxed);
    }

    fn increment_len(&self, n: usize) -> usize {
        self.len.fetch_add(n, Ordering::Relaxed)
    }

    fn maybe_expand_to_fit<T: SlabItem>(&self, len: usize) {
        let capacity = self.capacity();
        // log::trace!(
        //    "append_slice: {size} * {ts_len} + {len} ({}) >= {capacity}",
        //    size * ts_len + len
        //);
        let capacity_needed = self.len() + T::SLAB_SIZE * len;
        if capacity_needed > capacity {
            let mut new_capacity = capacity * 2;
            while new_capacity < capacity_needed {
                new_capacity = (new_capacity * 2).max(2);
            }
            self.reserve_capacity(new_capacity);
        }
    }

    /// Return the internal buffer used by this slab, if it has
    /// been created.
    pub fn get_buffer(&self) -> Option<SlabBuffer<R::Buffer>> {
        self.buffer.read().unwrap().clone()
    }

    /// Recreate the internal buffer, writing the contents of the previous buffer (if it
    /// exists) to the new one, then return the new buffer.
    fn recreate_buffer(&self) -> SlabBuffer<R::Buffer> {
        let new_buffer = self.runtime.buffer_create(
            self.capacity(),
            Some(self.label.as_ref()),
            self.buffer_usages.clone(),
        );
        let mut guard = self.buffer.write().unwrap();
        if let Some(old_buffer) = guard.take() {
            self.runtime
                .buffer_copy(&old_buffer, &new_buffer, Some(self.label.as_ref()));
        }
        let slab_buffer = SlabBuffer::new(
            self.invalidation_k.clone(),
            self.invocation_k.clone(),
            new_buffer,
            self.buffer.clone(),
        );
        *guard = Some(slab_buffer.clone());
        slab_buffer
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
                            self.label
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

    /// Perform upkeep on the slab, synchronizing changes to the internal buffer.
    ///
    /// Changes made to [`Hybrid`] and [`Gpu`](crate::value::Gpu) values created by this slab are not committed
    /// until this function has been called.
    ///
    /// The internal buffer is not created until the first time this function is called.
    ///
    /// Returns a [`SlabBuffer`] wrapping the internal buffer that is currently in use by the allocator.
    pub fn commit(&self) -> SlabBuffer<R::Buffer> {
        let invocation_k = self
            .invocation_k
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            + 1;
        let buffer = if self.needs_expansion.swap(false, Ordering::Relaxed) {
            self.invalidation_k
                .store(invocation_k, std::sync::atomic::Ordering::Relaxed);
            self.recreate_buffer()
        } else {
            // UNWRAP: Safe because we know it exists or else it would need expansion
            self.get_buffer().unwrap()
        };
        let writes = self.drain_updated_sources();
        if !writes.is_empty() {
            self.runtime
                .buffer_write(writes.ranges.into_iter(), &buffer);
        }
        buffer
    }

    #[deprecated(since = "0.1.5", note = "please use `commit` instead")]
    pub fn upkeep(&self) -> SlabBuffer<R::Buffer> {
        self.commit()
    }

    /// Defragments the internal "recycle" buffer.
    pub fn defrag(&self) {
        // UNWRAP: panic on purpose
        let mut recycle_guard = self.recycles.write().unwrap();
        for range in std::mem::take(&mut recycle_guard.ranges) {
            recycle_guard.add_range(range);
        }
    }

    pub fn runtime(&self) -> &R {
        &self.runtime
    }
}
