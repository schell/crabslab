//! An arena allocator built on a `u32` slab.
//!
//! [`Arena`] provides an API to dynamically allocate types that
//! implement [`SlabItem`] from the CPU, and then synchronize those changes to
//! the backend during [`Arena::commit`].
//! [`Arena`] also provides synchronization _back_ from the backend using
//! [`Arena::sychronize`]. All values are set to synchronize from the backend by
//! default and must be explicitly set to opt-out of synchronization with `TODO:
//! add sync opt-out`.
use std::{
    borrow::Cow,
    marker::PhantomData,
    num::NonZeroU32,
    sync::{Arc, RwLock},
};

use crabslab::{slab_read, slab_write, SlabItem};
use snafu::OptionExt;

use crate::{
    buffer::{manager::BumpAllocator, SlabBuffer},
    range::{Range, RangeManager},
    runtime::IsRuntime,
    update::{CpuUpdateSource, GpuUpdates, SourceId, Update, UpdateManager, UpdateSummary},
    Error, NoInternalBufferSnafu,
};

/// Read `count` items of type `T` from a `u32` slice, starting at offset 0.
fn slab_read_vec<T: SlabItem>(slab: &[u32], count: usize) -> Vec<T> {
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        out.push(slab_read::<T>(slab, i * T::SLAB_SIZE));
    }
    out
}

/// Write a slice of `T` items into a `u32` slice, starting at offset 0.
fn slab_write_slice<T: SlabItem>(slab: &mut [u32], items: &[T]) {
    for (i, item) in items.iter().enumerate() {
        slab_write(slab, i * T::SLAB_SIZE, item);
    }
}

pub trait CanUpdateFromCpu {}

/// Value is automatically synchronized from CPU to GPU on commit and back on
/// sync.
pub struct SyncBidirectional;
impl CanUpdateFromCpu for SyncBidirectional {}

/// Value is synchronized from CPU to the GPU on commit, but changes on the GPU
/// do not roundtrip automatically.
pub struct SyncOneWayFromCpu;
impl CanUpdateFromCpu for SyncOneWayFromCpu {}

/// Value is automatically synchronized from GPU to CPU.
pub struct SyncOneWayFromGpu;

/// Value is not updated.
pub struct SyncNone;

pub struct Value<T: ?Sized, Sync = SyncBidirectional> {
    update_source: CpuUpdateSource,
    /// A channell to send updates into.
    _phantom: PhantomData<(Sync, T)>,
}

impl<T: ?Sized, Sync> Value<T, Sync> {
    #[cfg(test)]
    pub(crate) fn ref_count(&self) -> usize {
        self.update_source.ref_count()
    }

    /// Returns the u32 [`Range`] that this value occupies on the slab.
    pub fn slab_range(&self) -> Range {
        self.update_source.source_id().range
    }

    /// Convert to bidirectional sync (CPU→GPU on commit, GPU→CPU on
    /// synchronize).
    pub fn into_bidirectional(self) -> Value<T, SyncBidirectional> {
        self.update_source.set_gpu_sync(true);
        Value {
            update_source: self.update_source,
            _phantom: PhantomData,
        }
    }

    /// Convert to one-way CPU→GPU sync. Changes made via [`Value::modify`]
    /// and [`Value::set`] are sent to the GPU on commit, but GPU changes
    /// are **not** read back on synchronize.
    pub fn into_one_way_from_cpu(self) -> Value<T, SyncOneWayFromCpu> {
        self.update_source.set_gpu_sync(false);
        Value {
            update_source: self.update_source,
            _phantom: PhantomData,
        }
    }

    /// Convert to one-way GPU→CPU sync. The GPU value is read back into
    /// the CPU cache on synchronize, but CPU-side modification methods
    /// ([`Value::modify`], [`Value::set`]) are not available.
    pub fn into_one_way_from_gpu(self) -> Value<T, SyncOneWayFromGpu> {
        self.update_source.set_gpu_sync(true);
        Value {
            update_source: self.update_source,
            _phantom: PhantomData,
        }
    }

    /// Convert to no sync. The value occupies slab space but is neither
    /// written to the GPU on commit nor read back on synchronize.
    pub fn into_sync_none(self) -> Value<T, SyncNone> {
        self.update_source.set_gpu_sync(false);
        Value {
            update_source: self.update_source,
            _phantom: PhantomData,
        }
    }
}

impl<T: ?Sized, S> std::fmt::Debug for Value<T, S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct(&format!(
            "Value<{}, {}>",
            std::any::type_name::<T>(),
            std::any::type_name::<S>()
        ))
        .field("slab_range", &self.slab_range())
        .finish()
    }
}

impl<T: ?Sized, S> Clone for Value<T, S> {
    fn clone(&self) -> Self {
        Self {
            update_source: self.update_source.clone(),
            _phantom: PhantomData,
        }
    }
}

impl<T: Default + SlabItem + Sized, Sync: CanUpdateFromCpu> Value<T, Sync> {
    /// Return the raw `u32` slab index of this value.
    pub fn id(&self) -> u32 {
        self.update_source.source_id().range.first_index
    }

    /// Return the `(index, len)` pair describing this value's slab region.
    pub fn array(&self) -> (u32, u32) {
        (self.id(), 1)
    }

    /// Modify the inner value, queing an update to the GPU.
    pub fn modify<X>(&self, f: impl FnOnce(&mut T) -> X) -> X {
        self.update_source.modify(|data| {
            let mut t = slab_read::<T>(data, 0);
            let x = f(&mut t);
            slab_write(data, 0, &t);
            x
        })
    }

    /// Set the inner value, queing the update to the GPU.
    pub fn set(&self, t: T) {
        self.modify(|data| *data = t);
    }

    /// Get a copy of the value from the local CPU cache.
    pub fn get(&self) -> T {
        self.update_source.read(|data| slab_read::<T>(data, 0))
    }
}

impl<T: Clone + Default + SlabItem + Sized, Sync: CanUpdateFromCpu> Value<[T], Sync> {
    /// The length of the inner array.
    pub fn len(&self) -> usize {
        let u32_size = self.update_source.source_id().range.len() as usize;
        u32_size / T::SLAB_SIZE
    }

    /// Returns whether this array of values is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Sanitizes the range to ensure that it fits within the bounds of these
    /// values.
    fn sanitize_range(&self, range: impl Into<Range>) -> Range {
        let total_len = self.len() as u32;
        let mut range = range.into();
        range.last_index = range.last_index.min(total_len.max(1) - 1);
        range
    }

    /// Modify a range inner values, queing an update to the GPU.
    pub fn modify_range<X>(&self, range: impl Into<Range>, f: impl FnOnce(&mut [T]) -> X) -> X {
        let input_range = self.sanitize_range(range);
        let len = input_range.len();
        let first_index = input_range.first_index * T::SLAB_SIZE as u32;
        let last_index = first_index + len * T::SLAB_SIZE as u32 - 1;
        let range = Range {
            first_index,
            last_index,
        };

        self.update_source.modify_range(range, |data| {
            let mut s = slab_read_vec::<T>(data, len as usize);
            let x = f(&mut s);
            slab_write_slice(data, &s);
            x
        })
    }

    /// Read a range of the inner value.
    pub fn read_range<X>(&self, range: impl Into<Range>, f: impl FnOnce(&[T]) -> X) -> X {
        let input_range = self.sanitize_range(range);
        let start = input_range.first_index as usize * T::SLAB_SIZE;
        let count = input_range.len() as usize;
        self.update_source.read(|data| {
            let sub = &data[start..start + count * T::SLAB_SIZE];
            let mut s = slab_read_vec::<T>(sub, count);
            f(&mut s)
        })
    }

    pub fn modify_item<X>(&self, index: usize, f: impl FnOnce(&mut T) -> X) -> Option<X> {
        (index < self.len()).then_some(())?;
        Some(self.modify_range(index..=index, |ts| f(&mut ts[0])))
    }

    pub fn read_item<X>(&self, index: usize, f: impl FnOnce(&T) -> X) -> Option<X> {
        (index < self.len()).then_some(())?;
        Some(self.read_range(index..=index, |ts| f(&ts[0])))
    }

    /// Returns a copy of the array items.
    pub fn get_vec(&self) -> Vec<T> {
        self.read_range(.., |data| data.to_vec())
    }

    /// Returns the `(index, len)` pair describing this array's slab region.
    pub fn array(&self) -> (u32, u32) {
        let first_index = self.update_source.source_id().range.first_index;
        let len = self.len() as u32;
        (first_index, len)
    }

    /// Returns the ranges that have been updated since last commit.
    pub fn updated_ranges(&self) -> Vec<Range> {
        self.update_source.updated_ranges()
    }
}

/// An arena allocator backed by a `u32` slab.
///
/// TODO: write about how values allocated from the arena can conditionally be
/// synchronized from the CPU to the GPU and back.
pub struct Arena<R: IsRuntime> {
    /// Manages bump allocation on the slab.
    bump_allocator: BumpAllocator<R>,
    /// Manages CPU side updates to the slab after allocation.
    update_manager: UpdateManager,
    /// Manages recycled value ranges.
    recycle_ranges: Arc<RwLock<RangeManager<Range>>>,
    /// Tracks GPU update requests from the CPU, and
    /// actual update data read from the GPU.
    ///
    /// This also _applies_ the updates to the sources it contains.
    gpu_updates: Arc<RwLock<GpuUpdates>>,
}

impl<Runtime: IsRuntime> Clone for Arena<Runtime> {
    fn clone(&self) -> Self {
        Self {
            bump_allocator: self.bump_allocator.clone(),
            update_manager: self.update_manager.clone(),
            recycle_ranges: self.recycle_ranges.clone(),
            gpu_updates: self.gpu_updates.clone(),
        }
    }
}

impl<R: IsRuntime> Arena<R> {
    /// Create a new arena.
    pub fn new(
        runtime: &R,
        label: impl Into<Cow<'static, str>>,
        buffer_usages: Option<R::BufferUsages>,
    ) -> Self {
        Self {
            bump_allocator: BumpAllocator::new(runtime, label, buffer_usages),
            update_manager: UpdateManager::default(),
            recycle_ranges: Default::default(),
            gpu_updates: Default::default(),
        }
    }

    #[cfg(test)]
    /// Allocate new space on the internal buffer.
    ///
    /// Just for tests.
    pub(crate) fn allocate<T: SlabItem>(&self) -> Range {
        self.bump_allocator
            // UNWRAP: Ok because this is during tests
            .alloc(NonZeroU32::new(T::SLAB_SIZE as u32).unwrap())
    }

    #[cfg(test)]
    /// Returns the recycle ranges.
    pub(crate) fn recycle_ranges(&self) -> impl std::ops::Deref<Target = RangeManager<Range>> + '_ {
        self.recycle_ranges.read().unwrap()
    }

    #[cfg(test)]
    /// Returns the number of recycled buffer spaces available.
    ///
    /// These are not necessarily contiguous spaces.
    pub(crate) fn recycle_spaces(&self) -> u32 {
        self.recycle_ranges
            .read()
            .unwrap()
            .ranges
            .iter()
            .map(|range| range.len())
            .sum()
    }

    #[cfg(test)]
    /// Returns the number of contiguous recycled buffer ranges.
    pub(crate) fn contiguous_recycle_ranges(&self) -> u32 {
        self.recycle_ranges.read().unwrap().ranges.len() as u32
    }

    #[cfg(test)]
    /// The underlying buffer's creation time.
    pub(crate) fn buffer_creation_time(&self) -> usize {
        self.bump_allocator.buffer_creation_time()
    }

    /// Deque space from recycle ranges or alloc new space for values.
    fn new_update_source(&self, data: Vec<u32>, ty: &'static str) -> CpuUpdateSource {
        if let Some(spaces) = NonZeroU32::new(data.len() as u32) {
            let maybe_recycled_range = self.recycle_ranges.write().unwrap().remove(spaces.into());
            let range = maybe_recycled_range.unwrap_or_else(|| self.bump_allocator.alloc(spaces));
            let update_source = self
                .update_manager
                .new_update_source(SourceId { range, type_is: ty });
            update_source.modify(|s| {
                s.copy_from_slice(&data);
            });
            update_source
        } else {
            CpuUpdateSource::null()
        }
    }

    /// Allocate a new value from the arena.
    ///
    /// This value has bidirectional synchonization.
    pub fn new_value<T: SlabItem>(&self, value: T) -> Value<T> {
        let update_source = self.new_update_source(
            value.to_array().as_ref().to_vec(),
            std::any::type_name::<T>(),
        );
        Value {
            update_source,
            _phantom: PhantomData,
        }
    }

    /// Allocate a new array of values from the arena.
    ///
    /// These values have bidirectional synchronization.
    pub fn new_array<T: SlabItem>(&self, values: impl IntoIterator<Item = T>) -> Value<[T]> {
        let data = values.into_iter().fold(vec![], |mut acc, value| {
            acc.extend(value.to_array().as_ref());
            acc
        });
        let update_source = self.new_update_source(data, std::any::type_name::<[T]>());
        Value {
            update_source,
            _phantom: PhantomData,
        }
    }

    /// Returns whether any values have queued updates waiting to be committed.
    pub fn has_queued_updates(&self) -> bool {
        self.update_manager.has_queued_updates()
    }

    /// Get the working internal buffer, if any.
    pub(crate) fn get_buffer(&self) -> Option<SlabBuffer<R::Buffer>> {
        self.bump_allocator.get_buffer()
    }

    #[cfg(test)]
    pub async fn read_slab<T: SlabItem>(&self, array: (u32, u32)) -> Result<Vec<T>, crate::Error> {
        let (index, len) = array;
        let buffer = self.commit();
        let buffer_len = self.bump_allocator.capacity();
        let u32_count = len as usize * T::SLAB_SIZE;
        let range = index as usize..index as usize + u32_count;
        let data = self
            .bump_allocator
            .runtime()
            .buffer_read(&buffer, buffer_len as usize, range)
            .await?;
        let mut output = Vec::with_capacity(len as usize);
        for i in 0..len as usize {
            output.push(slab_read::<T>(&data, i * T::SLAB_SIZE));
        }
        Ok(output)
    }

    /// Returns an iterator of [`SourceId`]s of all currently live [`Value`]s.
    ///
    /// ## Note
    /// Keep in mind that the returned [`SourceId`]s are only valid until the
    /// next commit.
    pub fn get_live_source_ids(&self) -> impl Iterator<Item = SourceId> {
        self.update_manager.get_managed_source_ids().into_iter()
    }

    /// Perform upkeep on the slab, synchronizing changes to the internal
    /// buffer.
    ///
    /// Changes made to allocated values stored on the underlying slab are not
    /// committed until this function has been called, and sometimes not until
    /// the runtime has finished synchronizing the CPU to the GPU (eg with
    /// `wgpu::Device::poll`).
    ///
    /// The internal buffer is not created until the after the first time this
    /// function is called.
    ///
    /// Returns a [`SlabBuffer`] wrapping the internal buffer that is currently
    /// in use by the arena.
    pub fn commit(&self) -> SlabBuffer<R::Buffer> {
        log::trace!("arena commit");
        let buffer = self.bump_allocator.commit();
        let UpdateSummary {
            cpu_update_ranges,
            gpu_updates,
            recycle_ranges: summary_recycle_ranges,
        } = self.update_manager.clear_updated_sources();
        log::trace!("  saving gpu requests: {gpu_updates:?}");
        *self.gpu_updates.write().unwrap() = gpu_updates;
        log::trace!("got summary");
        {
            // Add the dropped sources to the recycle pool
            log::trace!("  recycling ranges: {summary_recycle_ranges:?}");
            let mut recycle_ranges = self.recycle_ranges.write().unwrap();
            for source_id in summary_recycle_ranges.ranges.into_iter() {
                recycle_ranges.insert(source_id.range);
            }
            log::trace!("done recycling");
        }

        for update in cpu_update_ranges.ranges {
            log::trace!("writing update {update:?} to buffer");
            debug_assert_eq!(update.range.len(), update.data.len() as u32);
            self.bump_allocator
                .runtime()
                .buffer_write(&buffer, update.range.into(), &update.data);
        }

        buffer
    }

    /// Synchronize CPU-cached values with the backend.
    ///
    /// This reads portions of the backend slab and writes them back to their
    /// CPU cache sources.
    ///
    /// ## Errors
    /// Errs if
    /// * [`Arena::commit`] has not been called, and there is no internal buffer
    /// * there is a problem reading data from the backend
    pub async fn synchronize(&self) -> Result<(), Error> {
        log::trace!("synchronizing backend to CPU caches");
        let buffer = self.get_buffer().context(NoInternalBufferSnafu)?;
        let buffer_len = self.bump_allocator.capacity() as usize;

        // Read each range from the GPU and collect it.
        let mut updates = vec![];
        let ranges = self.gpu_updates.read().unwrap().ranges();
        log::trace!("  ranges: {ranges:?}");
        for range in ranges {
            let runtime = self.bump_allocator.runtime();
            let data = runtime
                .buffer_read(&buffer, buffer_len, std::ops::Range::from(range))
                .await?;
            let update = Update { range, data };
            updates.push(update);
        }

        if updates.is_empty() {
            log::trace!("  no sources have requested synchronization, skipping");
        } else {
            let count = updates.len();
            let plur = if updates.len() == 1 { "" } else { "s" };
            log::trace!("  synchronizing {count} update{plur}");
            // Apply the updates to the CPU sources.
            let mut gpu_updates = self.gpu_updates.write().unwrap();
            gpu_updates.apply(updates);
        }

        Ok(())
    }

    /// Return the runtime used by this `Arena`.
    pub fn runtime(&self) -> &R {
        self.bump_allocator.runtime()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::runtime::CpuRuntime;

    #[test]
    fn range_full() {
        let arena = Arena::new(&crate::wgpu_runtime(), "tests", None);
        let values = arena.new_array(0u32..10);
        assert_eq!(10, values.len());
    }

    #[test]
    fn value_sync_conversion() {
        let arena = Arena::new(&CpuRuntime, "test", None);
        let v = arena.new_value(42u32);
        assert_eq!(42, v.get());

        // Bidirectional → one-way-from-cpu: can still modify/get
        let v = v.into_one_way_from_cpu();
        v.set(100);
        assert_eq!(100, v.get());

        // One-way-from-cpu → one-way-from-gpu: read slab_range still works
        let v = v.into_one_way_from_gpu();
        assert_eq!(v.slab_range().len(), 1);

        // One-way-from-gpu → sync-none
        let v = v.into_sync_none();
        assert_eq!(v.slab_range().len(), 1);

        // Sync-none → back to bidirectional: can modify/get again
        let v = v.into_bidirectional();
        v.set(200);
        assert_eq!(200, v.get());
    }
}
