//! An arena allocator built on a `u32` slab.
//!
//! [`Arena`] provides an API to dynamically allocate types that
//! implement [`SlabItem`] from the CPU, and then synchronize those changes to the backend
//! during [`Arena::commit`].
//! [`Arena`] also provides synchronization _back_ from the backend using [`Arena::synchronize`].
//! All values synchronize from the backend by default; opt out with [`Value::into_sync_none`]
//! or pick another direction with the other `into_*` methods.
use std::{
    borrow::Cow,
    marker::PhantomData,
    num::NonZeroU32,
    sync::{Arc, RwLock},
};

use crabslab::{Array, CpuSlabItem, Id, Slab, SlabItem};
use snafu::OptionExt;

use crate::{
    buffer::{manager::BumpAllocator, SlabBuffer},
    range::{Range, RangeManager},
    runtime::IsRuntime,
    update::{CpuUpdateSource, GpuUpdates, SourceId, Update, UpdateManager, UpdateSummary},
    Error, NoInternalBufferSnafu,
};

/// Marker for the sync directions whose values queue CPU edits for the
/// GPU.
pub trait CanUpdateFromCpu {}

/// Value is automatically synchronized from CPU to GPU on commit and back on sync.
pub struct SyncBidirectional;
impl CanUpdateFromCpu for SyncBidirectional {}

/// Value is synchronized from CPU to the GPU on commit, but changes on the GPU do
/// not roundtrip automatically.
pub struct SyncOneWayFromCpu;
impl CanUpdateFromCpu for SyncOneWayFromCpu {}

/// Value is automatically synchronized from GPU to CPU.
pub struct SyncOneWayFromGpu;

/// Value is not updated.
pub struct SyncNone;

/// A RAII slab value.
///
/// A `Value<T>` is a handle to a `T` allocated on an [`Arena`]: it
/// reads and edits the local CPU cache, queues edits for the GPU, and -
/// depending on its sync direction - pulls GPU changes back into the
/// cache during [`Arena::synchronize`].
///
/// The `Sync` type parameter selects the synchronization direction and
/// defaults to [`SyncBidirectional`]. Direction can only be changed
/// through the value's sole live handle: the `into_*` methods fail
/// while clones exist, returning the original, so every live handle's
/// type always matches the arena's registration. (The deprecated
/// [`Value::set_gpu_sync`] escape hatch predates the typed API and can
/// still flip the registration without changing the type.)
///
/// The uniqueness check is best-effort and assumes single-threaded use:
/// a `clone()` racing the check from another thread can slip through,
/// and a commit or synchronize in progress briefly holds extra strong
/// references, so the count can over-report - the guard then fails
/// safe, refusing an otherwise valid conversion.
pub struct Value<T: ?Sized, Sync = SyncBidirectional> {
    update_source: CpuUpdateSource,
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
    ///
    /// # Errors
    ///
    /// Fails while other handles to this value exist, returning the
    /// original value unchanged. Changing direction re-registers the
    /// shared source, so it is only allowed through the value's sole
    /// live handle.
    pub fn into_bidirectional(self) -> Result<Value<T, SyncBidirectional>, Value<T, Sync>> {
        if self.update_source.handle_count() != 1 {
            return Err(self);
        }
        self.update_source.set_gpu_sync(true);
        Ok(Value {
            update_source: self.update_source,
            _phantom: PhantomData,
        })
    }

    /// Convert to one-way CPU→GPU sync. Changes made via [`Value::modify`]
    /// and [`Value::set`] are sent to the GPU on commit, but GPU changes
    /// are **not** read back on synchronize.
    ///
    /// # Errors
    ///
    /// Fails while other handles to this value exist, returning the
    /// original value unchanged. Changing direction re-registers the
    /// shared source, so it is only allowed through the value's sole
    /// live handle.
    pub fn into_one_way_from_cpu(self) -> Result<Value<T, SyncOneWayFromCpu>, Value<T, Sync>> {
        if self.update_source.handle_count() != 1 {
            return Err(self);
        }
        self.update_source.set_gpu_sync(false);
        Ok(Value {
            update_source: self.update_source,
            _phantom: PhantomData,
        })
    }

    /// Convert to one-way GPU→CPU sync. The GPU value is read back into
    /// the CPU cache on synchronize, but CPU-side modification methods
    /// ([`Value::modify`], [`Value::set`]) are not available.
    ///
    /// # Errors
    ///
    /// Fails while other handles to this value exist, returning the
    /// original value unchanged. Changing direction re-registers the
    /// shared source, so it is only allowed through the value's sole
    /// live handle.
    pub fn into_one_way_from_gpu(self) -> Result<Value<T, SyncOneWayFromGpu>, Value<T, Sync>> {
        if self.update_source.handle_count() != 1 {
            return Err(self);
        }
        self.update_source.set_gpu_sync(true);
        Ok(Value {
            update_source: self.update_source,
            _phantom: PhantomData,
        })
    }

    /// Convert to no sync. The value occupies slab space but is neither
    /// written to the GPU on commit nor read back on synchronize.
    ///
    /// # Errors
    ///
    /// Fails while other handles to this value exist, returning the
    /// original value unchanged. Changing direction re-registers the
    /// shared source, so it is only allowed through the value's sole
    /// live handle.
    pub fn into_sync_none(self) -> Result<Value<T, SyncNone>, Value<T, Sync>> {
        if self.update_source.handle_count() != 1 {
            return Err(self);
        }
        self.update_source.set_gpu_sync(false);
        Ok(Value {
            update_source: self.update_source,
            _phantom: PhantomData,
        })
    }

    /// Sets whether or not this `Value` should synchronize its data from the GPU
    /// when [`Arena::synchronize`] is called.
    ///
    /// If `should_sync` is `true`, this `Value` will synchronize.
    /// If `should_sync` is `false` this `Value` **will not** synchronize.
    ///
    /// Consider using one of the `into_*` functions instead.
    ///
    /// # Errors
    ///
    /// Fails with [`Error::SyncChangeWithClones`] while other handles to
    /// this value exist, since changing direction re-registers the shared
    /// source.
    #[deprecated(
        since = "0.5.0",
        note = "use one of the `into_*` methods, which keep the `Sync` type parameter and the arena's registration in agreement; this escape hatch flips the registration without changing the type"
    )]
    pub fn set_gpu_sync(&self, should_sync: bool) -> Result<(), Error> {
        if self.update_source.handle_count() != 1 {
            return Err(Error::SyncChangeWithClones);
        }
        self.update_source.set_gpu_sync(should_sync);
        Ok(())
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

impl<T: Clone + Default + SlabItem + Sized, Sync: CanUpdateFromCpu> Value<T, Sync> {
    /// Return the [`Id<T>`] that points to this `T` on the slab.
    pub fn id(&self) -> Id<T> {
        Id::new(self.update_source.source_id().range.first_index)
    }

    /// Return the [`Array<T>`] that defines this type's range in the
    /// slab.
    pub fn array(&self) -> Array<T> {
        Array::new(self.id(), 1)
    }

    /// Modify the inner value, queing an update to the GPU.
    pub fn modify<X>(&self, f: impl FnOnce(&mut T) -> X) -> X {
        self.update_source.modify(|data| {
            let mut t = data.read(Id::<T>::zero());
            let x = f(&mut t);
            data.write(Id::zero(), &t);
            x
        })
    }

    /// Set the inner value, queing the update to the GPU.
    pub fn set(&self, t: T) {
        self.modify(|data| *data = t);
    }

    /// Get a copy of the value from the local CPU cache.
    pub fn get(&self) -> T {
        self.update_source
            .read(|data| data.read_unchecked(Id::zero()))
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

    /// Sanitizes the range to ensure that it fits within the bounds of these values.
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
            let array = Array::new(Id::zero(), len);
            let mut s = data.read_vec(array);
            let x = f(&mut s);
            data.write_array(array, &s);
            x
        })
    }

    /// Read a range of the inner value.
    pub fn read_range<X>(&self, range: impl Into<Range>, f: impl FnOnce(&[T]) -> X) -> X {
        let input_range = self.sanitize_range(range);
        let starting_id = Id::<T>::new(input_range.first_index * T::SLAB_SIZE as u32);
        let item_range_array = Array::new(starting_id, input_range.len());
        self.update_source.read(|data| {
            let mut s = data.read_vec(item_range_array);
            f(&mut s)
        })
    }

    /// Modify the item at `index`, returning `None` if it is out of bounds.
    pub fn modify_item<X>(&self, index: usize, f: impl FnOnce(&mut T) -> X) -> Option<X> {
        (index < self.len()).then_some(())?;
        Some(self.modify_range(index..=index, |ts| f(&mut ts[0])))
    }

    /// Read the item at `index`, returning `None` if it is out of bounds.
    pub fn read_item<X>(&self, index: usize, f: impl FnOnce(&T) -> X) -> Option<X> {
        (index < self.len()).then_some(())?;
        Some(self.read_range(index..=index, |ts| f(&ts[0])))
    }

    /// Returns a copy of the array items.
    pub fn get_vec(&self) -> Vec<T> {
        self.read_range(.., |data| data.to_vec())
    }

    /// Returns the [`Array`] that these values occupy in the slab.
    pub fn array(&self) -> Array<T> {
        let len = self.len() as u32;
        Array {
            id: Id::new(self.update_source.source_id().range.first_index),
            len,
        }
    }

    /// Returns the ranges that have been updated since last commit.
    pub fn updated_ranges(&self) -> Vec<Range> {
        self.update_source.updated_ranges()
    }
}

/// An arena allocator backed by a `u32` slab.
///
/// [`Arena`] hands out RAII [`Value`]s backed by one contiguous slab
/// buffer. Each value owns a range of `u32` slots: reads and edits go
/// through its local CPU cache, and the arena batches all communication
/// with the GPU.
///
/// The lifecycle:
///
/// 1. Allocate with [`Arena::new_value`] or [`Arena::new_array`] to get
///    a [`Value<T>`] or a `Value<[T]>`. Values are RAII: when the last
///    handle to a value drops, its slab range is reclaimed into the
///    recycle pool on the next commit.
/// 2. Read and edit the value locally with [`Value::get`], [`Value::set`],
///    and [`Value::modify`]. Edits are queued, not sent immediately.
/// 3. Once per frame (or more, or less), [`Arena::commit`] uploads every
///    queued CPU edit to the GPU in a single pass.
/// 4. [`Arena::synchronize`] copies the ranges of values registered for
///    GPU sync back into their CPU caches, so [`Value::get`] sees
///    GPU-side changes.
///
/// # Conditional synchronization
///
/// Which directions a value synchronizes is part of its type. Values
/// start out [`SyncBidirectional`] - CPU edits go up on commit and GPU
/// changes come back down on synchronize. Direction is changed with the
/// `into_*` methods, which only succeed on the value's sole live
/// handle: while clones exist they fail and return the original, so
/// every live handle's type stays truthful (the deprecated
/// [`Value::set_gpu_sync`] escape hatch excepted):
///
/// * [`Value::into_bidirectional`] synchronizes both ways (the default).
///   CPU-side edits are queued for commit, and GPU changes are read back
///   on synchronize.
/// * [`Value::into_one_way_from_cpu`] uploads CPU edits on commit, but
///   GPU changes are not read back.
/// * [`Value::into_one_way_from_gpu`] reads GPU changes back on
///   synchronize, but the CPU-side edit methods
///   ([`Value::modify`], [`Value::set`]) are unavailable for it;
///   convert back with [`Value::into_bidirectional`] to edit or read
///   the value CPU-side.
/// * [`Value::into_sync_none`] synchronizes neither way; the value only
///   occupies its slab range.
///
/// ## Buffer invalidation
///
/// The slab buffer grows on demand, and growing it invalidates
/// previously returned [`SlabBuffer`]s. Create the values you need
/// _before_ calling [`Arena::commit`] to avoid unnecessary invalidation.
///
/// See the crate documentation for a complete end-to-end example.
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

    /// Returns the number of u32 slots currently backing the arena's slab
    /// buffer, from which it allocates.
    ///
    /// This grows on demand (doubling) whenever an allocation would not fit
    /// within the current buffer, and never shrinks.
    pub fn capacity(&self) -> u32 {
        self.bump_allocator.capacity()
    }

    /// Returns the number of u32 slots ever carved out of the arena's
    /// buffer: a monotonically growing high-water mark.
    ///
    /// Recycled values reuse already-allocated `Range`s and do not increase
    /// this count, so the number of live values may be less than the length
    /// reported here. This is always less than or equal to
    /// [`capacity`](Self::capacity).
    pub fn length(&self) -> u32 {
        self.bump_allocator.length()
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
    pub fn new_value<T: SlabItem + Clone>(&self, value: T) -> Value<T> {
        let update_source = self.new_update_source(value.slab_data(), std::any::type_name::<T>());
        Value {
            update_source,
            _phantom: PhantomData,
        }
    }

    /// Allocate a new array of values from the arena.
    ///
    /// These values have bidirectional synchronization.
    pub fn new_array<T: SlabItem + Clone>(
        &self,
        values: impl IntoIterator<Item = T>,
    ) -> Value<[T]> {
        let data = values.into_iter().fold(vec![], |mut acc, value| {
            acc.extend(value.slab_data());
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

    /// Read every element of `array` back from the GPU. Test-only.
    #[cfg(test)]
    pub async fn read_slab<T: SlabItem + Clone>(
        &self,
        array: Array<T>,
    ) -> Result<Vec<T>, crate::Error> {
        let buffer = self.commit();
        let buffer_len = self.bump_allocator.capacity();
        let u32_array = array.into_u32_array();
        let range = u32_array.starting_index()..(u32_array.starting_index() + u32_array.len());
        let data = self
            .bump_allocator
            .runtime()
            .buffer_read(&buffer, buffer_len as usize, range)
            .await?;
        let mut output = vec![];
        let output_array = Array::new(Id::<T>::zero(), array.len);
        for id in output_array.iter() {
            output.push(data.read_unchecked(id));
        }
        Ok(output)
    }

    /// Returns an iterator of the `SourceId`s of all currently live [`Value`]s.
    ///
    /// ## Note
    /// Keep in mind that the returned ids are only valid until the next
    /// commit.
    pub fn get_live_source_ids(&self) -> impl Iterator<Item = SourceId> {
        self.update_manager.get_managed_source_ids().into_iter()
    }

    /// Perform upkeep on the slab, synchronizing changes to the internal buffer.
    ///
    /// Changes made to allocated values stored on the underlying slab are not
    /// committed until this function has been called, and sometimes not until the runtime
    /// has finished synchronizing the CPU to the GPU (eg with `wgpu::Device::poll`).
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
    /// This reads portions of the backend slab and writes them back to their CPU cache sources.
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
        let v = v.into_one_way_from_cpu().expect("unique handle");
        v.set(100);
        assert_eq!(100, v.get());

        // One-way-from-cpu → one-way-from-gpu: read slab_range still works
        let v = v.into_one_way_from_gpu().expect("unique handle");
        assert_eq!(v.slab_range().len(), 1);

        // One-way-from-gpu → sync-none
        let v = v.into_sync_none().expect("unique handle");
        assert_eq!(v.slab_range().len(), 1);

        // Sync-none → back to bidirectional: can modify/get again
        let v = v.into_bidirectional().expect("unique handle");
        v.set(200);
        assert_eq!(200, v.get());
    }
}
