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
    num::NonZeroU32,
    ops::{Deref, DerefMut},
    sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard},
};

use crabslab::{Array, Id, Slab, SlabItem};

use crate::{
    buffer::{manager::BumpAllocator, SlabBuffer},
    range::{Range, RangeManager},
    runtime::IsRuntime,
    update::{
        CpuUpdateSource, SourceId, SynchronizationData, Update, UpdateManager, UpdateSummary,
    },
    Error,
};

pub trait CanUpdateFromCpu {}

/// Value is synchronized on both CPU and GPU.
pub struct Bidirectional;
impl CanUpdateFromCpu for Bidirectional {}

// Value is synchronized from CPU to the GPU, but any changes on the GPU will
// leave the value out of sync.
pub struct OneWayFromCpu;
impl CanUpdateFromCpu for OneWayFromCpu {}

// Value is synchronized from GPU to CPU but cannot be changed from the CPU.
pub struct OneWayFromGpu;

/// A read lock on a [`Value<T>`].
///
/// This is useful for preventing writes to the locked `Value` for the duration
/// of the scope.
pub struct ValueReadGuard<'a, T> {
    guard: RwLockReadGuard<'a, SynchronizationData>,
    value: T,
}

impl<'a, T> Deref for ValueReadGuard<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

/// A write lock on a [`Value<T>`].
pub struct ValueWriteGuard<'a, T: SlabItem> {
    _guard: RwLockWriteGuard<'a, SynchronizationData>,
    value: T,
}

impl<'a, T: SlabItem> Deref for ValueWriteGuard<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

impl<'a, T: SlabItem> DerefMut for ValueWriteGuard<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.value
    }
}

impl<'a, T: SlabItem> Drop for ValueWriteGuard<'a, T> {
    fn drop(&mut self) {
        self._guard.replace(self.value.slab_data());
    }
}

pub struct Value<T: ?Sized, Sync = Bidirectional> {
    update_source: CpuUpdateSource,
    /// A channell to send updates into.
    _phantom: PhantomData<(Sync, T)>,
}

impl<T: ?Sized, Sync> Value<T, Sync> {
    pub(crate) fn ref_count(&self) -> usize {
        self.update_source.ref_count()
    }
}

impl<T: Default + SlabItem + Sized, Sync: CanUpdateFromCpu> Value<T, Sync> {
    /// Modify the inner value, queing an update to the GPU.
    pub fn modify<X>(&self, f: impl FnOnce(&mut T) -> X) -> X {
        self.update_source.modify(|data| {
            let mut t = data.read(Id::<T>::ZERO);
            let x = f(&mut t);
            data.write(Id::ZERO, &t);
            x
        })
    }
}

impl<T: std::fmt::Debug + Clone + Default + SlabItem + Sized, Sync: CanUpdateFromCpu>
    Value<[T], Sync>
{
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
            let array = Array::new(Id::ZERO, len);
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

    pub fn modify_item<X>(&self, index: u32, f: impl FnOnce(&mut T) -> X) -> Option<X> {
        (index < self.len() as u32).then_some(())?;
        Some(self.modify_range(index..=index, |ts| f(&mut ts[0])))
    }

    pub fn read_item<X>(&self, index: u32, f: impl FnOnce(&T) -> X) -> Option<X> {
        (index < self.len() as u32).then_some(())?;
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
/// TODO: write about how values allocated from the arena can conditionally be
/// synchronized from the CPU to the GPU and back.
#[derive(Debug)]
pub struct Arena<R: IsRuntime> {
    /// Manages bump allocation on the slab.
    bump_allocator: BumpAllocator<R>,
    /// Manages CPU side updates to the slab after allocation.
    update_manager: UpdateManager,
    /// Manages recycled value ranges.
    recycle_ranges: Arc<RwLock<RangeManager<Range>>>,
}

impl<Runtime: IsRuntime> Clone for Arena<Runtime> {
    fn clone(&self) -> Self {
        Self {
            bump_allocator: self.bump_allocator.clone(),
            update_manager: self.update_manager.clone(),
            recycle_ranges: self.recycle_ranges.clone(),
        }
    }
}

impl<R: IsRuntime> Arena<R> {
    /// Create a new arena.
    pub fn new(
        runtime: impl AsRef<R>,
        label: impl Into<Cow<'static, str>>,
        default_buffer_usages: R::BufferUsages,
    ) -> Self {
        Self {
            bump_allocator: BumpAllocator::new(runtime, label, default_buffer_usages),
            update_manager: UpdateManager::default(),
            recycle_ranges: Default::default(),
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
    pub(crate) fn recycle_ranges(&self) -> impl Deref<Target = RangeManager<Range>> + '_ {
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
        let update_source = self.new_update_source(value.slab_data(), std::any::type_name::<T>());
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

    #[cfg(test)]
    pub async fn read_slab<T: SlabItem>(&self, array: Array<T>) -> Result<Vec<T>, Error> {
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
        for id in array.iter() {
            output.push(data.read_unchecked(id));
        }
        Ok(output)
    }

    /// Returns an iterator of [`SourceId`]s of all currently live [`Value`]s.
    ///
    /// ## Note
    /// Keep in mind that the returned [`SourceId`]s are only valid until the next
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
            gpu_update_ranges,
            recycle_ranges: summary_recycle_ranges,
        } = self.update_manager.clear_updated_sources();
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

        for Update { range, data } in cpu_update_ranges.ranges {
            log::trace!("writing range {range:?} to buffer");
            debug_assert_eq!(range.len(), data.len() as u32);
            self.bump_allocator
                .runtime()
                .buffer_write(&buffer, range.into(), &data);
        }

        buffer
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn range_full() {
        let arena = Arena::new(crate::wgpu_runtime(), "tests", wgpu::BufferUsages::empty());
        let values = arena.new_array(0u32..10);
        assert_eq!(10, values.len());
    }
}
