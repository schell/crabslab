//! Allocated values.

use std::{
    marker::PhantomData,
    ops::{Deref, DerefMut},
    sync::{Arc, Mutex, RwLock, RwLockWriteGuard, Weak as WeakStd},
};

use crabslab::{Array, Id, Slab, SlabItem};

use crate::{
    arena::{OneWayFromCpu, Value, ValueWriteGuard},
    runtime::{IsRuntime, SlabUpdate},
    slab::SlabAllocator,
    update::{CpuUpdateSource, SourceId},
};

/// RAII structure used to update a `[Hybrid<T>]`.
///
/// `HybridWriteGuard<T>` dereferences to `T`.
/// Modifying the dereferenced `T` will queue an update once the guard is dropped.
///
/// The following are equivalent:
///
/// ### Queue an update using a guard
///
/// ```rust
/// use craballoc::prelude::*;
///
/// let slab = SlabAllocator::new(CpuRuntime, "test-slab", ());
/// let value = slab.new_value(42u32);
///
/// {
///     let mut guard = value.lock();
///     *guard += 8;
/// }
///
/// // At this point the update has been queued, and synchronization can occur.
/// slab.commit();
///
/// // Confirm using the raw slab.
/// assert_eq!(&[50], slab.get_buffer().unwrap().as_vec().as_slice());
/// ```
///
/// ### Queue an update using `modify`
///
/// ```rust
/// use craballoc::prelude::*;
///
/// let slab = SlabAllocator::new(CpuRuntime, "test-slab", ());
/// let value = slab.new_value(42u32);
///
/// value.modify(|u| *u += 8);
///
/// // At this point the update has been queued, and synchronization can occur.
/// slab.commit();
///
/// // Confirm using the raw slab.
/// assert_eq!(&[50], slab.get_buffer().unwrap().as_vec().as_slice());
/// ```
pub struct HybridWriteGuard<'a, T: SlabItem + Clone + Send + Sync + 'static> {
    hybrid: &'a Hybrid<T>,
    guard: ValueWriteGuard<'a, T>,
}

impl<T: SlabItem + Clone + Send + Sync + 'static> Deref for HybridWriteGuard<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.guard.deref()
    }
}

impl<T: SlabItem + Clone + Send + Sync + 'static> DerefMut for HybridWriteGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.guard.deref_mut()
    }
}

impl<'a, T: SlabItem + Clone + Send + Sync + 'static> HybridWriteGuard<'a, T> {
    pub fn new(hybrid: &'a Hybrid<T>) -> Self {
        Self {
            hybrid,
            guard: hybrid.gpu_value.inner.write_lock(),
        }
    }
}

/// A "hybrid" type that lives on the CPU and the GPU.
///
/// Updates are syncronized to the GPU at the behest of the
/// `SlabAllocator<T>` that created this value.
///
/// Clones of a hybrid all point to the same CPU and GPU data.
pub struct Hybrid<T> {
    pub(crate) gpu_value: Gpu<T>,
}

impl<T> AsRef<Hybrid<T>> for Hybrid<T> {
    fn as_ref(&self) -> &Hybrid<T> {
        self
    }
}

impl<T: core::fmt::Debug> core::fmt::Debug for Hybrid<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct(&format!("Hybrid<{}>", std::any::type_name::<T>()))
            .field("id", &self.gpu_value.id)
            .field("cpu_value", &self.cpu_value.read().unwrap())
            .finish()
    }
}

impl<T> Clone for Hybrid<T> {
    fn clone(&self) -> Self {
        Hybrid {
            cpu_value: self.cpu_value.clone(),
            gpu_value: self.gpu_value.clone(),
        }
    }
}

impl<T> Hybrid<T> {
    pub fn id(&self) -> Id<T> {
        self.gpu_value.id()
    }
}

impl<T: SlabItem + Clone + Send + Sync + 'static> Hybrid<T> {
    pub fn new(mngr: &SlabAllocator<impl IsRuntime>, value: T) -> Self {
        let cpu_value = Arc::new(RwLock::new(value.clone()));
        let gpu_value = Gpu::new(mngr, value);
        Self {
            cpu_value,
            gpu_value,
        }
    }

    /// Returns the number of clones of this Hybrid on the CPU.
    pub fn ref_count(&self) -> usize {
        Arc::strong_count(&self.gpu_value.update)
    }

    pub fn get(&self) -> T {
        self.cpu_value.read().unwrap().clone()
    }

    pub fn modify<A: std::any::Any>(&self, f: impl FnOnce(&mut T) -> A) -> A {
        let mut value_guard = self.cpu_value.write().unwrap();
        let a = f(&mut value_guard);
        let t = value_guard.clone();
        self.gpu_value.set(t);
        a
    }

    pub fn set(&self, value: T) {
        self.modify(move |old| {
            *old = value;
        })
    }

    /// Drop the CPU portion of the hybrid value, returning a type that wraps
    /// only the GPU resources.
    pub fn into_gpu_only(self) -> Gpu<T> {
        self.gpu_value
    }

    /// Acquire a write lock on this value in order to mutate it.
    pub fn lock(&self) -> HybridWriteGuard<'_, T> {
        HybridWriteGuard::new(self)
    }

    /// A unique identifier.
    pub fn notifier_index(&self) -> SourceId {
        self.gpu_value.source_id
    }

    /// Sets the inner value, but **does not sync to the GPU**.
    ///
    /// Used primarily to bring changes from the GPU back to the
    /// CPU manually.
    ///
    /// Do not use this unless you really know what you're doing.
    pub fn set_without_sync(&self, value: T) {
        *self.cpu_value.write().unwrap() = value;
    }
}

/// A type that lives only on the GPU.
///
/// Updates are synchronized to the GPU at the behest of the [`SlabAllocator`]
/// that created this value.
#[repr(transparent)]
pub struct Gpu<T: SlabItem> {
    inner: Value<T, OneWayFromCpu>,
    _phantom: PhantomData<T>,
}

impl<T: SlabItem> Clone for Gpu<T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            _phantom: PhantomData,
        }
    }
}

impl<T> Gpu<T> {
    pub fn id(&self) -> Id<T> {
        let source = self.inner.source_id();
        Id::new(source.range.first_index)
    }
}

impl<T: SlabItem + Clone + Send + Sync + 'static> Gpu<T> {
    pub fn new(inner: CpuUpdateSource) -> Self {
        Self {
            inner,
            _phantom: PhantomData,
        }
    }

    pub fn set(&self, value: T) {
        self.inner.modify(|data| {
            data.write(Id::ZERO, &value);
        });
    }

    /// A unique identifier.
    pub fn source_id(&self) -> SourceId {
        self.inner.source_id()
    }
}

/// A array type that lives on the GPU.
///
/// Once created, the array cannot be resized.
///
/// Updates are syncronized to the GPU at the behest of the
/// [`SlabAllocator`] that created this array.
pub struct GpuArray<T> {
    inner: CpuUpdateSource,
    _phantom: PhantomData<T>,
}

impl<T> Clone for GpuArray<T> {
    fn clone(&self) -> Self {
        GpuArray {
            inner: self.inner.clone(),
            _phantom: PhantomData,
        }
    }
}

impl<T: SlabItem> GpuArray<T> {
    pub fn len(&self) -> usize {
        self.inner.source_id().range.len() as usize / T::SLAB_SIZE
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn array(&self) -> Array<T> {
        let len = self.len() as u32;
        let id = Id::new(self.inner.source_id().range.first_index);
        Array { id, len }
    }
}

impl<T: SlabItem + Clone + Send + Sync + 'static> GpuArray<T> {
    pub fn new(mngr: &SlabAllocator<impl IsRuntime>, values: &[T]) -> Self {
        let array = mngr.allocate_array::<T>(values.len());
        let update = {
            let mut elements = vec![0u32; T::SLAB_SIZE * array.len()];
            elements.write_indexed_slice(values, 0);
            SlabUpdate {
                array: array.into_u32_array(),
                elements,
            }
        };
        let notifier_index = SourceId {
            range: array.into(),
            type_is: std::any::type_name::<[T]>(),
        };
        let g = GpuArray {
            notifier_index,
            notifier: mngr.notifier.0.clone(),
            array,
            updates: Arc::new(Mutex::new(vec![update])),
        };
        mngr.insert_update_source(notifier_index, WeakGpuRef::from_gpu_array(&g));
        g
    }

    pub fn get_id(&self, index: usize) -> Id<T> {
        self.array().at(index)
    }

    /// Copy the item into the index within the array.
    ///
    /// ## Panics
    /// Panics if the index is outside of the array's range.
    pub fn set_item(&self, index: usize, value: &T) {
        self.set_items(index..index + 1, core::slice::from_ref(value))
    }

    /// Copy the items into the subslice denoted by the range.
    ///
    /// ## Panics
    /// Panics if the range does not fit within the array, or if the item
    /// length is greater than the range.
    pub fn set_items(&self, range: std::ops::Range<usize>, items: &[T]) {
        let end = range.end;
        let is_contained = end <= self.len();
        let array_len = self.len();
        assert!(
            is_contained,
            "range {range:?} would access outside of this GpuArray (length {array_len})"
        );
        let len = items.len();
        let range_len = range.end - range.start;
        assert!(
            len <= range_len,
            "length of items {len} is greater than the range provided {range_len}"
        );
        let mut data = vec![0u32; T::SLAB_SIZE * len];
        {
            // Write the items into the data
            let array = Array::<T>::new(Id::ZERO, len as u32);
            for (id, item) in array.iter().zip(items) {
                data.write(id, item);
            }
        }
        let inner_offset = range.start * T::SLAB_SIZE;
        let index = self.array.id.inner() + inner_offset as u32;
        self.updates.lock().unwrap().push(SlabUpdate {
            array: Array::new(Id::new(index), data.len() as u32),
            elements: data,
        });
        // UNWRAP: safe because it's unbounded
        self.notifier.try_send(self.notifier_index).unwrap();
    }

    /// A unique identifier.
    pub fn notifier_index(&self) -> SourceId {
        self.notifier_index
    }
}

/// A "hybrid" array type that lives on the CPU and the GPU.
///
/// Once created, the array cannot be resized.
///
/// Updates are syncronized to the GPU at the behest of the
/// [`SlabAllocator`] that created this array.
pub struct HybridArray<T> {
    cpu_value: Arc<RwLock<Vec<T>>>,
    gpu_value: GpuArray<T>,
}

impl<T: core::fmt::Debug> core::fmt::Debug for HybridArray<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct(&format!("HybridArray<{}>", std::any::type_name::<T>()))
            .field("array", &self.gpu_value.array)
            .field("cpu_value", &self.cpu_value.read().unwrap())
            .finish()
    }
}

impl<T> Clone for HybridArray<T> {
    fn clone(&self) -> Self {
        HybridArray {
            cpu_value: self.cpu_value.clone(),
            gpu_value: self.gpu_value.clone(),
        }
    }
}

impl<T> HybridArray<T> {
    pub fn len(&self) -> usize {
        self.gpu_value.array.len()
    }

    pub fn is_empty(&self) -> bool {
        self.gpu_value.is_empty()
    }

    pub fn array(&self) -> Array<T> {
        self.gpu_value.array()
    }
}

impl<T: SlabItem + Clone + Send + Sync + 'static> HybridArray<T> {
    pub fn new(mngr: &SlabAllocator<impl IsRuntime>, values: impl IntoIterator<Item = T>) -> Self {
        let values = values.into_iter().collect::<Vec<_>>();
        let gpu_value = GpuArray::<T>::new(mngr, &values);
        let cpu_value = Arc::new(RwLock::new(values));
        HybridArray {
            cpu_value,
            gpu_value,
        }
    }

    pub fn ref_count(&self) -> usize {
        Arc::strong_count(&self.gpu_value.updates)
    }

    pub fn get(&self, index: usize) -> Option<T> {
        self.cpu_value.read().unwrap().get(index).cloned()
    }

    pub fn get_vec(&self) -> Vec<T> {
        self.cpu_value.read().unwrap().clone()
    }

    /// Return a vec of items within the given range.
    ///
    /// ## Panics
    /// Panics if the range is out of bounds.
    pub fn get_vec_range(&self, range: std::ops::Range<usize>) -> Vec<T> {
        self.cpu_value.read().unwrap()[range].to_vec()
    }

    pub fn get_id(&self, index: usize) -> Id<T> {
        self.gpu_value.get_id(index)
    }

    pub fn modify<S>(&self, index: usize, f: impl FnOnce(&mut T) -> S) -> Option<S> {
        let mut value_guard = self.cpu_value.write().unwrap();
        let t = value_guard.get_mut(index)?;
        let output = Some(f(t));
        self.gpu_value.set_item(index, t);
        output
    }

    /// Modify the sub-slice of the array designated by the range, if possible.
    ///
    /// ## Panics
    /// Panics if the end of the range is greater than the number of items in the array.
    pub fn modify_range<S>(
        &self,
        range: std::ops::Range<usize>,
        f: impl FnOnce(&mut [T]) -> S,
    ) -> Option<S> {
        let mut value_guard = self.cpu_value.write().unwrap();
        let slice = value_guard.as_mut_slice();
        let sub_slice = slice.get_mut(range.clone())?;
        let output = f(sub_slice);
        self.gpu_value.set_items(range, sub_slice);
        Some(output)
    }

    pub fn set_item(&self, index: usize, value: T) -> Option<T> {
        self.modify(index, move |t| std::mem::replace(t, value))
    }

    /// Replace the items in the range with the items provided.
    ///
    /// ## Panics
    /// Panics if the end of the range is outside the bounds of the array,
    /// or if the item length is greater than the range.
    pub fn set_items(&self, range: std::ops::Range<usize>, items: &[T]) {
        self.modify_range(range, |current_items| {
            for (old, new) in current_items.iter_mut().zip(items.iter()) {
                *old = new.clone();
            }
        });
    }

    pub fn into_gpu_only(self) -> GpuArray<T> {
        self.gpu_value
    }

    /// A unique identifier.
    pub fn notifier_index(&self) -> SourceId {
        self.gpu_value.notifier_index
    }

    /// Sets the items within the range **without syncing to the GPU**.
    ///
    /// Used primarily to bring changes from the GPU back to the
    /// CPU manually.
    ///
    /// Do not use this unless you really know what you're doing.
    ///
    /// ## Panics
    /// Panics if the end of the range is greater than the number of items in
    /// the `HybridArray`.
    pub fn set_without_sync(&self, range: std::ops::Range<usize>, items: &[T]) {
        let mut guard = self.cpu_value.write().unwrap();
        let slice = &mut guard[range];
        for (here, there) in slice.iter_mut().zip(items) {
            *here = there.clone();
        }
    }
}

#[cfg(test)]
mod test {
    use crabslab::{Slab, SlabItem};

    use crate::{runtime::CpuRuntime, slab::SlabAllocator, value::HybridArray};

    #[test]
    /// Ensures that a `HybridArray` can update a range.
    fn array_subslice_sanity() {
        let _ = env_logger::builder().is_test(true).try_init();

        #[derive(Clone, Copy, Debug, Default, PartialEq, SlabItem)]
        struct Data {
            i: u32,
            float: f32,
            int: i64,
        }

        fn ensure(
            slab: &SlabAllocator<CpuRuntime>,
            initial_values: &Vec<Data>,
            values: &HybridArray<Data>,
        ) {
            // get it back from the cpu side
            assert_eq!(initial_values, &values.get_vec(), "cpu side wrong");
            // check they still match
            let buffer = slab.commit();
            let buffer_vec = buffer.as_vec();
            let values = buffer_vec.read_vec(values.array());
            assert_eq!(initial_values, &values, "gpu side wrong");
        }

        log::info!("creating the slab");
        let slab = SlabAllocator::new(CpuRuntime, "test", ());
        let mut initial_values = vec![
            Data {
                i: 0,
                float: 1.0,
                int: 1,
            },
            Data {
                i: 1,
                float: 2.0,
                int: 2,
            },
            Data {
                i: 2,
                float: 3.0,
                int: 3,
            },
        ];

        log::info!("staging initial values");
        let values = slab.new_array(initial_values.clone());
        ensure(&slab, &initial_values, &values);

        log::info!("updating the initial values");
        // change the initial values
        initial_values[1].int = 666;
        initial_values[2].int = 420;
        // modify the slab values to match
        values.modify_range(1..3, |items| {
            items[0].int = 666;
            items[1].int = 420;
        });
        ensure(&slab, &initial_values, &values);

        log::info!("updating the initial values with overlapping updates");
        // Now ensure that two updates within one commit apply correctly.
        // 1. update an outer range
        // 2. update an inner range
        // 3. ensure they are as expected
        initial_values[0].float = 10.0;
        initial_values[1].float = 20.0;
        initial_values[2].float = 30.0;
        values.modify_range(0..3, |items| {
            items[0].float = 10.0;
            items[1].float = 20.0;
            items[2].float = 30.0;
        });

        initial_values[1].float = 666.0;
        values.modify_range(1..2, |items| {
            items[0].float = 666.0;
        });
        ensure(&slab, &initial_values, &values);

        // Ensure the other setting functions work too
        initial_values[2].int = -32;
        values.modify(2, |data| data.int = -32);
        ensure(&slab, &initial_values, &values);
    }
}
