//! A slab-allocated array.
use core::marker::PhantomData;

use crate::{id::Id, slab::SlabItem};

/// Iterator over [`Id`]s in an [`Array`].
#[derive(Clone, Copy)]
pub struct ArrayIter<T> {
    array: Array<T>,
    index: usize,
}

impl<T: SlabItem> Iterator for ArrayIter<T> {
    type Item = Id<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.array.len() {
            None
        } else {
            let id = self.array.at(self.index);
            self.index += 1;
            Some(id)
        }
    }
}

/// A pointer to contiguous `T` elements in a slab.
///
/// ## Note
/// It's very important to keep in mind that although the array carries
/// around `T`, the `index` is always a u32 offset.
///
/// `len` is the number of contiguous `T` elements, but `index` is the
/// u32 offset of the first `T` element in the slab.
///
/// ```rust
/// use crabslab::{Array, CpuSlab, GrowableSlab, Id, Slab, SlabItem};
///
/// let ints = [1, 2, 3, 4, 5u32];
/// let mut slab = CpuSlab::new(vec![]);
/// let array = slab.append_array(&ints);
/// for (i, id) in array.iter().enumerate() {
///     let value = slab.read(id);
///     assert_eq!(ints[i], value);
/// }
/// ```
#[repr(C)]
pub struct Array<T> {
    /// Identifier of the first element in the array.
    pub id: Id<T>,
    // number of `T` elements in the array
    pub len: u32,
}

impl<T> Clone for Array<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for Array<T> {}

/// An `Id<T>` is an `Array<T>` with a length of 1.
impl<T> From<Id<T>> for Array<T> {
    fn from(id: Id<T>) -> Self {
        Self { id, len: 1 }
    }
}

impl<T> core::fmt::Debug for Array<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if self.is_null() {
            f.write_fmt(core::format_args!(
                "Array<{}>(null)",
                core::any::type_name::<T>()
            ))
        } else {
            f.write_fmt(core::format_args!("Array[{:?}; {}]", self.id, self.len))
        }
    }
}

impl<T> PartialEq for Array<T> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.len == other.len
    }
}

impl<T: SlabItem> SlabItem for Array<T> {
    const SLAB_SIZE: usize = 2;

    fn write_slab(&self, index: usize, slab: &mut [u32]) -> usize {
        let index = self.id.write_slab(index, slab);
        self.len.write_slab(index, slab)
    }

    fn read_slab(index: usize, slab: &[u32]) -> Self {
        let start = u32::read_slab(index, slab);
        let len = u32::read_slab(index + 1, slab);
        Array::new(Id::new(start), len)
    }
}

impl<T: SlabItem> Default for Array<T> {
    fn default() -> Self {
        Array::NONE
    }
}

impl<T> Array<T> {
    pub const NONE: Self = Array::new(Id::NONE, 0);

    pub const fn new(id: Id<T>, len: u32) -> Self {
        Self { id, len }
    }

    pub fn len(&self) -> usize {
        self.len as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn is_null(&self) -> bool {
        self.id.is_none()
    }

    pub fn at(&self, index: usize) -> Id<T>
    where
        T: SlabItem,
    {
        if index >= self.len() {
            Id::NONE
        } else {
            Id::new(self.id.inner() + (T::SLAB_SIZE * index) as u32)
        }
    }

    pub fn starting_index(&self) -> usize {
        self.id.index()
    }

    pub fn iter(&self) -> ArrayIter<T> {
        ArrayIter {
            array: *self,
            index: 0,
        }
    }

    /// Convert this array into a `u32` array.
    pub fn into_u32_array(self) -> Array<u32>
    where
        T: SlabItem,
    {
        Array {
            id: Id::new(self.id.inner()),
            len: self.len * T::SLAB_SIZE as u32,
        }
    }

    #[cfg(not(target_arch = "spirv"))]
    /// Return the slice of the slab that this array represents.
    pub fn sub_slab<'a>(&'a self, slab: &'a [u32]) -> &'a [u32]
    where
        T: SlabItem,
    {
        let u32_array = self.into_u32_array();
        let start = u32_array.starting_index();
        let range = start..start + u32_array.len();
        &slab[range]
    }
}

impl Array<u32> {
    pub fn union(&mut self, other: &Array<u32>) {
        let start = self.id.min(other.id);
        let end = (self.id.inner() + self.len).max(other.id.inner() + other.len);
        self.id = start;
        self.len = end - start.inner();
    }
}
