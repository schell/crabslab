//! Typed identifiers that can also be used as indices.
use core::marker::PhantomData;

use crate::slab::SlabItem;

/// `u32` value of an [`Id`] that does not point to any item.
pub const ID_NONE: u32 = u32::MAX;

/// An identifier that can be used to read or write a type from/into the slab.
#[repr(transparent)]
pub struct Id<T>(pub(crate) u32, PhantomData<T>);

impl<T: core::any::Any> SlabItem for Id<T> {
    const SLAB_SIZE: usize = { 1 };

    fn read_slab(index: usize, slab: &[u32]) -> Self {
        Id::new(*crate::slice_index(slab, index))
    }

    fn write_slab(&self, index: usize, slab: &mut [u32]) -> usize {
        *crate::slice_index_mut(slab, index) = self.0;
        index + 1
    }
}

impl<T> PartialOrd for Id<T> {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.0.cmp(&other.0))
    }
}

impl<T> Ord for Id<T> {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.0.cmp(&other.0)
    }
}

impl<T> Copy for Id<T> {}

impl<T> Clone for Id<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> core::hash::Hash for Id<T> {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl<T> PartialEq for Id<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<T> Eq for Id<T> {}

impl<T> From<Id<T>> for u32 {
    fn from(value: Id<T>) -> Self {
        value.0
    }
}

impl<T> From<usize> for Id<T> {
    fn from(value: usize) -> Self {
        Id::new(value as u32)
    }
}

impl<T> From<u32> for Id<T> {
    fn from(value: u32) -> Self {
        Id::new(value)
    }
}

/// `Id::NONE` is the default.
impl<T> Default for Id<T> {
    fn default() -> Self {
        Id::NONE
    }
}

impl<T> core::fmt::Debug for Id<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if self.is_none() {
            f.write_fmt(core::format_args!(
                "Id<{}>(null)",
                &core::any::type_name::<T>(),
            ))
        } else {
            f.write_fmt(core::format_args!(
                "Id<{}>({})",
                &core::any::type_name::<T>(),
                &self.0
            ))
        }
    }
}

impl<T> core::ops::Add<usize> for Id<T> {
    type Output = Self;

    fn add(self, rhs: usize) -> Self::Output {
        Id::new(self.0 + rhs as u32)
    }
}

impl<T> core::ops::Add<Id<T>> for usize {
    type Output = Id<T>;

    fn add(self, rhs: Id<T>) -> Self::Output {
        Id::new(self as u32 + rhs.0)
    }
}

impl<T> core::ops::Add<u32> for Id<T> {
    type Output = Self;

    fn add(self, rhs: u32) -> Self::Output {
        Id::new(self.0 + rhs)
    }
}

impl<T> core::ops::Add<Id<T>> for u32 {
    type Output = Id<T>;

    fn add(self, rhs: Id<T>) -> Self::Output {
        Id::new(self + rhs.0)
    }
}

impl<T> Id<T> {
    pub const NONE: Self = Id::new(ID_NONE);

    pub const fn new(i: u32) -> Self {
        Id(i, PhantomData)
    }

    /// Convert this id into a usize for use as an index.
    pub fn index(&self) -> usize {
        self.0 as usize
    }

    /// The raw u32 value of this id.
    pub fn inner(&self) -> u32 {
        self.0
    }

    pub fn is_none(&self) -> bool {
        *self == Id::NONE
    }

    pub fn is_some(&self) -> bool {
        !self.is_none()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate as crabslab;
    use crabslab::SlabItem;

    #[test]
    fn id_size() {
        #[derive(SlabItem)]
        struct MyEntity {
            name: u32,
            age: f32,
            destiny: [u32; 3],
        }

        assert_eq!(
            std::mem::size_of::<u32>(),
            std::mem::size_of::<Id<MyEntity>>(),
            "id is not u32"
        );
    }

    #[test]
    fn write_id() {
        let mut slab = [0u32; 4];
        let id = Id::<u32>::new(666);
        let index = 0;
        let index = id.write_slab(index, &mut slab);
        assert_eq!(1, index);
        let index = id.write_slab(index, &mut slab);
        assert_eq!(2, index);
    }
}
