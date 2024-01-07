//! Typed identifiers that can also be used as indices.
use core::marker::PhantomData;

use crate::{self as crabslab, slab::SlabItem};

/// `u32` value of an [`Id`] that does not point to any item.
pub const ID_NONE: u32 = u32::MAX;

/// An identifier that can be used to read or write a type from/into the slab.
#[repr(transparent)]
#[derive(SlabItem)]
pub struct Id<T>(pub(crate) u32, PhantomData<T>);

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
        Id::new(self as u32 + rhs.0 as u32)
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

/// The slab offset of field `F` within a type `T`.
///
/// An offset `Offset<F, T>` can be added to an [`Id<T>`](Id) to get an
/// [`Id<F>`].
///
/// Offset functions are automatically derived for [`SlabItem`] structs.
///
/// ```rust
/// use crabslab::{Id, Offset, Slab, SlabItem};
///
/// #[derive(Debug, Default, PartialEq, SlabItem)]
/// pub struct Parent {
///     pub child_a: u32,
///     pub child_b: u32,
/// }
///
/// let mut slab = [0u32; 10];
///
/// let parent_id = Id::new(3);
/// let parent = Parent {
///     child_a: 0,
///     child_b: 1,
/// };
/// slab.write(parent_id, &parent);
/// assert_eq!(parent, slab.read(parent_id));
///
/// slab.write(parent_id + Parent::offset_of_child_a(), &42);
/// let a = slab.read(parent_id + Parent::offset_of_child_a());
/// assert_eq!(42, a);
/// ```
///
/// Furthermore, offsets cannot be added unless their types are compatible.
/// This helps when chaining together offsets to drill down through a struct:
///
/// ```rust, compile_fail
/// use crabslab::*;
///
/// #[derive(Debug, Default, PartialEq, SlabItem)]
/// pub struct Child {
///     pub value: u32,
/// }
///
/// #[derive(Debug, Default, PartialEq, SlabItem)]
/// pub struct Changeling {
///     pub value: u32,
/// }
///
/// #[derive(Debug, Default, PartialEq, SlabItem)]
/// pub struct Parent {
///     pub child: Child,
/// }
///
/// let mut slab = CpuSlab::new(vec![]);
/// let parent_id = slab.append(&Parent::default());
///
/// // This will write `42` into the `value` field of the `Child` struct:
/// slab.write(
///     (parent_id + Parent::offset_of_child()) + Child::offset_of_value(),
///     &42u32,
/// );
///
/// // This will not compile:
/// slab.write(
///     (parent_id + Parent::offset_of_child()) + Changeling::offset_of_value(),
///     &666u32,
/// );
/// ```
pub struct Offset<F, T> {
    pub offset: u32,
    _phantom: PhantomData<(F, T)>,
}

impl<F, T> core::ops::Add<Id<T>> for Offset<F, T> {
    type Output = Id<F>;

    fn add(self, rhs: Id<T>) -> Self::Output {
        Id::new(self.offset + rhs.0)
    }
}

impl<F, T> core::ops::Add<Offset<F, T>> for Id<T> {
    type Output = Id<F>;

    fn add(self, rhs: Offset<F, T>) -> Self::Output {
        Id::new(self.0 + rhs.offset)
    }
}

impl<F, T> From<Offset<F, T>> for Id<F> {
    fn from(value: Offset<F, T>) -> Self {
        Id::new(value.offset)
    }
}

impl<F, T> Offset<F, T> {
    pub const fn new(offset: usize) -> Self {
        Self {
            offset: offset as u32,
            _phantom: PhantomData,
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[derive(SlabItem)]
    struct MyEntity {
        name: u32,
        age: f32,
        destiny: [u32; 3],
    }

    #[test]
    fn id_size() {
        assert_eq!(
            std::mem::size_of::<u32>(),
            std::mem::size_of::<Id<MyEntity>>(),
            "id is not u32"
        );
    }
}
