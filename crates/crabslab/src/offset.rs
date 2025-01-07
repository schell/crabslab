//! Struct field identifiers.

use core::marker::PhantomData;

use crate::Id;

#[cfg(doc)]
use crate::SlabItem;

/// The slab offset of field `F` within a type `T`.
///
/// An offset `Offset<F, T>` can be added to an [`Id<T>`](Id) to get an
/// [`Id<F>`].
///
/// Offset const values are automatically derived for structs that derive
/// [`SlabItem`] with the additional attribute `offset`:
///
/// ```rust
/// use crabslab::{Id, offset::Offset, Slab, SlabItem};
///
/// #[derive(Debug, Default, PartialEq, SlabItem)]
/// #[offsets]
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
/// slab.write(parent_id + Parent::OFFSET_OF_CHILD_A, &42);
/// let a = slab.read(parent_id + Parent::OFFSET_OF_CHILD_A);
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
/// #[offsets]
/// pub struct Child {
///     pub value: u32,
/// }
///
/// #[derive(Debug, Default, PartialEq, SlabItem)]
/// #[offsets]
/// pub struct Changeling {
///     pub value: u32,
/// }
///
/// #[derive(Debug, Default, PartialEq, SlabItem)]
/// #[offsets]
/// pub struct Parent {
///     pub child: Child,
/// }
///
/// let mut slab = CpuSlab::new(vec![]);
/// let parent_id = slab.append(&Parent::default());
///
/// // This will write `42` into the `value` field of the `Child` struct:
/// slab.write(
///     (parent_id + Parent::OFFSET_OF_CHILD) + Child::OFFSET_OF_VALUE,
///     &42u32,
/// );
///
/// // This will not compile:
/// slab.write(
///     (parent_id + Parent::OFFSET_OF_CHILD) + Changeling::OFFSET_OF_VALUE,
///     //                                    ^ cannot add ...
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
