mod primitives;

#[cfg(feature = "glam")]
mod glam;

use core::marker::PhantomData;

use crate::SlabItem;

/// `PhantomData<T>` occupies no slab slots.
impl<T> SlabItem for PhantomData<T> {
    const SLAB_SIZE: usize = 0;
    type Array = [u32; 0];

    fn to_array(_data: Self) -> Self::Array {
        []
    }

    fn from_array(_arr: Self::Array) -> Self {
        PhantomData
    }

    fn array_container() -> Self::Array {
        []
    }
}
