mod primitives;
mod tuples;

#[cfg(feature = "glam")]
mod glam;

use core::marker::PhantomData;

use crate::SlabItem;

/// `Option<T>` is stored as a discriminant slot followed by `T`'s slots.
///
/// `type Array` is a `Vec<u32>` because the size depends on a generic
/// parameter; a fixed-size array would require the unstable
/// `generic_const_exprs` feature.
impl<T: SlabItem> SlabItem for Option<T> {
    const SLAB_SIZE: usize = { 1 + T::SLAB_SIZE };
    type Array = Vec<u32>;

    fn to_array(data: Self) -> Self::Array {
        let mut dest = Self::array_container();
        if let Some(t) = data {
            dest[0] = 1;
            let inner = T::to_array(t);
            dest[1..1 + T::SLAB_SIZE].copy_from_slice(inner.as_ref());
        }
        dest
    }

    fn from_array(arr: Self::Array) -> Self {
        if AsRef::<[u32]>::as_ref(&arr)[0] == 1 {
            let mut inner = T::array_container();
            AsMut::<[u32]>::as_mut(&mut inner)
                .copy_from_slice(&AsRef::<[u32]>::as_ref(&arr)[1..1 + T::SLAB_SIZE]);
            Some(T::from_array(inner))
        } else {
            None
        }
    }

    fn array_container() -> Self::Array {
        vec![0u32; Self::SLAB_SIZE]
    }
}

/// Arrays store their elements contiguously, with no per-element padding.
///
/// Like `Option<T>`, `type Array` is a `Vec<u32>` because the size depends
/// on a generic parameter (`generic_const_exprs` would allow a stack array).
impl<T: SlabItem + Copy + Default, const N: usize> SlabItem for [T; N]
where
    [T; N]: Default,
{
    const SLAB_SIZE: usize = { <T as SlabItem>::SLAB_SIZE * N };
    type Array = Vec<u32>;

    fn to_array(data: Self) -> Self::Array {
        let mut dest = Self::array_container();
        for (i, element) in data.iter().enumerate() {
            let inner = T::to_array(*element);
            let offset = i * T::SLAB_SIZE;
            dest[offset..offset + T::SLAB_SIZE].copy_from_slice(inner.as_ref());
        }
        dest
    }

    fn from_array(arr: Self::Array) -> Self {
        let mut array: [T; N] = Default::default();
        for (i, slot) in array.iter_mut().enumerate() {
            let mut inner = T::array_container();
            let offset = i * T::SLAB_SIZE;
            AsMut::<[u32]>::as_mut(&mut inner)
                .copy_from_slice(&AsRef::<[u32]>::as_ref(&arr)[offset..offset + T::SLAB_SIZE]);
            *slot = T::from_array(inner);
        }
        array
    }

    fn array_container() -> Self::Array {
        vec![0u32; Self::SLAB_SIZE]
    }
}

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
