mod primitives;
mod tuples;

#[cfg(feature = "glam")]
mod glam;

use crate::SlabItem;

impl<T: SlabItem + Default> SlabItem for Option<T> {
    const SLAB_SIZE: usize = { 1 + T::SLAB_SIZE };

    fn read_slab(index: usize, slab: &[u32]) -> Self {
        let proxy = u32::read_slab(index, slab);
        if proxy == 1 {
            let t = T::read_slab(index + 1, slab);
            Some(t)
        } else {
            None
        }
    }

    fn write_slab(&self, index: usize, slab: &mut [u32]) -> usize {
        if let Some(t) = self {
            let index = 1u32.write_slab(index, slab);
            t.write_slab(index, slab)
        } else {
            let index = 0u32.write_slab(index, slab);
            index + T::SLAB_SIZE
        }
    }
}

impl<T: SlabItem + Copy + Default, const N: usize> SlabItem for [T; N] {
    const SLAB_SIZE: usize = { <T as SlabItem>::SLAB_SIZE * N };

    fn read_slab(index: usize, slab: &[u32]) -> Self {
        let mut array = [T::default(); N];
        for i in 0..N {
            let j = index + i * T::SLAB_SIZE;
            let t = T::read_slab(j, slab);
            let a: &mut T = crate::array_index_mut(&mut array, i);
            *a = t;
        }
        array
    }

    fn write_slab(&self, mut index: usize, slab: &mut [u32]) -> usize {
        for i in 0..N {
            let n = crate::slice_index(self, i);
            index = n.write_slab(index, slab);
        }
        index
    }
}

use core::marker::PhantomData;

impl<T: core::any::Any> SlabItem for PhantomData<T> {
    const SLAB_SIZE: usize = { 0 };

    fn read_slab(_: usize, _: &[u32]) -> Self {
        PhantomData
    }

    fn write_slab(&self, index: usize, _: &mut [u32]) -> usize {
        index
    }
}
