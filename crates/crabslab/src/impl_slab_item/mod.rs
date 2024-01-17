mod primitives;
mod tuples;

#[cfg(feature = "glam")]
mod glam;

use crate::SlabItem;

impl<T: SlabItem + Default> SlabItem for Option<T> {
    fn slab_size() -> usize {
        1 + T::slab_size()
    }

    fn read_slab(&mut self, index: usize, slab: &[u32]) -> usize {
        let mut proxy = 0u32;
        let index = proxy.read_slab(index, slab);
        if proxy == 1 {
            let mut t = T::default();
            let index = t.read_slab(index, slab);
            *self = Some(t);
            index
        } else {
            *self = None;
            index + T::slab_size()
        }
    }

    fn write_slab(&self, index: usize, slab: &mut [u32]) -> usize {
        if let Some(t) = self {
            let index = 1u32.write_slab(index, slab);
            t.write_slab(index, slab)
        } else {
            let index = 0u32.write_slab(index, slab);
            index + T::slab_size()
        }
    }
}

impl<T: SlabItem, const N: usize> SlabItem for [T; N] {
    fn slab_size() -> usize {
        <T as SlabItem>::slab_size() * N
    }

    fn read_slab(&mut self, mut index: usize, slab: &[u32]) -> usize {
        for i in 0..N {
            index = self[i].read_slab(index, slab);
        }
        index
    }

    fn write_slab(&self, mut index: usize, slab: &mut [u32]) -> usize {
        for i in 0..N {
            index = self[i].write_slab(index, slab);
        }
        index
    }
}

use core::marker::PhantomData;

impl<T: core::any::Any> SlabItem for PhantomData<T> {
    fn slab_size() -> usize {
        0
    }

    fn read_slab(&mut self, index: usize, _: &[u32]) -> usize {
        index
    }

    fn write_slab(&self, index: usize, _: &mut [u32]) -> usize {
        index
    }
}
