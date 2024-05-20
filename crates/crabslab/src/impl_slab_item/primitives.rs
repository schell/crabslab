use crate::SlabItem;

macro_rules! impl_underflow_primitive {
    ($type: ty) => {
        impl SlabItem for $type {
            const SLAB_SIZE: usize = { 1 };

            fn read_slab(index: usize, slab: &[u32]) -> Self {
                *crate::slice_index(slab, index) as $type
            }

            fn write_slab(&self, index: usize, slab: &mut [u32]) -> usize {
                *crate::slice_index_mut(slab, index) = *self as u32;
                index + 1
            }
        }
    };
}

macro_rules! impl_overflow_primitive {
    ($type: ty, $num_slots: expr) => {
        impl SlabItem for $type {
            const SLAB_SIZE: usize = { $num_slots };

            fn read_slab(index: usize, slab: &[u32]) -> Self {
                (0..$num_slots).fold(0, |acc, i| {
                    acc | ((<$type>::from(*crate::slice_index(slab, index + i))) << (i * 32))
                })
            }

            fn write_slab(&self, index: usize, slab: &mut [u32]) -> usize {
                for i in 0..$num_slots {
                    *crate::slice_index_mut(slab, index + i) = (*self >> (i * 32)) as u32;
                }
                index + $num_slots
            }
        }
    };
}

impl_underflow_primitive!(u8);
impl_underflow_primitive!(i8);
impl_underflow_primitive!(u16);
impl_underflow_primitive!(i16);
impl_underflow_primitive!(u32);
impl_underflow_primitive!(i32);

impl_overflow_primitive!(u64, 2);
impl_overflow_primitive!(i64, 2);
impl_overflow_primitive!(u128, 4);
impl_overflow_primitive!(i128, 4);

impl SlabItem for f32 {
    const SLAB_SIZE: usize = { 1 };

    fn read_slab(index: usize, slab: &[u32]) -> Self {
        let bits = crate::slice_index(slab, index);
        f32::from_bits(*bits)
    }

    fn write_slab(&self, index: usize, slab: &mut [u32]) -> usize {
        *crate::slice_index_mut(slab, index) = self.to_bits();
        index + 1
    }
}

impl SlabItem for f64 {
    const SLAB_SIZE: usize = { 2 };

    fn read_slab(index: usize, slab: &[u32]) -> Self {
        let temp_u64 = u64::read_slab(index, slab);
        f64::from_bits(temp_u64)
    }

    fn write_slab(&self, index: usize, slab: &mut [u32]) -> usize {
        let temp_u64 = self.to_bits();
        temp_u64.write_slab(index, slab)
    }
}

impl SlabItem for bool {
    const SLAB_SIZE: usize = { 1 };

    fn read_slab(index: usize, slab: &[u32]) -> Self {
        u32::read_slab(index, slab) == 1
    }

    fn write_slab(&self, index: usize, slab: &mut [u32]) -> usize {
        if *self { 1u32 } else { 0u32 }.write_slab(index, slab)
    }
}
