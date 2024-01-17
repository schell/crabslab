use crate::SlabItem;

macro_rules! impl_underflow_primitive {
    ($type: ty) => {
        impl SlabItem for $type {
            fn slab_size() -> usize {
                1
            }

            fn read_slab(&mut self, index: usize, slab: &[u32]) -> usize {
                if slab.len() > index {
                    *self = slab[index] as $type;
                    index + 1
                } else {
                    index
                }
            }

            fn write_slab(&self, index: usize, slab: &mut [u32]) -> usize {
                if slab.len() > index {
                    slab[index] = *self as u32;
                    index + 1
                } else {
                    index
                }
            }
        }
    };
}

macro_rules! impl_overflow_primitive {
    ($type: ty, $num_slots: expr) => {
        impl SlabItem for $type {
            fn slab_size() -> usize {
                $num_slots
            }

            fn read_slab(&mut self, index: usize, slab: &[u32]) -> usize {
                if slab.len() >= index + $num_slots {
                    *self = (0..$num_slots).fold(0, |acc, i| {
                        acc | ((<$type>::from(slab[index + i])) << (i * 32))
                    });
                    index + $num_slots
                } else {
                    index
                }
            }

            fn write_slab(&self, index: usize, slab: &mut [u32]) -> usize {
                if slab.len() >= index + $num_slots {
                    for i in 0..$num_slots {
                        slab[index + i] = (*self >> (i * 32)) as u32;
                    }
                    index + $num_slots
                } else {
                    index
                }
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
    fn slab_size() -> usize {
        1
    }

    fn read_slab(&mut self, index: usize, slab: &[u32]) -> usize {
        if slab.len() > index {
            *self = f32::from_bits(slab[index]);
            index + 1
        } else {
            index
        }
    }

    fn write_slab(&self, index: usize, slab: &mut [u32]) -> usize {
        if slab.len() > index {
            slab[index] = self.to_bits();
            index + 1
        } else {
            index
        }
    }
}

impl SlabItem for f64 {
    fn slab_size() -> usize {
        2
    }

    fn read_slab(&mut self, index: usize, slab: &[u32]) -> usize {
        let mut temp_u64 = 0u64;
        let index = temp_u64.read_slab(index, slab);
        *self = f64::from_bits(temp_u64);
        index
    }

    fn write_slab(&self, index: usize, slab: &mut [u32]) -> usize {
        let temp_u64 = self.to_bits();
        temp_u64.write_slab(index, slab)
    }
}

impl SlabItem for bool {
    fn slab_size() -> usize {
        1
    }

    fn read_slab(&mut self, index: usize, slab: &[u32]) -> usize {
        let mut proxy = 0u32;
        let index = proxy.read_slab(index, slab);
        *self = proxy == 1;
        index
    }

    fn write_slab(&self, index: usize, slab: &mut [u32]) -> usize {
        if *self { 1u32 } else { 0u32 }.write_slab(index, slab)
    }
}
