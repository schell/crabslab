//! CPU-only `SlabItem` impls for primitive types that are not expressible in
//! WGSL (`u8`..`i128`, `f64`).
//!
//! The WGSL-expressible primitives (`u32`, `i32`, `f32`, `bool`) are
//! implemented inside the promoted `#[wgsl]` module in [`crate::slab_item`]
//! so that a single impl serves both worlds.

use crate::SlabItem;

macro_rules! impl_underflow_primitive {
    ($type: ty) => {
        impl SlabItem for $type {
            const SLAB_SIZE: usize = 1;
            type Array = [u32; 1];

            fn to_array(data: Self) -> Self::Array {
                [data as u32]
            }

            fn from_array(arr: Self::Array) -> Self {
                arr[0] as $type
            }

            fn array_container() -> Self::Array {
                [0]
            }
        }
    };
}

macro_rules! impl_overflow_primitive {
    ($type: ty, $num_slots: expr) => {
        impl SlabItem for $type {
            const SLAB_SIZE: usize = { $num_slots };
            type Array = [u32; $num_slots];

            fn to_array(data: Self) -> Self::Array {
                let mut arr = [0u32; $num_slots];
                for (i, slot) in arr.iter_mut().enumerate() {
                    *slot = (data >> (i * 32)) as u32;
                }
                arr
            }

            fn from_array(arr: Self::Array) -> Self {
                (0..$num_slots).fold(0, |acc, i| acc | ((<$type>::from(arr[i])) << (i * 32)))
            }

            fn array_container() -> Self::Array {
                [0u32; $num_slots]
            }
        }
    };
}

impl_underflow_primitive!(u8);
impl_underflow_primitive!(i8);
impl_underflow_primitive!(u16);
impl_underflow_primitive!(i16);

impl_overflow_primitive!(u64, 2);
impl_overflow_primitive!(i64, 2);
impl_overflow_primitive!(u128, 4);
impl_overflow_primitive!(i128, 4);

impl SlabItem for f64 {
    const SLAB_SIZE: usize = 2;
    type Array = [u32; 2];

    fn to_array(data: Self) -> Self::Array {
        u64::to_array(data.to_bits())
    }

    fn from_array(arr: Self::Array) -> Self {
        f64::from_bits(u64::from_array(arr))
    }

    fn array_container() -> Self::Array {
        [0, 0]
    }
}
