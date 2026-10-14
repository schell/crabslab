//! The unified CPU/GPU `SlabItem` trait (Option B).
//!
//! This module promotes the trait and its WGSL-expressible primitive impls
//! out of the `wgsl_impl.rs` experiment into the crate's real surface. The
//! module is a `#[wgsl]` module so that it transpiles: the trait definition
//! is Rust-only (WGSL has no traits), while the primitive impls render as
//! mangled functions (`u32__1to_array`, ...) that any `#[wgsl]` module can
//! import via `use crabslab::slab_item::*;`.
//!
//! CPU-only impls (u8..u128, f64, tuples, arrays, glam) live in
//! [`crate::impl_slab_item`]; slab I/O (`read_slab`/`write_slab`) lives on
//! the `CpuSlabItem` extension trait in [`crate::slab`].
use wgsl_rs::wgsl;

#[wgsl]
pub mod slab {
    use wgsl_rs::std::*;

    /// A type that can be serialized to and from a `[u32; N]` array for slab
    /// storage.
    ///
    /// The `type Array` associated type is the fixed-size `[u32; N]` array
    /// corresponding to `SLAB_SIZE`. This indirection is needed because
    /// `[u32; Self::SLAB_SIZE]` is not valid in a trait signature on stable
    /// Rust (requires `generic_const_exprs`).
    ///
    /// On the CPU, `#[derive(SlabItem)]` generates this impl. On the GPU,
    /// the `SlabItemExt` `WgslExtension` generates the same method names as
    /// inherent methods for `#[derive(SlabItem)]` structs, because derives
    /// run after `#[wgsl]` and their output is invisible to the
    /// transpiler.
    pub trait SlabItem: Sized {
        /// The number of `u32` slots this type occupies in a slab.
        const SLAB_SIZE: usize;

        /// The fixed-size `[u32; Self::SLAB_SIZE]` array type.
        type Array: AsRef<[u32]> + AsMut<[u32]>;

        /// Serialize this value into a `[u32; N]` array.
        fn to_array(data: Self) -> Self::Array;

        /// Deserialize a value from a `[u32; N]` array.
        fn from_array(arr: Self::Array) -> Self;

        /// Create a zero-initialized array container of the right size.
        fn array_container() -> Self::Array;
    }

    impl SlabItem for u32 {
        const SLAB_SIZE: usize = 1;
        type Array = [u32; 1];
        fn to_array(data: Self) -> [u32; 1] {
            [data]
        }
        fn from_array(arr: [u32; 1]) -> Self {
            arr[0]
        }

        fn array_container() -> Self::Array {
            [0]
        }
    }

    impl SlabItem for i32 {
        const SLAB_SIZE: usize = 1;
        type Array = [u32; 1];
        fn to_array(data: Self) -> [u32; 1] {
            [bitcast_u32(data)]
        }
        fn from_array(arr: [u32; 1]) -> Self {
            bitcast_i32(arr[0])
        }
        fn array_container() -> Self::Array {
            [0]
        }
    }

    impl SlabItem for f32 {
        const SLAB_SIZE: usize = 1;
        type Array = [u32; 1];
        fn from_array(arr: [u32; 1]) -> Self {
            bitcast_f32(arr[0])
        }
        fn to_array(data: Self) -> [u32; 1] {
            [bitcast_u32(data)]
        }
        fn array_container() -> Self::Array {
            [0]
        }
    }

    impl SlabItem for bool {
        const SLAB_SIZE: usize = 1;
        type Array = [u32; 1];

        fn to_array(data: Self) -> [u32; 1] {
            [select(0u32, 1u32, data)]
        }

        fn from_array(arr: [u32; 1]) -> Self {
            arr[0] != 0u32
        }

        fn array_container() -> Self::Array {
            [0]
        }
    }
}

pub use slab::*;

// Re-export the derive alongside the trait so a glob import
// (`use crabslab::slab_item::*;`) brings both the trait (type
// namespace) and the derive macro (macro namespace) into scope —
// `#[wgsl]` modules only accept glob imports.
pub use ::crabslab_derive::SlabItem;

#[cfg(test)]
mod test {
    use super::*;
    use crate::CpuSlabItem;

    /// Round-trip every promoted primitive through `to_array`/`from_array`.
    #[test]
    fn primitives_round_trip() {
        assert_eq!([42u32], u32::to_array(42u32));
        assert_eq!(42u32, u32::from_array([42]));
        assert_eq!(u32::MAX, u32::from_array(u32::to_array(u32::MAX)));

        assert_eq!(-7i32, i32::from_array(i32::to_array(-7i32)));
        assert_eq!(i32::MIN, i32::from_array(i32::to_array(i32::MIN)));

        assert_eq!(3.14f32, f32::from_array(f32::to_array(3.14f32)));
        assert_eq!(f32::MIN, f32::from_array(f32::to_array(f32::MIN)));

        assert_eq!(true, bool::from_array(bool::to_array(true)));
        assert_eq!(false, bool::from_array(bool::to_array(false)));
    }

    /// The f32 and i32 arrays carry the exact bit patterns (CPU bitcasts
    /// must agree with `f32::to_bits` / `i32 as u32`).
    #[test]
    fn primitive_bit_patterns() {
        let f = -123.456f32;
        assert_eq!(f.to_bits(), f32::to_array(f)[0]);
        let i = -12345i32;
        assert_eq!(i as u32, i32::to_array(i)[0]);
    }

    /// `array_container` is zero-initialized and the right length.
    #[test]
    fn array_containers_are_zeroed() {
        assert_eq!([0u32], u32::array_container());
        assert_eq!([0u32], i32::array_container());
        assert_eq!([0u32], f32::array_container());
        assert_eq!([0u32], bool::array_container());
    }

    /// All promoted primitives occupy a single slab slot.
    #[test]
    fn slab_sizes() {
        assert_eq!(1, u32::SLAB_SIZE);
        assert_eq!(1, i32::SLAB_SIZE);
        assert_eq!(1, f32::SLAB_SIZE);
        assert_eq!(1, bool::SLAB_SIZE);
    }

    /// `CpuSlabItem` blanket I/O round-trips primitives through a slab.
    #[test]
    fn cpu_slab_item_round_trip() {
        let mut slab = [0u32; 16];

        let mut index = 5u32.write_slab(0, &mut slab);
        assert_eq!(1, index);
        index = (-7i32).write_slab(index, &mut slab);
        assert_eq!(2, index);
        index = 3.14f32.write_slab(index, &mut slab);
        assert_eq!(3, index);
        index = true.write_slab(index, &mut slab);
        assert_eq!(4, index);

        assert_eq!(5u32, u32::read_slab(0, &slab));
        assert_eq!(-7i32, i32::read_slab(1, &slab));
        assert_eq!(3.14f32, f32::read_slab(2, &slab));
        assert!(bool::read_slab(3, &slab));

        // bool slots only ever contain 0 or 1 from our writers.
        assert_eq!(1u32, slab[3]);
    }
}
