//! Creating and crafting a tasty slab of memory.
#![doc = include_str!("../README.md")]
#![warn(missing_docs)]

// Allow the `#[derive(SlabItem)]` macro to reference `crabslab::SlabItem`
// from within the crate itself.
pub extern crate self as crabslab;

mod array;
pub use array::*;
mod id;
mod slab;
pub use slab::*;
mod slab_item_ext;
pub use slab_item_ext::*;

#[macro_use]
mod macros;

pub use crate::wgsl::*;

// Derive-macro support items below. `__saturating_sub` is referenced by
// code generated from `#[derive(SlabItem)]`; the index helpers predate
// it and appear unused, but stay for compatibility. None are intended
// API, so they are hidden from the docs.

#[doc(hidden)]
pub const fn __saturating_sub(a: usize, b: usize) -> usize {
    a.saturating_sub(b)
}

#[doc(hidden)]
#[inline]
pub fn slice_index<T>(slab: &[T], index: usize) -> &T {
    &slab[index]
}

#[doc(hidden)]
#[inline]
pub fn array_index<const N: usize, T>(slab: &[T; N], index: usize) -> &T {
    &slab[index]
}

#[doc(hidden)]
#[inline]
pub fn array_index_mut<const N: usize, T>(slab: &mut [T; N], index: usize) -> &mut T {
    &mut slab[index]
}

#[allow(missing_docs)]
#[wgsl_rs::wgsl]
pub mod wgsl {
    //! Standard WGSL shader prelude, meant to be glob imported into `wgsl-rs` modules.
    use wgsl_rs::std::*;

    #[wgsl_ignore]
    pub use crabslab_derive::SlabItem;

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

    impl SlabItem for Vec2f {
        const SLAB_SIZE: usize = 2;
        type Array = [u32; 2];

        fn to_array(data: Self) -> [u32; 2] {
            [bitcast_u32(data.x), bitcast_u32(data.y)]
        }

        fn from_array(array: [u32; 2]) -> Self {
            vec2f(bitcast_f32(array[0]), bitcast_f32(array[1]))
        }

        fn array_container() -> [u32; 2] {
            [0u32, 0u32]
        }
    }

    impl SlabItem for Vec2i {
        const SLAB_SIZE: usize = 2;
        type Array = [u32; 2];

        fn to_array(data: Self) -> [u32; 2] {
            [bitcast_u32(data.x), bitcast_u32(data.y)]
        }

        fn from_array(array: [u32; 2]) -> Self {
            vec2i(bitcast_i32(array[0]), bitcast_i32(array[1]))
        }

        fn array_container() -> [u32; 2] {
            [0u32, 0u32]
        }
    }

    impl SlabItem for Vec2u {
        const SLAB_SIZE: usize = 2;
        type Array = [u32; 2];

        fn to_array(data: Self) -> [u32; 2] {
            [data.x, data.y]
        }

        fn from_array(array: [u32; 2]) -> Self {
            vec2u(array[0], array[1])
        }

        fn array_container() -> [u32; 2] {
            [0u32, 0u32]
        }
    }

    impl SlabItem for Vec2b {
        const SLAB_SIZE: usize = 2;
        type Array = [u32; 2];

        fn to_array(data: Self) -> [u32; 2] {
            [select(0, 1, data.x), select(0, 1, data.y)]
        }

        fn from_array(array: [u32; 2]) -> Self {
            vec2b(array[0] != 0u32, array[1] != 0u32)
        }

        fn array_container() -> [u32; 2] {
            [0u32, 0u32]
        }
    }

    impl SlabItem for Vec3f {
        const SLAB_SIZE: usize = 3;
        type Array = [u32; 3];

        fn to_array(data: Self) -> [u32; 3] {
            [
                bitcast_u32(data.x),
                bitcast_u32(data.y),
                bitcast_u32(data.z),
            ]
        }

        fn from_array(array: [u32; 3]) -> Self {
            vec3f(
                bitcast_f32(array[0]),
                bitcast_f32(array[1]),
                bitcast_f32(array[2]),
            )
        }

        fn array_container() -> [u32; 3] {
            [0u32, 0u32, 0u32]
        }
    }

    impl SlabItem for Vec3i {
        const SLAB_SIZE: usize = 3;
        type Array = [u32; 3];

        fn to_array(data: Self) -> [u32; 3] {
            [
                bitcast_u32(data.x),
                bitcast_u32(data.y),
                bitcast_u32(data.z),
            ]
        }

        fn from_array(array: [u32; 3]) -> Self {
            vec3i(
                bitcast_i32(array[0]),
                bitcast_i32(array[1]),
                bitcast_i32(array[2]),
            )
        }

        fn array_container() -> [u32; 3] {
            [0u32, 0u32, 0u32]
        }
    }

    impl SlabItem for Vec3u {
        const SLAB_SIZE: usize = 3;
        type Array = [u32; 3];

        fn to_array(data: Self) -> [u32; 3] {
            [data.x, data.y, data.z]
        }

        fn from_array(array: [u32; 3]) -> Self {
            vec3u(array[0], array[1], array[2])
        }

        fn array_container() -> [u32; 3] {
            [0u32, 0u32, 0u32]
        }
    }

    impl SlabItem for Vec3b {
        const SLAB_SIZE: usize = 3;
        type Array = [u32; 3];

        fn to_array(data: Self) -> [u32; 3] {
            [
                select(0, 1, data.x),
                select(0, 1, data.y),
                select(0, 1, data.z),
            ]
        }

        fn from_array(array: [u32; 3]) -> Self {
            vec3b(array[0] != 0u32, array[1] != 0u32, array[2] != 0u32)
        }

        fn array_container() -> [u32; 3] {
            [0u32, 0u32, 0u32]
        }
    }

    impl SlabItem for Vec4f {
        const SLAB_SIZE: usize = 4;
        type Array = [u32; 4];

        fn to_array(data: Self) -> Self::Array {
            [
                bitcast_u32(data.x),
                bitcast_u32(data.y),
                bitcast_u32(data.z),
                bitcast_u32(data.w),
            ]
        }

        fn from_array(array: Self::Array) -> Self {
            vec4f(
                bitcast_f32(array[0]),
                bitcast_f32(array[1]),
                bitcast_f32(array[2]),
                bitcast_f32(array[3]),
            )
        }

        fn array_container() -> Self::Array {
            [0u32, 0u32, 0u32, 0u32]
        }
    }

    impl SlabItem for Vec4i {
        const SLAB_SIZE: usize = 4;
        type Array = [u32; 4];

        fn to_array(data: Self) -> Self::Array {
            [
                bitcast_u32(data.x),
                bitcast_u32(data.y),
                bitcast_u32(data.z),
                bitcast_u32(data.w),
            ]
        }

        fn from_array(array: Self::Array) -> Self {
            vec4i(
                bitcast_i32(array[0]),
                bitcast_i32(array[1]),
                bitcast_i32(array[2]),
                bitcast_i32(array[3]),
            )
        }

        fn array_container() -> Self::Array {
            [0u32, 0u32, 0u32, 0u32]
        }
    }

    impl SlabItem for Vec4u {
        const SLAB_SIZE: usize = 4;
        type Array = [u32; 4];

        fn to_array(data: Self) -> Self::Array {
            [data.x, data.y, data.z, data.w]
        }

        fn from_array(array: Self::Array) -> Self {
            vec4u(array[0], array[1], array[2], array[3])
        }

        fn array_container() -> Self::Array {
            [0u32, 0u32, 0u32, 0u32]
        }
    }

    impl SlabItem for Vec4b {
        const SLAB_SIZE: usize = 4;
        type Array = [u32; 4];

        fn to_array(data: Self) -> Self::Array {
            [
                select(0, 1, data.x),
                select(0, 1, data.y),
                select(0, 1, data.z),
                select(0, 1, data.w),
            ]
        }

        fn from_array(array: Self::Array) -> Self {
            vec4b(
                array[0] != 0u32,
                array[1] != 0u32,
                array[2] != 0u32,
                array[3] != 0u32,
            )
        }

        fn array_container() -> Self::Array {
            [0u32, 0u32, 0u32, 0u32]
        }
    }

    impl SlabItem for Mat2x2f {
        const SLAB_SIZE: usize = 4;
        type Array = [u32; 4];

        fn to_array(data: Mat2x2f) -> [u32; 4] {
            [
                bitcast_u32(data[0u32].x),
                bitcast_u32(data[0u32].y),
                bitcast_u32(data[1u32].x),
                bitcast_u32(data[1u32].y),
            ]
        }

        fn from_array(array: [u32; 4]) -> Mat2x2f {
            mat2x2f(
                vec2f(bitcast_f32(array[0]), bitcast_f32(array[1])),
                vec2f(bitcast_f32(array[2]), bitcast_f32(array[3])),
            )
        }

        fn array_container() -> [u32; 4] {
            [0u32, 0u32, 0u32, 0u32]
        }
    }

    impl SlabItem for Mat2x3f {
        const SLAB_SIZE: usize = 6;
        type Array = [u32; 6];

        fn to_array(data: Mat2x3f) -> [u32; 6] {
            [
                bitcast_u32(data[0u32].x),
                bitcast_u32(data[0u32].y),
                bitcast_u32(data[0u32].z),
                bitcast_u32(data[1u32].x),
                bitcast_u32(data[1u32].y),
                bitcast_u32(data[1u32].z),
            ]
        }

        fn from_array(array: [u32; 6]) -> Mat2x3f {
            mat2x3f(
                vec3f(
                    bitcast_f32(array[0]),
                    bitcast_f32(array[1]),
                    bitcast_f32(array[2]),
                ),
                vec3f(
                    bitcast_f32(array[3]),
                    bitcast_f32(array[4]),
                    bitcast_f32(array[5]),
                ),
            )
        }

        fn array_container() -> [u32; 6] {
            [0u32, 0u32, 0u32, 0u32, 0u32, 0u32]
        }
    }

    impl SlabItem for Mat2x4f {
        const SLAB_SIZE: usize = 8;
        type Array = [u32; 8];

        fn to_array(data: Mat2x4f) -> [u32; 8] {
            [
                bitcast_u32(data[0u32].x),
                bitcast_u32(data[0u32].y),
                bitcast_u32(data[0u32].z),
                bitcast_u32(data[0u32].w),
                bitcast_u32(data[1u32].x),
                bitcast_u32(data[1u32].y),
                bitcast_u32(data[1u32].z),
                bitcast_u32(data[1u32].w),
            ]
        }

        fn from_array(array: [u32; 8]) -> Mat2x4f {
            mat2x4f(
                vec4f(
                    bitcast_f32(array[0]),
                    bitcast_f32(array[1]),
                    bitcast_f32(array[2]),
                    bitcast_f32(array[3]),
                ),
                vec4f(
                    bitcast_f32(array[4]),
                    bitcast_f32(array[5]),
                    bitcast_f32(array[6]),
                    bitcast_f32(array[7]),
                ),
            )
        }

        fn array_container() -> [u32; 8] {
            [0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32]
        }
    }

    impl SlabItem for Mat3x2f {
        const SLAB_SIZE: usize = 6;
        type Array = [u32; 6];

        fn to_array(data: Mat3x2f) -> [u32; 6] {
            [
                bitcast_u32(data[0u32].x),
                bitcast_u32(data[0u32].y),
                bitcast_u32(data[1u32].x),
                bitcast_u32(data[1u32].y),
                bitcast_u32(data[2u32].x),
                bitcast_u32(data[2u32].y),
            ]
        }

        fn from_array(array: [u32; 6]) -> Mat3x2f {
            mat3x2f(
                vec2f(bitcast_f32(array[0]), bitcast_f32(array[1])),
                vec2f(bitcast_f32(array[2]), bitcast_f32(array[3])),
                vec2f(bitcast_f32(array[4]), bitcast_f32(array[5])),
            )
        }

        fn array_container() -> [u32; 6] {
            [0u32, 0u32, 0u32, 0u32, 0u32, 0u32]
        }
    }

    impl SlabItem for Mat3x3f {
        const SLAB_SIZE: usize = 9;
        type Array = [u32; 9];

        fn to_array(data: Mat3x3f) -> [u32; 9] {
            [
                bitcast_u32(data[0u32].x),
                bitcast_u32(data[0u32].y),
                bitcast_u32(data[0u32].z),
                bitcast_u32(data[1u32].x),
                bitcast_u32(data[1u32].y),
                bitcast_u32(data[1u32].z),
                bitcast_u32(data[2u32].x),
                bitcast_u32(data[2u32].y),
                bitcast_u32(data[2u32].z),
            ]
        }

        fn from_array(array: [u32; 9]) -> Mat3x3f {
            mat3x3f(
                vec3f(
                    bitcast_f32(array[0]),
                    bitcast_f32(array[1]),
                    bitcast_f32(array[2]),
                ),
                vec3f(
                    bitcast_f32(array[3]),
                    bitcast_f32(array[4]),
                    bitcast_f32(array[5]),
                ),
                vec3f(
                    bitcast_f32(array[6]),
                    bitcast_f32(array[7]),
                    bitcast_f32(array[8]),
                ),
            )
        }

        fn array_container() -> [u32; 9] {
            [0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32]
        }
    }

    impl SlabItem for Mat3x4f {
        const SLAB_SIZE: usize = 12;
        type Array = [u32; 12];

        fn to_array(data: Mat3x4f) -> [u32; 12] {
            [
                bitcast_u32(data[0u32].x),
                bitcast_u32(data[0u32].y),
                bitcast_u32(data[0u32].z),
                bitcast_u32(data[0u32].w),
                bitcast_u32(data[1u32].x),
                bitcast_u32(data[1u32].y),
                bitcast_u32(data[1u32].z),
                bitcast_u32(data[1u32].w),
                bitcast_u32(data[2u32].x),
                bitcast_u32(data[2u32].y),
                bitcast_u32(data[2u32].z),
                bitcast_u32(data[2u32].w),
            ]
        }

        fn from_array(array: [u32; 12]) -> Mat3x4f {
            mat3x4f(
                vec4f(
                    bitcast_f32(array[0]),
                    bitcast_f32(array[1]),
                    bitcast_f32(array[2]),
                    bitcast_f32(array[3]),
                ),
                vec4f(
                    bitcast_f32(array[4]),
                    bitcast_f32(array[5]),
                    bitcast_f32(array[6]),
                    bitcast_f32(array[7]),
                ),
                vec4f(
                    bitcast_f32(array[8]),
                    bitcast_f32(array[9]),
                    bitcast_f32(array[10]),
                    bitcast_f32(array[11]),
                ),
            )
        }

        fn array_container() -> [u32; 12] {
            [
                0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32,
            ]
        }
    }

    impl SlabItem for Mat4x2f {
        const SLAB_SIZE: usize = 8;
        type Array = [u32; 8];

        fn to_array(data: Mat4x2f) -> [u32; 8] {
            [
                bitcast_u32(data[0u32].x),
                bitcast_u32(data[0u32].y),
                bitcast_u32(data[1u32].x),
                bitcast_u32(data[1u32].y),
                bitcast_u32(data[2u32].x),
                bitcast_u32(data[2u32].y),
                bitcast_u32(data[3u32].x),
                bitcast_u32(data[3u32].y),
            ]
        }

        fn from_array(array: [u32; 8]) -> Mat4x2f {
            mat4x2f(
                vec2f(bitcast_f32(array[0]), bitcast_f32(array[1])),
                vec2f(bitcast_f32(array[2]), bitcast_f32(array[3])),
                vec2f(bitcast_f32(array[4]), bitcast_f32(array[5])),
                vec2f(bitcast_f32(array[6]), bitcast_f32(array[7])),
            )
        }

        fn array_container() -> [u32; 8] {
            [0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32]
        }
    }

    impl SlabItem for Mat4x3f {
        const SLAB_SIZE: usize = 12;
        type Array = [u32; 12];

        fn to_array(data: Mat4x3f) -> [u32; 12] {
            [
                bitcast_u32(data[0u32].x),
                bitcast_u32(data[0u32].y),
                bitcast_u32(data[0u32].z),
                bitcast_u32(data[1u32].x),
                bitcast_u32(data[1u32].y),
                bitcast_u32(data[1u32].z),
                bitcast_u32(data[2u32].x),
                bitcast_u32(data[2u32].y),
                bitcast_u32(data[2u32].z),
                bitcast_u32(data[3u32].x),
                bitcast_u32(data[3u32].y),
                bitcast_u32(data[3u32].z),
            ]
        }

        fn from_array(array: [u32; 12]) -> Mat4x3f {
            mat4x3f(
                vec3f(
                    bitcast_f32(array[0]),
                    bitcast_f32(array[1]),
                    bitcast_f32(array[2]),
                ),
                vec3f(
                    bitcast_f32(array[3]),
                    bitcast_f32(array[4]),
                    bitcast_f32(array[5]),
                ),
                vec3f(
                    bitcast_f32(array[6]),
                    bitcast_f32(array[7]),
                    bitcast_f32(array[8]),
                ),
                vec3f(
                    bitcast_f32(array[9]),
                    bitcast_f32(array[10]),
                    bitcast_f32(array[11]),
                ),
            )
        }

        fn array_container() -> [u32; 12] {
            [
                0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32,
            ]
        }
    }

    impl SlabItem for Mat4x4f {
        const SLAB_SIZE: usize = 16;
        type Array = [u32; 16];

        fn to_array(data: Mat4x4f) -> [u32; 16] {
            [
                bitcast_u32(data[0u32].x),
                bitcast_u32(data[0u32].y),
                bitcast_u32(data[0u32].z),
                bitcast_u32(data[0u32].w),
                bitcast_u32(data[1u32].x),
                bitcast_u32(data[1u32].y),
                bitcast_u32(data[1u32].z),
                bitcast_u32(data[1u32].w),
                bitcast_u32(data[2u32].x),
                bitcast_u32(data[2u32].y),
                bitcast_u32(data[2u32].z),
                bitcast_u32(data[2u32].w),
                bitcast_u32(data[3u32].x),
                bitcast_u32(data[3u32].y),
                bitcast_u32(data[3u32].z),
                bitcast_u32(data[3u32].w),
            ]
        }

        fn from_array(array: [u32; 16]) -> Mat4x4f {
            mat4x4f(
                vec4f(
                    bitcast_f32(array[0]),
                    bitcast_f32(array[1]),
                    bitcast_f32(array[2]),
                    bitcast_f32(array[3]),
                ),
                vec4f(
                    bitcast_f32(array[4]),
                    bitcast_f32(array[5]),
                    bitcast_f32(array[6]),
                    bitcast_f32(array[7]),
                ),
                vec4f(
                    bitcast_f32(array[8]),
                    bitcast_f32(array[9]),
                    bitcast_f32(array[10]),
                    bitcast_f32(array[11]),
                ),
                vec4f(
                    bitcast_f32(array[12]),
                    bitcast_f32(array[13]),
                    bitcast_f32(array[14]),
                    bitcast_f32(array[15]),
                ),
            )
        }

        fn array_container() -> [u32; 16] {
            [
                0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32, 0u32,
                0u32, 0u32,
            ]
        }
    }

    #[wgsl_ignore]
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

    /// `u32` value of an [`Id`] that does not point to any item.
    pub const ID_NONE: u32 = 4294967295u32;

    /// A typed index into a slab.
    ///
    /// `Id<T>` identifies a `T` stored at some slot of a slab: it wraps a
    /// `u32` slot index, and the `T` parameter keeps ids of different
    /// slab item types from being mixed up. Ids are plain values that
    /// remain valid as long as the underlying slab slot does. The
    /// CPU-side API (conversion, ordering, arithmetic, slab indexing)
    /// lives in the inherent and trait impls in `id.rs`.
    #[repr(transparent)]
    pub struct Id<T> {
        /// The raw slot index. `u32::MAX` is the non-id.
        pub index: u32,
        pub(crate) _phantom: PhantomData<T>,
    }

    impl<T> SlabItem for Id<T> {
        const SLAB_SIZE: usize = 1;
        type Array = [u32; 1];

        fn to_array(data: Id<T>) -> [u32; 1] {
            [data.index]
        }

        fn from_array(array: [u32; 1]) -> Id<T> {
            Id {
                index: array[0],
                _phantom: PhantomData,
            }
        }

        fn array_container() -> [u32; 1] {
            [0]
        }
    }

    impl<T> Id<T> {
        // https://github.com/schell/wgsl-rs/issues/236
        //
        // Neither Self nor Id<T> work.
        //
        // /// The non-id. Analogous to `None::<Id>`.
        pub const NONE: Id<T> = Id {
            index: ID_NONE,
            _phantom: PhantomData,
        };

        // /// An `Id` that points to the first slot on a slab.
        pub const ZERO: Id<T> = Id {
            index: 0,
            _phantom: PhantomData,
        };

        /// Create an `Id` from a raw slot index.
        pub const fn new(i: u32) -> Id<T> {
            Id {
                index: i,
                _phantom: PhantomData,
            }
        }

        /// The non-id. Analogous to `None::<Id>`.
        pub const fn none() -> Id<T> {
            Id::<T>::new(ID_NONE)
        }

        /// An `Id` that points to the first slot on a slab.
        pub const fn zero() -> Id<T> {
            Id::<T>::new(0)
        }
    }

    /// A pointer to contiguous `T` elements in a slab.
    ///
    /// `id` points at the first element and `len` counts `T` elements;
    /// the id's slot index is always a raw `u32` offset. The CPU-side
    /// API (iteration, element addressing, subslicing) lives in
    /// `array.rs`.
    #[repr(C)]
    pub struct Array<T> {
        /// Identifier of the first element in the array.
        pub id: Id<T>,
        /// Number of `T` elements in the array.
        pub len: u32,
    }

    impl<T> Default for Array<T> {
        fn default() -> Self {
            Array::<T>::none()
        }
    }

    impl<T> SlabItem for Array<T> {
        const SLAB_SIZE: usize = 2;
        type Array = [u32; 2];

        fn to_array(data: Array<T>) -> [u32; 2] {
            [data.id.index, data.len]
        }

        fn from_array(array: [u32; 2]) -> Array<T> {
            Array {
                id: Id {
                    index: array[0],
                    _phantom: PhantomData,
                },
                len: array[1],
            }
        }

        fn array_container() -> [u32; 2] {
            [0, 0]
        }
    }

    impl<T> Array<T> {
        /// The non-array: no id, zero length. Analogous to `None`.
        pub const NONE: Self = Array {
            id: <Id<T>>::NONE,
            len: 0,
        };

        pub const fn none() -> Array<T> {
            Array::<T>::new(Id::<T>::none(), 0)
        }

        /// Create a new array starting at `id` with `len` elements.
        pub const fn new(id: Id<T>, len: u32) -> Self {
            Self { id, len }
        }
    }
}

#[cfg(test)]
#[allow(clippy::approx_constant)]
mod test {
    use crate::{slab::CpuSlabItem, SlabItem};

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

        assert!(bool::from_array(bool::to_array(true)));
        assert!(!bool::from_array(bool::to_array(false)));
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

    #[test]
    fn source() {
        #[allow(dead_code)]
        #[wgsl_rs::wgsl(extensions = [crate::SlabItemExt])]
        pub mod source {
            // https://github.com/schell/wgsl-rs/issues/238
            use crate::*;
            use wgsl_rs::std::*;

            #[derive(Clone, Copy, SlabItem)]
            pub struct Camera {
                pub position: Vec3f,
                pub projection: Mat4f,
            }
        }
        let src = source::WGSL_SOURCE.wgsl_source().unwrap();
        println!("{src}");
        source::WGSL_SOURCE.validate().unwrap();
    }
}
