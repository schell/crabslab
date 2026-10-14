//! CPU-only `SlabItem` impls for glam types, enabled by the `glam` feature.
//!
//! These are plain-Rust impls outside any `#[wgsl]` module: glam types are
//! CPU-side only. When wgsl-rs std vector/matrix impls land
//! ([7iu.1.6](#beads/schell-7iu.1.6)) they will cover the GPU side, and these
//! glam impls stay for downstream compatibility.
use glam::{Mat4, Quat, UVec2, UVec3, UVec4, Vec2, Vec3, Vec4};

use crate::SlabItem;

impl SlabItem for Mat4 {
    const SLAB_SIZE: usize = 16;
    type Array = [u32; 16];

    fn to_array(data: Self) -> Self::Array {
        let mut dest = [0u32; 16];
        for (i, f) in data.to_cols_array().iter().enumerate() {
            dest[i] = f.to_bits();
        }
        dest
    }

    fn from_array(arr: Self::Array) -> Self {
        let mut cols = [0f32; 16];
        for (i, slot) in arr.iter().enumerate() {
            cols[i] = f32::from_bits(*slot);
        }
        Mat4::from_cols_array(&cols)
    }

    fn array_container() -> Self::Array {
        [0u32; 16]
    }
}

impl SlabItem for Vec2 {
    const SLAB_SIZE: usize = 2;
    type Array = [u32; 2];

    fn to_array(data: Self) -> Self::Array {
        [data.x.to_bits(), data.y.to_bits()]
    }

    fn from_array(arr: Self::Array) -> Self {
        Vec2::new(f32::from_bits(arr[0]), f32::from_bits(arr[1]))
    }

    fn array_container() -> Self::Array {
        [0, 0]
    }
}

impl SlabItem for Vec3 {
    const SLAB_SIZE: usize = 3;
    type Array = [u32; 3];

    fn to_array(data: Self) -> Self::Array {
        [data.x.to_bits(), data.y.to_bits(), data.z.to_bits()]
    }

    fn from_array(arr: Self::Array) -> Self {
        Vec3::new(
            f32::from_bits(arr[0]),
            f32::from_bits(arr[1]),
            f32::from_bits(arr[2]),
        )
    }

    fn array_container() -> Self::Array {
        [0, 0, 0]
    }
}

impl SlabItem for Vec4 {
    const SLAB_SIZE: usize = 4;
    type Array = [u32; 4];

    fn to_array(data: Self) -> Self::Array {
        [
            data.x.to_bits(),
            data.y.to_bits(),
            data.z.to_bits(),
            data.w.to_bits(),
        ]
    }

    fn from_array(arr: Self::Array) -> Self {
        Vec4::new(
            f32::from_bits(arr[0]),
            f32::from_bits(arr[1]),
            f32::from_bits(arr[2]),
            f32::from_bits(arr[3]),
        )
    }

    fn array_container() -> Self::Array {
        [0, 0, 0, 0]
    }
}

impl SlabItem for Quat {
    const SLAB_SIZE: usize = 4;
    type Array = [u32; 4];

    fn to_array(data: Self) -> Self::Array {
        [
            data.x.to_bits(),
            data.y.to_bits(),
            data.z.to_bits(),
            data.w.to_bits(),
        ]
    }

    fn from_array(arr: Self::Array) -> Self {
        Quat::from_xyzw(
            f32::from_bits(arr[0]),
            f32::from_bits(arr[1]),
            f32::from_bits(arr[2]),
            f32::from_bits(arr[3]),
        )
    }

    fn array_container() -> Self::Array {
        [0, 0, 0, 0]
    }
}

impl SlabItem for UVec2 {
    const SLAB_SIZE: usize = 2;
    type Array = [u32; 2];

    fn to_array(data: Self) -> Self::Array {
        [data.x, data.y]
    }

    fn from_array(arr: Self::Array) -> Self {
        UVec2::new(arr[0], arr[1])
    }

    fn array_container() -> Self::Array {
        [0, 0]
    }
}

impl SlabItem for UVec3 {
    const SLAB_SIZE: usize = 3;
    type Array = [u32; 3];

    fn to_array(data: Self) -> Self::Array {
        [data.x, data.y, data.z]
    }

    fn from_array(arr: Self::Array) -> Self {
        UVec3::new(arr[0], arr[1], arr[2])
    }

    fn array_container() -> Self::Array {
        [0, 0, 0]
    }
}

impl SlabItem for UVec4 {
    const SLAB_SIZE: usize = 4;
    type Array = [u32; 4];

    fn to_array(data: Self) -> Self::Array {
        [data.x, data.y, data.z, data.w]
    }

    fn from_array(arr: Self::Array) -> Self {
        UVec4::new(arr[0], arr[1], arr[2], arr[3])
    }

    fn array_container() -> Self::Array {
        [0, 0, 0, 0]
    }
}
