//! [`SlabItem`] implementations for `wgsl-rs` vector and matrix types.
//!
//! This module is conditionally compiled when the `wgsl-rs` feature is
//! enabled. It provides [`SlabItem`] for all 12 concrete vector types
//! and all 9 matrix types defined in `wgsl_rs::std`.
//!
//! ## Vector types
//!
//! | Type   | Element | SLAB_SIZE |
//! |--------|---------|-----------|
//! | Vec2f  | f32     | 2         |
//! | Vec3f  | f32     | 3         |
//! | Vec4f  | f32     | 4         |
//! | Vec2i  | i32     | 2         |
//! | Vec3i  | i32     | 3         |
//! | Vec4i  | i32     | 4         |
//! | Vec2u  | u32     | 2         |
//! | Vec3u  | u32     | 3         |
//! | Vec4u  | u32     | 4         |
//! | Vec2b  | bool    | 2         |
//! | Vec3b  | bool    | 3         |
//! | Vec4b  | bool    | 4         |
//!
//! ## Matrix types
//!
//! All matrices are f32-only and column-major.
//!
//! | Type     | Columns | Row size | SLAB_SIZE |
//! |----------|---------|----------|-----------|
//! | Mat2x2f  | 2       | 2        | 4         |
//! | Mat2x3f  | 2       | 3        | 6         |
//! | Mat2x4f  | 2       | 4        | 8         |
//! | Mat3x2f  | 3       | 2        | 6         |
//! | Mat3x3f  | 3       | 3        | 9         |
//! | Mat3x4f  | 3       | 4        | 12        |
//! | Mat4x2f  | 4       | 2        | 8         |
//! | Mat4x3f  | 4       | 3        | 12        |
//! | Mat4x4f  | 4       | 4        | 16        |
//!
//! The type aliases `Mat2f`, `Mat3f`, `Mat4f` are re-exports of
//! `Mat2x2f`, `Mat3x3f`, `Mat4x4f` and inherit their impls automatically.

use wgsl_rs::std::{
    Mat2x2f, Mat2x3f, Mat2x4f, Mat3x2f, Mat3x3f, Mat3x4f, Mat4x2f, Mat4x3f, Mat4x4f, Vec2, Vec3,
    Vec4,
};

use crate::SlabItem;

// ---------------------------------------------------------------------------
// Helper: f32 component serialization
// ---------------------------------------------------------------------------

#[inline]
fn f32_to_u32(v: f32) -> u32 {
    v.to_bits()
}

#[inline]
fn u32_to_f32(v: u32) -> f32 {
    f32::from_bits(v)
}

// ---------------------------------------------------------------------------
// Vec2<f32>
// ---------------------------------------------------------------------------

impl SlabItem for Vec2<f32> {
    const SLAB_SIZE: usize = 2;
    type Array = [u32; 2];

    fn to_array(&self) -> [u32; 2] {
        [f32_to_u32(self.x), f32_to_u32(self.y)]
    }

    fn from_array(arr: [u32; 2]) -> Self {
        Vec2 {
            x: u32_to_f32(arr[0]),
            y: u32_to_f32(arr[1]),
        }
    }
}

// ---------------------------------------------------------------------------
// Vec3<f32>
// ---------------------------------------------------------------------------

impl SlabItem for Vec3<f32> {
    const SLAB_SIZE: usize = 3;
    type Array = [u32; 3];

    fn to_array(&self) -> [u32; 3] {
        [f32_to_u32(self.x), f32_to_u32(self.y), f32_to_u32(self.z)]
    }

    fn from_array(arr: [u32; 3]) -> Self {
        Vec3 {
            x: u32_to_f32(arr[0]),
            y: u32_to_f32(arr[1]),
            z: u32_to_f32(arr[2]),
        }
    }
}

// ---------------------------------------------------------------------------
// Vec4<f32>
// ---------------------------------------------------------------------------

impl SlabItem for Vec4<f32> {
    const SLAB_SIZE: usize = 4;
    type Array = [u32; 4];

    fn to_array(&self) -> [u32; 4] {
        [
            f32_to_u32(self.x),
            f32_to_u32(self.y),
            f32_to_u32(self.z),
            f32_to_u32(self.w),
        ]
    }

    fn from_array(arr: [u32; 4]) -> Self {
        Vec4 {
            x: u32_to_f32(arr[0]),
            y: u32_to_f32(arr[1]),
            z: u32_to_f32(arr[2]),
            w: u32_to_f32(arr[3]),
        }
    }
}

// ---------------------------------------------------------------------------
// Vec2<i32>
// ---------------------------------------------------------------------------

impl SlabItem for Vec2<i32> {
    const SLAB_SIZE: usize = 2;
    type Array = [u32; 2];

    fn to_array(&self) -> [u32; 2] {
        [self.x as u32, self.y as u32]
    }

    fn from_array(arr: [u32; 2]) -> Self {
        Vec2 {
            x: arr[0] as i32,
            y: arr[1] as i32,
        }
    }
}

// ---------------------------------------------------------------------------
// Vec3<i32>
// ---------------------------------------------------------------------------

impl SlabItem for Vec3<i32> {
    const SLAB_SIZE: usize = 3;
    type Array = [u32; 3];

    fn to_array(&self) -> [u32; 3] {
        [self.x as u32, self.y as u32, self.z as u32]
    }

    fn from_array(arr: [u32; 3]) -> Self {
        Vec3 {
            x: arr[0] as i32,
            y: arr[1] as i32,
            z: arr[2] as i32,
        }
    }
}

// ---------------------------------------------------------------------------
// Vec4<i32>
// ---------------------------------------------------------------------------

impl SlabItem for Vec4<i32> {
    const SLAB_SIZE: usize = 4;
    type Array = [u32; 4];

    fn to_array(&self) -> [u32; 4] {
        [self.x as u32, self.y as u32, self.z as u32, self.w as u32]
    }

    fn from_array(arr: [u32; 4]) -> Self {
        Vec4 {
            x: arr[0] as i32,
            y: arr[1] as i32,
            z: arr[2] as i32,
            w: arr[3] as i32,
        }
    }
}

// ---------------------------------------------------------------------------
// Vec2<u32>
// ---------------------------------------------------------------------------

impl SlabItem for Vec2<u32> {
    const SLAB_SIZE: usize = 2;
    type Array = [u32; 2];

    fn to_array(&self) -> [u32; 2] {
        [self.x, self.y]
    }

    fn from_array(arr: [u32; 2]) -> Self {
        Vec2 {
            x: arr[0],
            y: arr[1],
        }
    }
}

// ---------------------------------------------------------------------------
// Vec3<u32>
// ---------------------------------------------------------------------------

impl SlabItem for Vec3<u32> {
    const SLAB_SIZE: usize = 3;
    type Array = [u32; 3];

    fn to_array(&self) -> [u32; 3] {
        [self.x, self.y, self.z]
    }

    fn from_array(arr: [u32; 3]) -> Self {
        Vec3 {
            x: arr[0],
            y: arr[1],
            z: arr[2],
        }
    }
}

// ---------------------------------------------------------------------------
// Vec4<u32>
// ---------------------------------------------------------------------------

impl SlabItem for Vec4<u32> {
    const SLAB_SIZE: usize = 4;
    type Array = [u32; 4];

    fn to_array(&self) -> [u32; 4] {
        [self.x, self.y, self.z, self.w]
    }

    fn from_array(arr: [u32; 4]) -> Self {
        Vec4 {
            x: arr[0],
            y: arr[1],
            z: arr[2],
            w: arr[3],
        }
    }
}

// ---------------------------------------------------------------------------
// Vec2<bool>
// ---------------------------------------------------------------------------

impl SlabItem for Vec2<bool> {
    const SLAB_SIZE: usize = 2;
    type Array = [u32; 2];

    fn to_array(&self) -> [u32; 2] {
        [
            if self.x { 1u32 } else { 0u32 },
            if self.y { 1u32 } else { 0u32 },
        ]
    }

    fn from_array(arr: [u32; 2]) -> Self {
        Vec2 {
            x: arr[0] != 0,
            y: arr[1] != 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Vec3<bool>
// ---------------------------------------------------------------------------

impl SlabItem for Vec3<bool> {
    const SLAB_SIZE: usize = 3;
    type Array = [u32; 3];

    fn to_array(&self) -> [u32; 3] {
        [
            if self.x { 1u32 } else { 0u32 },
            if self.y { 1u32 } else { 0u32 },
            if self.z { 1u32 } else { 0u32 },
        ]
    }

    fn from_array(arr: [u32; 3]) -> Self {
        Vec3 {
            x: arr[0] != 0,
            y: arr[1] != 0,
            z: arr[2] != 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Vec4<bool>
// ---------------------------------------------------------------------------

impl SlabItem for Vec4<bool> {
    const SLAB_SIZE: usize = 4;
    type Array = [u32; 4];

    fn to_array(&self) -> [u32; 4] {
        [
            if self.x { 1u32 } else { 0u32 },
            if self.y { 1u32 } else { 0u32 },
            if self.z { 1u32 } else { 0u32 },
            if self.w { 1u32 } else { 0u32 },
        ]
    }

    fn from_array(arr: [u32; 4]) -> Self {
        Vec4 {
            x: arr[0] != 0,
            y: arr[1] != 0,
            z: arr[2] != 0,
            w: arr[3] != 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Matrix helper: serialize/deserialize columns of f32 vectors
// ---------------------------------------------------------------------------

/// Write a `Vec2<f32>` column into a `u32` slice at `offset`.
#[inline]
fn write_vec2f(arr: &mut [u32], offset: usize, v: &Vec2<f32>) {
    arr[offset] = f32_to_u32(v.x);
    arr[offset + 1] = f32_to_u32(v.y);
}

/// Read a `Vec2<f32>` column from a `u32` slice at `offset`.
#[inline]
fn read_vec2f(arr: &[u32], offset: usize) -> Vec2<f32> {
    Vec2 {
        x: u32_to_f32(arr[offset]),
        y: u32_to_f32(arr[offset + 1]),
    }
}

/// Write a `Vec3<f32>` column into a `u32` slice at `offset`.
#[inline]
fn write_vec3f(arr: &mut [u32], offset: usize, v: &Vec3<f32>) {
    arr[offset] = f32_to_u32(v.x);
    arr[offset + 1] = f32_to_u32(v.y);
    arr[offset + 2] = f32_to_u32(v.z);
}

/// Read a `Vec3<f32>` column from a `u32` slice at `offset`.
#[inline]
fn read_vec3f(arr: &[u32], offset: usize) -> Vec3<f32> {
    Vec3 {
        x: u32_to_f32(arr[offset]),
        y: u32_to_f32(arr[offset + 1]),
        z: u32_to_f32(arr[offset + 2]),
    }
}

/// Write a `Vec4<f32>` column into a `u32` slice at `offset`.
#[inline]
fn write_vec4f(arr: &mut [u32], offset: usize, v: &Vec4<f32>) {
    arr[offset] = f32_to_u32(v.x);
    arr[offset + 1] = f32_to_u32(v.y);
    arr[offset + 2] = f32_to_u32(v.z);
    arr[offset + 3] = f32_to_u32(v.w);
}

/// Read a `Vec4<f32>` column from a `u32` slice at `offset`.
#[inline]
fn read_vec4f(arr: &[u32], offset: usize) -> Vec4<f32> {
    Vec4 {
        x: u32_to_f32(arr[offset]),
        y: u32_to_f32(arr[offset + 1]),
        z: u32_to_f32(arr[offset + 2]),
        w: u32_to_f32(arr[offset + 3]),
    }
}

// ---------------------------------------------------------------------------
// Mat2x2f (2 columns of Vec2f, 4 u32 slots)
// ---------------------------------------------------------------------------

impl SlabItem for Mat2x2f {
    const SLAB_SIZE: usize = 4;
    type Array = [u32; 4];

    fn to_array(&self) -> [u32; 4] {
        let mut arr = [0u32; 4];
        write_vec2f(&mut arr, 0, &self[0usize]);
        write_vec2f(&mut arr, 2, &self[1usize]);
        arr
    }

    fn from_array(arr: [u32; 4]) -> Self {
        wgsl_rs::std::mat2x2f(read_vec2f(&arr, 0), read_vec2f(&arr, 2))
    }
}

// ---------------------------------------------------------------------------
// Mat2x3f (2 columns of Vec3f, 6 u32 slots)
// ---------------------------------------------------------------------------

impl SlabItem for Mat2x3f {
    const SLAB_SIZE: usize = 6;
    type Array = [u32; 6];

    fn to_array(&self) -> [u32; 6] {
        let mut arr = [0u32; 6];
        write_vec3f(&mut arr, 0, &self[0usize]);
        write_vec3f(&mut arr, 3, &self[1usize]);
        arr
    }

    fn from_array(arr: [u32; 6]) -> Self {
        wgsl_rs::std::mat2x3f(read_vec3f(&arr, 0), read_vec3f(&arr, 3))
    }
}

// ---------------------------------------------------------------------------
// Mat2x4f (2 columns of Vec4f, 8 u32 slots)
// ---------------------------------------------------------------------------

impl SlabItem for Mat2x4f {
    const SLAB_SIZE: usize = 8;
    type Array = [u32; 8];

    fn to_array(&self) -> [u32; 8] {
        let mut arr = [0u32; 8];
        write_vec4f(&mut arr, 0, &self[0usize]);
        write_vec4f(&mut arr, 4, &self[1usize]);
        arr
    }

    fn from_array(arr: [u32; 8]) -> Self {
        wgsl_rs::std::mat2x4f(read_vec4f(&arr, 0), read_vec4f(&arr, 4))
    }
}

// ---------------------------------------------------------------------------
// Mat3x2f (3 columns of Vec2f, 6 u32 slots)
// ---------------------------------------------------------------------------

impl SlabItem for Mat3x2f {
    const SLAB_SIZE: usize = 6;
    type Array = [u32; 6];

    fn to_array(&self) -> [u32; 6] {
        let mut arr = [0u32; 6];
        write_vec2f(&mut arr, 0, &self[0usize]);
        write_vec2f(&mut arr, 2, &self[1usize]);
        write_vec2f(&mut arr, 4, &self[2usize]);
        arr
    }

    fn from_array(arr: [u32; 6]) -> Self {
        wgsl_rs::std::mat3x2f(
            read_vec2f(&arr, 0),
            read_vec2f(&arr, 2),
            read_vec2f(&arr, 4),
        )
    }
}

// ---------------------------------------------------------------------------
// Mat3x3f (3 columns of Vec3f, 9 u32 slots)
// ---------------------------------------------------------------------------

impl SlabItem for Mat3x3f {
    const SLAB_SIZE: usize = 9;
    type Array = [u32; 9];

    fn to_array(&self) -> [u32; 9] {
        let mut arr = [0u32; 9];
        write_vec3f(&mut arr, 0, &self[0usize]);
        write_vec3f(&mut arr, 3, &self[1usize]);
        write_vec3f(&mut arr, 6, &self[2usize]);
        arr
    }

    fn from_array(arr: [u32; 9]) -> Self {
        wgsl_rs::std::mat3x3f(
            read_vec3f(&arr, 0),
            read_vec3f(&arr, 3),
            read_vec3f(&arr, 6),
        )
    }
}

// ---------------------------------------------------------------------------
// Mat3x4f (3 columns of Vec4f, 12 u32 slots)
// ---------------------------------------------------------------------------

impl SlabItem for Mat3x4f {
    const SLAB_SIZE: usize = 12;
    type Array = [u32; 12];

    fn to_array(&self) -> [u32; 12] {
        let mut arr = [0u32; 12];
        write_vec4f(&mut arr, 0, &self[0usize]);
        write_vec4f(&mut arr, 4, &self[1usize]);
        write_vec4f(&mut arr, 8, &self[2usize]);
        arr
    }

    fn from_array(arr: [u32; 12]) -> Self {
        wgsl_rs::std::mat3x4f(
            read_vec4f(&arr, 0),
            read_vec4f(&arr, 4),
            read_vec4f(&arr, 8),
        )
    }
}

// ---------------------------------------------------------------------------
// Mat4x2f (4 columns of Vec2f, 8 u32 slots)
// ---------------------------------------------------------------------------

impl SlabItem for Mat4x2f {
    const SLAB_SIZE: usize = 8;
    type Array = [u32; 8];

    fn to_array(&self) -> [u32; 8] {
        let mut arr = [0u32; 8];
        write_vec2f(&mut arr, 0, &self[0usize]);
        write_vec2f(&mut arr, 2, &self[1usize]);
        write_vec2f(&mut arr, 4, &self[2usize]);
        write_vec2f(&mut arr, 6, &self[3usize]);
        arr
    }

    fn from_array(arr: [u32; 8]) -> Self {
        wgsl_rs::std::mat4x2f(
            read_vec2f(&arr, 0),
            read_vec2f(&arr, 2),
            read_vec2f(&arr, 4),
            read_vec2f(&arr, 6),
        )
    }
}

// ---------------------------------------------------------------------------
// Mat4x3f (4 columns of Vec3f, 12 u32 slots)
// ---------------------------------------------------------------------------

impl SlabItem for Mat4x3f {
    const SLAB_SIZE: usize = 12;
    type Array = [u32; 12];

    fn to_array(&self) -> [u32; 12] {
        let mut arr = [0u32; 12];
        write_vec3f(&mut arr, 0, &self[0usize]);
        write_vec3f(&mut arr, 3, &self[1usize]);
        write_vec3f(&mut arr, 6, &self[2usize]);
        write_vec3f(&mut arr, 9, &self[3usize]);
        arr
    }

    fn from_array(arr: [u32; 12]) -> Self {
        wgsl_rs::std::mat4x3f(
            read_vec3f(&arr, 0),
            read_vec3f(&arr, 3),
            read_vec3f(&arr, 6),
            read_vec3f(&arr, 9),
        )
    }
}

// ---------------------------------------------------------------------------
// Mat4x4f (4 columns of Vec4f, 16 u32 slots)
// ---------------------------------------------------------------------------

impl SlabItem for Mat4x4f {
    const SLAB_SIZE: usize = 16;
    type Array = [u32; 16];

    fn to_array(&self) -> [u32; 16] {
        let mut arr = [0u32; 16];
        write_vec4f(&mut arr, 0, &self[0usize]);
        write_vec4f(&mut arr, 4, &self[1usize]);
        write_vec4f(&mut arr, 8, &self[2usize]);
        write_vec4f(&mut arr, 12, &self[3usize]);
        arr
    }

    fn from_array(arr: [u32; 16]) -> Self {
        wgsl_rs::std::mat4x4f(
            read_vec4f(&arr, 0),
            read_vec4f(&arr, 4),
            read_vec4f(&arr, 8),
            read_vec4f(&arr, 12),
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod test {
    use super::*;
    use crate::{slab_read, slab_write};
    use wgsl_rs::std::*;

    /// Helper: round-trip a value through SlabItem's `to_array`/`from_array`.
    fn rt<T: SlabItem + Copy>(v: &T) -> T {
        T::from_array(v.to_array())
    }

    // -- Vector round-trip tests ------------------------------------------

    #[test]
    fn vec2f_round_trip() {
        let v = vec2f(1.0, -2.5);
        assert_eq!(v, rt(&v));

        let mut slab = [0u32; 4];
        slab_write(&mut slab, 0, &v);
        let v3: Vec2<f32> = slab_read(&slab, 0);
        assert_eq!(v, v3);

        // Non-zero offset.
        slab_write(&mut slab, 2, &v);
        let v4: Vec2<f32> = slab_read(&slab, 2);
        assert_eq!(v, v4);
    }

    #[test]
    fn vec3f_round_trip() {
        let v = vec3f(3.14, 0.0, -1.0);
        assert_eq!(v, rt(&v));

        let mut slab = [0u32; 6];
        slab_write(&mut slab, 0, &v);
        let v3: Vec3<f32> = slab_read(&slab, 0);
        assert_eq!(v, v3);

        slab_write(&mut slab, 3, &v);
        let v4: Vec3<f32> = slab_read(&slab, 3);
        assert_eq!(v, v4);
    }

    #[test]
    fn vec4f_round_trip() {
        let v = vec4f(1.0, 2.0, 3.0, 4.0);
        assert_eq!(v, rt(&v));

        let mut slab = [0u32; 8];
        slab_write(&mut slab, 0, &v);
        let v3: Vec4<f32> = slab_read(&slab, 0);
        assert_eq!(v, v3);

        slab_write(&mut slab, 4, &v);
        let v4: Vec4<f32> = slab_read(&slab, 4);
        assert_eq!(v, v4);
    }

    #[test]
    fn vec2i_round_trip() {
        let v = vec2i(-1, 42);
        assert_eq!(v, rt(&v));

        let mut slab = [0u32; 4];
        slab_write(&mut slab, 0, &v);
        let v3: Vec2<i32> = slab_read(&slab, 0);
        assert_eq!(v, v3);
    }

    #[test]
    fn vec3i_round_trip() {
        let v = vec3i(i32::MIN, 0, i32::MAX);
        assert_eq!(v, rt(&v));

        let mut slab = [0u32; 6];
        slab_write(&mut slab, 0, &v);
        let v3: Vec3<i32> = slab_read(&slab, 0);
        assert_eq!(v, v3);
    }

    #[test]
    fn vec4i_round_trip() {
        let v = vec4i(-100, 0, 100, -1);
        assert_eq!(v, rt(&v));

        let mut slab = [0u32; 8];
        slab_write(&mut slab, 0, &v);
        let v3: Vec4<i32> = slab_read(&slab, 0);
        assert_eq!(v, v3);
    }

    #[test]
    fn vec2u_round_trip() {
        let v = vec2u(0, u32::MAX);
        assert_eq!(v, rt(&v));

        let mut slab = [0u32; 4];
        slab_write(&mut slab, 0, &v);
        let v3: Vec2<u32> = slab_read(&slab, 0);
        assert_eq!(v, v3);
    }

    #[test]
    fn vec3u_round_trip() {
        let v = vec3u(1, 2, 3);
        assert_eq!(v, rt(&v));

        let mut slab = [0u32; 6];
        slab_write(&mut slab, 0, &v);
        let v3: Vec3<u32> = slab_read(&slab, 0);
        assert_eq!(v, v3);
    }

    #[test]
    fn vec4u_round_trip() {
        let v = vec4u(10, 20, 30, 40);
        assert_eq!(v, rt(&v));

        let mut slab = [0u32; 8];
        slab_write(&mut slab, 0, &v);
        let v3: Vec4<u32> = slab_read(&slab, 0);
        assert_eq!(v, v3);
    }

    #[test]
    fn vec2b_round_trip() {
        let v = vec2b(true, false);
        assert_eq!(v, rt(&v));

        let mut slab = [0u32; 4];
        slab_write(&mut slab, 0, &v);
        let v3: Vec2<bool> = slab_read(&slab, 0);
        assert_eq!(v, v3);
    }

    #[test]
    fn vec3b_round_trip() {
        let v = vec3b(false, true, false);
        assert_eq!(v, rt(&v));

        let mut slab = [0u32; 6];
        slab_write(&mut slab, 0, &v);
        let v3: Vec3<bool> = slab_read(&slab, 0);
        assert_eq!(v, v3);
    }

    #[test]
    fn vec4b_round_trip() {
        let v = vec4b(true, true, false, true);
        assert_eq!(v, rt(&v));

        let mut slab = [0u32; 8];
        slab_write(&mut slab, 0, &v);
        let v3: Vec4<bool> = slab_read(&slab, 0);
        assert_eq!(v, v3);
    }

    // -- Vector edge cases ------------------------------------------------

    #[test]
    fn vec_f32_special_values() {
        // NaN, infinity, negative zero.
        let v = vec4f(f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.0);
        let v2 = rt(&v);
        assert!(v2.x.is_nan());
        assert_eq!(f32::INFINITY, v2.y);
        assert_eq!(f32::NEG_INFINITY, v2.z);
        assert_eq!((-0.0f32).to_bits(), v2.w.to_bits());
    }

    // -- Vector SLAB_SIZE constants ---------------------------------------

    #[test]
    fn vector_slab_sizes() {
        assert_eq!(2, <Vec2<f32> as SlabItem>::SLAB_SIZE);
        assert_eq!(3, <Vec3<f32> as SlabItem>::SLAB_SIZE);
        assert_eq!(4, <Vec4<f32> as SlabItem>::SLAB_SIZE);

        assert_eq!(2, <Vec2<i32> as SlabItem>::SLAB_SIZE);
        assert_eq!(3, <Vec3<i32> as SlabItem>::SLAB_SIZE);
        assert_eq!(4, <Vec4<i32> as SlabItem>::SLAB_SIZE);

        assert_eq!(2, <Vec2<u32> as SlabItem>::SLAB_SIZE);
        assert_eq!(3, <Vec3<u32> as SlabItem>::SLAB_SIZE);
        assert_eq!(4, <Vec4<u32> as SlabItem>::SLAB_SIZE);

        assert_eq!(2, <Vec2<bool> as SlabItem>::SLAB_SIZE);
        assert_eq!(3, <Vec3<bool> as SlabItem>::SLAB_SIZE);
        assert_eq!(4, <Vec4<bool> as SlabItem>::SLAB_SIZE);
    }

    // -- Matrix round-trip tests ------------------------------------------

    /// Helper to compare two matrices by column.
    fn assert_mat_eq<M: std::ops::Index<usize, Output = V>, V: PartialEq + std::fmt::Debug>(
        a: &M,
        b: &M,
        cols: usize,
    ) {
        for c in 0..cols {
            assert_eq!(a[c], b[c], "column {c} mismatch");
        }
    }

    #[test]
    fn mat2x2f_round_trip() {
        let m = mat2x2f(vec2f(1.0, 2.0), vec2f(3.0, 4.0));
        let m2 = rt(&m);
        assert_mat_eq(&m, &m2, 2);

        let mut slab = [0u32; 8];
        slab_write(&mut slab, 0, &m);
        let m3: Mat2x2f = slab_read(&slab, 0);
        assert_mat_eq(&m, &m3, 2);
    }

    #[test]
    fn mat2x3f_round_trip() {
        let m = mat2x3f(vec3f(1.0, 2.0, 3.0), vec3f(4.0, 5.0, 6.0));
        let m2 = rt(&m);
        assert_mat_eq(&m, &m2, 2);

        let mut slab = [0u32; 12];
        slab_write(&mut slab, 0, &m);
        let m3: Mat2x3f = slab_read(&slab, 0);
        assert_mat_eq(&m, &m3, 2);
    }

    #[test]
    fn mat2x4f_round_trip() {
        let m = mat2x4f(vec4f(1.0, 2.0, 3.0, 4.0), vec4f(5.0, 6.0, 7.0, 8.0));
        let m2 = rt(&m);
        assert_mat_eq(&m, &m2, 2);

        let mut slab = [0u32; 16];
        slab_write(&mut slab, 0, &m);
        let m3: Mat2x4f = slab_read(&slab, 0);
        assert_mat_eq(&m, &m3, 2);
    }

    #[test]
    fn mat3x2f_round_trip() {
        let m = mat3x2f(vec2f(1.0, 2.0), vec2f(3.0, 4.0), vec2f(5.0, 6.0));
        let m2 = rt(&m);
        assert_mat_eq(&m, &m2, 3);

        let mut slab = [0u32; 12];
        slab_write(&mut slab, 0, &m);
        let m3: Mat3x2f = slab_read(&slab, 0);
        assert_mat_eq(&m, &m3, 3);
    }

    #[test]
    fn mat3x3f_round_trip() {
        let m = mat3x3f(
            vec3f(1.0, 2.0, 3.0),
            vec3f(4.0, 5.0, 6.0),
            vec3f(7.0, 8.0, 9.0),
        );
        let m2 = rt(&m);
        assert_mat_eq(&m, &m2, 3);

        let mut slab = [0u32; 18];
        slab_write(&mut slab, 0, &m);
        let m3: Mat3x3f = slab_read(&slab, 0);
        assert_mat_eq(&m, &m3, 3);
    }

    #[test]
    fn mat3x4f_round_trip() {
        let m = mat3x4f(
            vec4f(1.0, 2.0, 3.0, 4.0),
            vec4f(5.0, 6.0, 7.0, 8.0),
            vec4f(9.0, 10.0, 11.0, 12.0),
        );
        let m2 = rt(&m);
        assert_mat_eq(&m, &m2, 3);

        let mut slab = [0u32; 24];
        slab_write(&mut slab, 0, &m);
        let m3: Mat3x4f = slab_read(&slab, 0);
        assert_mat_eq(&m, &m3, 3);
    }

    #[test]
    fn mat4x2f_round_trip() {
        let m = mat4x2f(
            vec2f(1.0, 2.0),
            vec2f(3.0, 4.0),
            vec2f(5.0, 6.0),
            vec2f(7.0, 8.0),
        );
        let m2 = rt(&m);
        assert_mat_eq(&m, &m2, 4);

        let mut slab = [0u32; 16];
        slab_write(&mut slab, 0, &m);
        let m3: Mat4x2f = slab_read(&slab, 0);
        assert_mat_eq(&m, &m3, 4);
    }

    #[test]
    fn mat4x3f_round_trip() {
        let m = mat4x3f(
            vec3f(1.0, 2.0, 3.0),
            vec3f(4.0, 5.0, 6.0),
            vec3f(7.0, 8.0, 9.0),
            vec3f(10.0, 11.0, 12.0),
        );
        let m2 = rt(&m);
        assert_mat_eq(&m, &m2, 4);

        let mut slab = [0u32; 24];
        slab_write(&mut slab, 0, &m);
        let m3: Mat4x3f = slab_read(&slab, 0);
        assert_mat_eq(&m, &m3, 4);
    }

    #[test]
    fn mat4x4f_round_trip() {
        let m = mat4x4f(
            vec4f(1.0, 2.0, 3.0, 4.0),
            vec4f(5.0, 6.0, 7.0, 8.0),
            vec4f(9.0, 10.0, 11.0, 12.0),
            vec4f(13.0, 14.0, 15.0, 16.0),
        );
        let m2 = rt(&m);
        assert_mat_eq(&m, &m2, 4);

        let mut slab = [0u32; 32];
        slab_write(&mut slab, 0, &m);
        let m3: Mat4x4f = slab_read(&slab, 0);
        assert_mat_eq(&m, &m3, 4);

        // Non-zero offset.
        slab_write(&mut slab, 16, &m);
        let m4: Mat4x4f = slab_read(&slab, 16);
        assert_mat_eq(&m, &m4, 4);
    }

    // -- Matrix SLAB_SIZE constants ---------------------------------------

    #[test]
    fn matrix_slab_sizes() {
        assert_eq!(4, <Mat2x2f as SlabItem>::SLAB_SIZE);
        assert_eq!(6, <Mat2x3f as SlabItem>::SLAB_SIZE);
        assert_eq!(8, <Mat2x4f as SlabItem>::SLAB_SIZE);
        assert_eq!(6, <Mat3x2f as SlabItem>::SLAB_SIZE);
        assert_eq!(9, <Mat3x3f as SlabItem>::SLAB_SIZE);
        assert_eq!(12, <Mat3x4f as SlabItem>::SLAB_SIZE);
        assert_eq!(8, <Mat4x2f as SlabItem>::SLAB_SIZE);
        assert_eq!(12, <Mat4x3f as SlabItem>::SLAB_SIZE);
        assert_eq!(16, <Mat4x4f as SlabItem>::SLAB_SIZE);
    }

    // -- Matrix raw layout verification -----------------------------------

    #[test]
    fn mat2x2f_raw_layout() {
        let m = mat2x2f(vec2f(1.0, 2.0), vec2f(3.0, 4.0));
        let arr = SlabItem::to_array(&m);
        // Column-major: col0(x,y), col1(x,y).
        assert_eq!(1.0f32.to_bits(), arr[0]);
        assert_eq!(2.0f32.to_bits(), arr[1]);
        assert_eq!(3.0f32.to_bits(), arr[2]);
        assert_eq!(4.0f32.to_bits(), arr[3]);
    }

    #[test]
    fn mat4x4f_raw_layout() {
        let m = mat4x4f(
            vec4f(1.0, 2.0, 3.0, 4.0),
            vec4f(5.0, 6.0, 7.0, 8.0),
            vec4f(9.0, 10.0, 11.0, 12.0),
            vec4f(13.0, 14.0, 15.0, 16.0),
        );
        let arr = SlabItem::to_array(&m);
        // Column-major: col0(x,y,z,w), col1(x,y,z,w), ...
        for i in 0..16 {
            assert_eq!(((i + 1) as f32).to_bits(), arr[i], "mismatch at index {i}");
        }
    }
}
