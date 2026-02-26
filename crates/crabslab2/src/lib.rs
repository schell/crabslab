//! `crabslab2` — slab allocator for CPU/GPU data marshalling.
//!
//! Types that implement [`SlabItem`] can be serialized to and deserialized
//! from `[u32]` slabs, enabling efficient data transfer between CPU and GPU.
//!
//! ## Overview
//!
//! The core trait is [`SlabItem`], which defines how a type is converted
//! to and from a fixed-size `[u32; N]` array. The helper
//! functions [`slab_read`] and [`slab_write`] use this trait to read/write
//! values at arbitrary positions in a `&[u32]` slab.
//!
//! For GPU-shared types, use the `#[slab_module]` and `#[slab_item]`
//! attribute macros to generate both the Rust `SlabItem` impl and the
//! corresponding WGSL serialization functions.

pub use crabslab2_macros::{slab_item, slab_module};

/// CPU-side trait for types that can be stored in a `u32` slab.
///
/// Auto-implemented by the `#[slab_module]` macro for `#[slab_item]` types.
/// Can also be implemented manually for primitive types.
///
/// # Example
///
/// ```
/// use crabslab2::{SlabItem, slab_read, slab_write};
///
/// let val = 42u32;
/// let mut slab = [0u32; 4];
/// slab_write(&mut slab, 0, &val);
/// let read_back: u32 = slab_read(&slab, 0);
/// assert_eq!(val, read_back);
/// ```
pub trait SlabItem: Sized + 'static {
    /// The number of `u32` slots this type occupies in a slab.
    const SLAB_SIZE: usize;

    /// The fixed-size `[u32; Self::SLAB_SIZE]` array type for this slab item.
    type Array: AsRef<[u32]> + AsMut<[u32]> + Default;

    /// Serialize this value into a `[u32; Self::SLAB_SIZE]` array.
    fn to_array(&self) -> Self::Array;

    /// Deserialize a value from a `[u32; Self::SLAB_SIZE]` array.
    fn from_array(arr: Self::Array) -> Self;
}

/// Read a [`SlabItem`] from a `u32` slice at the given index.
///
/// # Panics
///
/// Panics if the slice is too short to contain `T::SLAB_SIZE` elements
/// starting at `index`.
pub fn slab_read<T: SlabItem>(slab: &[u32], index: usize) -> T {
    let mut arr = T::Array::default();
    arr.as_mut()
        .copy_from_slice(&slab[index..index + T::SLAB_SIZE]);
    T::from_array(arr)
}

/// Write a [`SlabItem`] into a `u32` slice at the given index.
///
/// # Panics
///
/// Panics if the slice is too short to contain `T::SLAB_SIZE` elements
/// starting at `index`.
pub fn slab_write<T: SlabItem>(slab: &mut [u32], index: usize, val: &T) {
    let arr = val.to_array();
    slab[index..index + T::SLAB_SIZE].copy_from_slice(arr.as_ref());
}

// ---------------------------------------------------------------------------
// Primitive SlabItem implementations
// ---------------------------------------------------------------------------

impl SlabItem for u32 {
    const SLAB_SIZE: usize = 1;
    type Array = [u32; 1];

    fn to_array(&self) -> [u32; 1] {
        [*self]
    }

    fn from_array(arr: [u32; 1]) -> Self {
        arr[0]
    }
}

impl SlabItem for i32 {
    const SLAB_SIZE: usize = 1;
    type Array = [u32; 1];

    fn to_array(&self) -> [u32; 1] {
        [*self as u32]
    }

    fn from_array(arr: [u32; 1]) -> Self {
        arr[0] as i32
    }
}

impl SlabItem for f32 {
    const SLAB_SIZE: usize = 1;
    type Array = [u32; 1];

    fn to_array(&self) -> [u32; 1] {
        [self.to_bits()]
    }

    fn from_array(arr: [u32; 1]) -> Self {
        f32::from_bits(arr[0])
    }
}

impl SlabItem for bool {
    const SLAB_SIZE: usize = 1;
    type Array = [u32; 1];

    fn to_array(&self) -> [u32; 1] {
        [if *self { 1u32 } else { 0u32 }]
    }

    fn from_array(arr: [u32; 1]) -> Self {
        arr[0] != 0
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn u32_round_trip() {
        let mut slab = [0u32; 4];
        slab_write(&mut slab, 0, &42u32);
        slab_write(&mut slab, 1, &u32::MAX);
        slab_write(&mut slab, 2, &0u32);

        assert_eq!(42u32, slab_read::<u32>(&slab, 0));
        assert_eq!(u32::MAX, slab_read::<u32>(&slab, 1));
        assert_eq!(0u32, slab_read::<u32>(&slab, 2));
    }

    #[test]
    fn i32_round_trip() {
        let mut slab = [0u32; 4];
        slab_write(&mut slab, 0, &-1i32);
        slab_write(&mut slab, 1, &i32::MIN);
        slab_write(&mut slab, 2, &i32::MAX);

        assert_eq!(-1i32, slab_read::<i32>(&slab, 0));
        assert_eq!(i32::MIN, slab_read::<i32>(&slab, 1));
        assert_eq!(i32::MAX, slab_read::<i32>(&slab, 2));
    }

    #[test]
    fn f32_round_trip() {
        let mut slab = [0u32; 4];
        slab_write(&mut slab, 0, &3.14f32);
        slab_write(&mut slab, 1, &-0.0f32);
        slab_write(&mut slab, 2, &f32::INFINITY);

        assert_eq!(3.14f32, slab_read::<f32>(&slab, 0));
        assert_eq!((-0.0f32).to_bits(), slab_read::<f32>(&slab, 1).to_bits());
        assert_eq!(f32::INFINITY, slab_read::<f32>(&slab, 2));
    }

    #[test]
    fn bool_round_trip() {
        let mut slab = [0u32; 2];
        slab_write(&mut slab, 0, &true);
        slab_write(&mut slab, 1, &false);

        assert!(slab_read::<bool>(&slab, 0));
        assert!(!slab_read::<bool>(&slab, 1));
    }

    #[test]
    fn multiple_types_in_slab() {
        let mut slab = [0u32; 4];
        slab_write(&mut slab, 0, &42u32);
        slab_write(&mut slab, 1, &2.718f32);
        slab_write(&mut slab, 2, &-7i32);
        slab_write(&mut slab, 3, &true);

        assert_eq!(42u32, slab_read::<u32>(&slab, 0));
        assert_eq!(2.718f32, slab_read::<f32>(&slab, 1));
        assert_eq!(-7i32, slab_read::<i32>(&slab, 2));
        assert!(slab_read::<bool>(&slab, 3));
    }

    #[test]
    fn slab_size_constants() {
        assert_eq!(1, u32::SLAB_SIZE);
        assert_eq!(1, i32::SLAB_SIZE);
        assert_eq!(1, f32::SLAB_SIZE);
        assert_eq!(1, bool::SLAB_SIZE);
    }

    #[test]
    fn to_array_from_array_identity() {
        let vals_u32 = [0u32, 1, u32::MAX, 42];
        for v in vals_u32 {
            assert_eq!(v, u32::from_array(v.to_array()));
        }

        let vals_i32 = [0i32, 1, -1, i32::MIN, i32::MAX];
        for v in vals_i32 {
            assert_eq!(v, i32::from_array(v.to_array()));
        }

        let vals_f32 = [0.0f32, -0.0, 1.0, -1.0, f32::INFINITY, f32::NAN];
        for v in vals_f32 {
            let rt = f32::from_array(v.to_array());
            assert_eq!(v.to_bits(), rt.to_bits());
        }

        assert_eq!(true, bool::from_array(true.to_array()));
        assert_eq!(false, bool::from_array(false.to_array()));
    }
}
