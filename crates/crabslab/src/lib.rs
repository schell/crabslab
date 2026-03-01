//! `crabslab` — slab allocator for CPU/GPU data marshalling.
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

#[doc(hidden)]
pub extern crate self as crabslab;

pub use crabslab_macros::{slab_item, slab_module};

/// CPU-side trait for types that can be stored in a `u32` slab.
///
/// Auto-implemented by the `#[slab_module]` macro for `#[slab_item]` types.
/// Can also be implemented manually for primitive types.
///
/// # Example
///
/// ```
/// use crabslab::{SlabItem, slab_read, slab_write};
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
// slab_read! / slab_write! macros
// ---------------------------------------------------------------------------

/// Read a [`SlabItem`] from a storage slab at the given offset.
///
/// Inside a `#[slab_module]` this macro is expanded by the proc-macro into
/// `slab_read_array!` + `Type::from_array(...)` before `#[wgsl]` runs, so it
/// works on both CPU and GPU. Outside a `#[slab_module]` the `macro_rules!`
/// fallback delegates to [`slab_read`].
///
/// # Syntax
///
/// ```ignore
/// let value = slab_read!(Type, slab_expr, offset_expr);
/// ```
#[macro_export]
macro_rules! slab_read {
    ($ty:ty, $slab:expr, $offset:expr) => {{
        let mut arr = <$ty as $crate::SlabItem>::Array::default();
        let offset = $offset as usize;
        let slice: &[u32] = &$slab[offset..offset + <$ty as $crate::SlabItem>::SLAB_SIZE];
        arr.as_mut().copy_from_slice(slice);
        <$ty>::from_array(arr)
    }};
}

/// Write a [`SlabItem`] to a storage slab at the given offset.
///
/// Inside a `#[slab_module]` this macro is expanded by the proc-macro into
/// `Type::to_array(...)` + `slab_write_array!` before `#[wgsl]` runs, so it
/// works on both CPU and GPU. Outside a `#[slab_module]` the `macro_rules!`
/// fallback delegates to [`slab_write`].
///
/// # Syntax
///
/// ```ignore
/// slab_write!(Type, slab_expr, offset_expr, value_expr);
/// ```
#[macro_export]
macro_rules! slab_write {
    ($ty:ty, $slab:expr, $offset:expr, $val:expr) => {{
        let arr = <$ty>::to_array($val);
        let offset = $offset as usize;
        let size = <$ty as $crate::SlabItem>::SLAB_SIZE;
        $slab[offset..offset + size].copy_from_slice(arr.as_ref());
    }};
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
pub mod test {
    #![allow(clippy::approx_constant)]
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

        assert!(bool::from_array(true.to_array()));
        assert!(!bool::from_array(false.to_array()));
    }

    // -----------------------------------------------------------------------
    // #[slab_module] integration tests
    // -----------------------------------------------------------------------

    #[slab_module]
    pub mod test_types {
        #[slab_item]
        #[derive(Clone, Copy, Debug, Default, PartialEq)]
        pub struct Data {
            pub i: u32,
            pub float_val: f32,
            pub ints_0: u32,
            pub ints_1: u32,
        }
    }

    #[test]
    fn struct_slab_size() {
        assert_eq!(4, test_types::Data::SLAB_SIZE);
        assert_eq!(4, <test_types::Data as SlabItem>::SLAB_SIZE);
    }

    #[test]
    fn struct_round_trip_via_inherent() {
        let d = test_types::Data {
            i: 42,
            float_val: 3.14,
            ints_0: 1,
            ints_1: 2,
        };
        let arr = test_types::Data::to_array(d);
        let d2 = test_types::Data::from_array(arr);
        assert_eq!(d, d2);
    }

    #[test]
    fn struct_round_trip_via_slab() {
        let d = test_types::Data {
            i: 42,
            float_val: 3.14,
            ints_0: 1,
            ints_1: 2,
        };
        let mut slab = [0u32; 8];
        slab_write(&mut slab, 0, &d);
        let d2: test_types::Data = slab_read(&slab, 0);
        assert_eq!(d, d2);

        // Write at a non-zero offset.
        slab_write(&mut slab, 4, &d);
        let d3: test_types::Data = slab_read(&slab, 4);
        assert_eq!(d, d3);
    }

    #[test]
    fn struct_id_type() {
        assert_eq!(1, test_types::DataId::SLAB_SIZE);

        let id = test_types::DataId::new(5);
        assert_eq!(5, id.inner);
        assert!(!test_types::DataId::is_none(id));

        let none = test_types::DataId::NONE;
        assert!(test_types::DataId::is_none(none));

        let zero = test_types::DataId::ZERO;
        assert_eq!(0, zero.inner);
        assert!(!test_types::DataId::is_none(zero));

        // Round-trip through array.
        let arr = test_types::DataId::to_array(id);
        let id2 = test_types::DataId::from_array(arr);
        assert_eq!(id, id2);

        // Round-trip via SlabItem trait.
        let mut slab = [0u32; 2];
        slab_write(&mut slab, 0, &id);
        let id3: test_types::DataId = slab_read(&slab, 0);
        assert_eq!(id, id3);
    }

    #[test]
    fn struct_array_type() {
        assert_eq!(2, test_types::DataArray::SLAB_SIZE);

        let none = test_types::DataArray::NONE;
        assert!(test_types::DataId::is_none(none.id));
        assert_eq!(0, none.len);

        // at() with valid index.
        let arr = test_types::DataArray {
            id: test_types::DataId::new(10),
            len: 3,
        };
        let id0 = test_types::DataArray::at(arr, 0);
        assert_eq!(10, id0.inner);
        let id1 = test_types::DataArray::at(arr, 1);
        assert_eq!(10 + test_types::Data::SLAB_SIZE as u32, id1.inner);
        let id2 = test_types::DataArray::at(arr, 2);
        assert_eq!(10 + 2 * test_types::Data::SLAB_SIZE as u32, id2.inner);

        // at() out of bounds returns NONE.
        let id3 = test_types::DataArray::at(arr, 3);
        assert!(test_types::DataId::is_none(id3));

        // Round-trip through array.
        let arr_data = test_types::DataArray::to_array(arr);
        let arr2 = test_types::DataArray::from_array(arr_data);
        assert_eq!(arr, arr2);

        // Round-trip via SlabItem trait.
        let mut slab = [0u32; 4];
        slab_write(&mut slab, 0, &arr);
        let arr3: test_types::DataArray = slab_read(&slab, 0);
        assert_eq!(arr, arr3);
    }

    // -----------------------------------------------------------------------
    // #[slab_module] enum integration tests
    // -----------------------------------------------------------------------

    #[slab_module]
    mod test_enum_types {
        #[slab_item]
        #[repr(u32)]
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
        pub enum DataChangeTy {
            #[default]
            I = 0,
            Float = 1,
            Ints = 2,
        }
    }

    #[test]
    fn enum_slab_size() {
        assert_eq!(1, test_enum_types::DataChangeTy::SLAB_SIZE);
        assert_eq!(1, <test_enum_types::DataChangeTy as SlabItem>::SLAB_SIZE);
    }

    #[test]
    fn enum_round_trip_via_inherent() {
        use test_enum_types::DataChangeTy;

        let variants = [DataChangeTy::I, DataChangeTy::Float, DataChangeTy::Ints];
        for v in variants {
            let arr = DataChangeTy::to_array(v);
            let v2 = DataChangeTy::from_array(arr);
            assert_eq!(v, v2);
        }
    }

    #[test]
    fn enum_round_trip_via_slab() {
        use test_enum_types::DataChangeTy;

        let mut slab = [0u32; 4];
        slab_write(&mut slab, 0, &DataChangeTy::Float);
        slab_write(&mut slab, 1, &DataChangeTy::Ints);
        slab_write(&mut slab, 2, &DataChangeTy::I);

        assert_eq!(DataChangeTy::Float, slab_read::<DataChangeTy>(&slab, 0));
        assert_eq!(DataChangeTy::Ints, slab_read::<DataChangeTy>(&slab, 1));
        assert_eq!(DataChangeTy::I, slab_read::<DataChangeTy>(&slab, 2));
    }

    #[test]
    fn enum_discriminant_values() {
        use test_enum_types::DataChangeTy;

        assert_eq!([0], DataChangeTy::to_array(DataChangeTy::I));
        assert_eq!([1], DataChangeTy::to_array(DataChangeTy::Float));
        assert_eq!([2], DataChangeTy::to_array(DataChangeTy::Ints));
    }

    #[test]
    fn enum_unknown_discriminant_defaults_to_first() {
        use test_enum_types::DataChangeTy;

        // Unknown discriminant value should map to the first variant.
        let unknown = DataChangeTy::from_array([999]);
        assert_eq!(DataChangeTy::I, unknown);
    }

    #[test]
    fn enum_id_type() {
        assert_eq!(1, test_enum_types::DataChangeTyId::SLAB_SIZE);

        let id = test_enum_types::DataChangeTyId::new(5);
        assert_eq!(5, id.inner);
        assert!(!test_enum_types::DataChangeTyId::is_none(id));

        let none = test_enum_types::DataChangeTyId::NONE;
        assert!(test_enum_types::DataChangeTyId::is_none(none));

        // Round-trip through array.
        let arr = test_enum_types::DataChangeTyId::to_array(id);
        let id2 = test_enum_types::DataChangeTyId::from_array(arr);
        assert_eq!(id, id2);

        // Round-trip via SlabItem trait.
        let mut slab = [0u32; 2];
        slab_write(&mut slab, 0, &id);
        let id3: test_enum_types::DataChangeTyId = slab_read(&slab, 0);
        assert_eq!(id, id3);
    }

    #[test]
    fn enum_array_type() {
        assert_eq!(2, test_enum_types::DataChangeTyArray::SLAB_SIZE);

        let none = test_enum_types::DataChangeTyArray::NONE;
        assert!(test_enum_types::DataChangeTyId::is_none(none.id));
        assert_eq!(0, none.len);

        // at() with valid index (SLAB_SIZE of enum is 1, so stride is 1).
        let arr = test_enum_types::DataChangeTyArray {
            id: test_enum_types::DataChangeTyId::new(10),
            len: 3,
        };
        let id0 = test_enum_types::DataChangeTyArray::at(arr, 0);
        assert_eq!(10, id0.inner);
        let id1 = test_enum_types::DataChangeTyArray::at(arr, 1);
        assert_eq!(11, id1.inner); // stride = 1
        let id2 = test_enum_types::DataChangeTyArray::at(arr, 2);
        assert_eq!(12, id2.inner);

        // at() out of bounds returns NONE.
        let id3 = test_enum_types::DataChangeTyArray::at(arr, 3);
        assert!(test_enum_types::DataChangeTyId::is_none(id3));

        // Round-trip via SlabItem trait.
        let mut slab = [0u32; 4];
        slab_write(&mut slab, 0, &arr);
        let arr2: test_enum_types::DataChangeTyArray = slab_read(&slab, 0);
        assert_eq!(arr, arr2);
    }

    // -----------------------------------------------------------------------
    // #[slab_module] nested struct integration tests
    // -----------------------------------------------------------------------

    #[slab_module]
    mod test_nested_types {
        #[slab_item]
        #[repr(u32)]
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
        pub enum DataChangeTy {
            #[default]
            I = 0,
            Float = 1,
            Ints = 2,
        }

        #[slab_item]
        #[derive(Clone, Copy, Debug, Default, PartialEq)]
        pub struct DataChange {
            pub ty: DataChangeTy,
            pub index: u32,
            pub value: f32,
        }

        #[slab_item]
        #[derive(Clone, Copy, Debug, Default, PartialEq)]
        pub struct ArrayChange {
            pub i: u32,
            pub change: DataChange,
        }
    }

    #[test]
    fn nested_slab_sizes() {
        // DataChangeTy is a repr(u32) enum => 1 slot.
        assert_eq!(1, test_nested_types::DataChangeTy::SLAB_SIZE);
        // DataChange has: DataChangeTy(1) + u32(1) + f32(1) = 3 slots.
        assert_eq!(3, test_nested_types::DataChange::SLAB_SIZE);
        // ArrayChange has: u32(1) + DataChange(3) = 4 slots.
        assert_eq!(4, test_nested_types::ArrayChange::SLAB_SIZE);
    }

    #[test]
    fn nested_datachange_round_trip() {
        use test_nested_types::*;

        let dc = DataChange {
            ty: DataChangeTy::Float,
            index: 7,
            value: 2.718,
        };
        let arr = DataChange::to_array(dc);
        let dc2 = DataChange::from_array(arr);
        assert_eq!(dc, dc2);

        // Via slab.
        let mut slab = [0u32; 8];
        slab_write(&mut slab, 0, &dc);
        let dc3: DataChange = slab_read(&slab, 0);
        assert_eq!(dc, dc3);
    }

    #[test]
    fn nested_arraychange_round_trip() {
        use test_nested_types::*;

        let ac = ArrayChange {
            i: 42,
            change: DataChange {
                ty: DataChangeTy::Ints,
                index: 3,
                value: -1.0,
            },
        };
        let arr = ArrayChange::to_array(ac);
        let ac2 = ArrayChange::from_array(arr);
        assert_eq!(ac, ac2);

        // Verify the raw u32 layout is correct.
        assert_eq!(42, arr[0]); // i
        assert_eq!(2, arr[1]); // change.ty = Ints = 2
        assert_eq!(3, arr[2]); // change.index
        assert_eq!((-1.0f32).to_bits(), arr[3]); // change.value

        // Via slab.
        let mut slab = [0u32; 8];
        slab_write(&mut slab, 0, &ac);
        let ac3: ArrayChange = slab_read(&slab, 0);
        assert_eq!(ac, ac3);

        // Write at non-zero offset.
        slab_write(&mut slab, 4, &ac);
        let ac4: ArrayChange = slab_read(&slab, 4);
        assert_eq!(ac, ac4);
    }

    #[test]
    fn nested_trait_impls_work() {
        use test_nested_types::*;

        // Verify all types implement SlabItem.
        assert_eq!(1, <DataChangeTy as SlabItem>::SLAB_SIZE);
        assert_eq!(3, <DataChange as SlabItem>::SLAB_SIZE);
        assert_eq!(4, <ArrayChange as SlabItem>::SLAB_SIZE);

        // ID and Array types for nested structs.
        assert_eq!(1, DataChangeId::SLAB_SIZE);
        assert_eq!(2, DataChangeArray::SLAB_SIZE);
        assert_eq!(1, ArrayChangeId::SLAB_SIZE);
        assert_eq!(2, ArrayChangeArray::SLAB_SIZE);

        // Array at() uses correct stride.
        let arr = ArrayChangeArray {
            id: ArrayChangeId::new(0),
            len: 3,
        };
        let id0 = ArrayChangeArray::at(arr, 0);
        assert_eq!(0, id0.inner);
        let id1 = ArrayChangeArray::at(arr, 1);
        assert_eq!(4, id1.inner); // stride = ArrayChange::SLAB_SIZE = 4
        let id2 = ArrayChangeArray::at(arr, 2);
        assert_eq!(8, id2.inner);
    }

    // -----------------------------------------------------------------------
    // #[slab_module(wgsl(...))] WGSL integration test
    // -----------------------------------------------------------------------

    /// Test that `#[slab_module(wgsl(...))]` correctly generates companion
    /// types and then emits `#[wgsl_rs::wgsl(...)]` on the output module.
    /// wgsl-rs transpiles the generated structs + inherent impls to WGSL
    /// while passing through the `impl crabslab::SlabItem` trait impls.
    #[slab_module(wgsl())]
    mod wgsl_test_types {
        #[slab_item]
        #[derive(Clone, Copy, Debug, Default, PartialEq)]
        pub struct SimpleData {
            pub x: u32,
            pub y: f32,
        }
    }

    #[test]
    fn wgsl_module_is_generated() {
        // The #[wgsl] attribute should have generated a WGSL_MODULE constant.
        let source = wgsl_test_types::WGSL_MODULE.wgsl_source();
        assert!(!source.is_empty(), "WGSL source should not be empty");
    }

    #[test]
    fn wgsl_module_contains_struct() {
        let source = wgsl_test_types::WGSL_MODULE.wgsl_source();
        let source_str = source.join("\n");
        // The struct should appear in the WGSL output.
        assert!(
            source_str.contains("struct SimpleData"),
            "WGSL source should contain 'struct SimpleData', got:\n{source_str}"
        );
    }

    #[test]
    fn wgsl_module_types_still_work_on_cpu() {
        // Verify the types are still usable on the CPU side.
        let d = wgsl_test_types::SimpleData { x: 42, y: 3.14 };
        let arr = wgsl_test_types::SimpleData::to_array(d);
        let d2 = wgsl_test_types::SimpleData::from_array(arr);
        assert_eq!(d, d2);

        // SlabItem trait still works.
        let mut slab = [0u32; 4];
        slab_write(&mut slab, 0, &d);
        let d3: wgsl_test_types::SimpleData = slab_read(&slab, 0);
        assert_eq!(d, d3);
    }

    #[test]
    fn slab_read_write_macros_cpu_fallback() {
        use test_types::*;

        // slab_read!/slab_write! macro_rules fallback (outside #[slab_module]).
        let d = Data {
            i: 7,
            float_val: 2.5,
            ints_0: 10,
            ints_1: 20,
        };
        let mut slab = [0u32; 8];
        slab_write!(Data, slab, 0, d);
        let d2 = slab_read!(Data, slab, 0);
        assert_eq!(d, d2);

        // At non-zero offset.
        slab_write!(Data, slab, 4, d);
        let d3 = slab_read!(Data, slab, 4);
        assert_eq!(d, d3);
    }
}
