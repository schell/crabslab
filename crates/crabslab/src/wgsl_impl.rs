use wgsl_rs::wgsl;

#[wgsl(extensions = [crate::SlabItemExt])]
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
    /// The `#[derive(SlabItem)]` macro generates this impl on the CPU side.
    /// The `SlabItemExt` `WgslExtension` generates the same method names
    /// (`SLAB_SIZE`, `from_array`, `to_array`, `array_container`) as inherent
    /// methods on the GPU side.
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

    // NOTE: `[u32; Self::SLAB_SIZE]` in a trait method signature does not
    // compile on stable Rust - `Self::SLAB_SIZE` is a generic projection and
    // array lengths in type position must be evaluable before
    // monomorphization (the `generic_const_exprs` feature would unlock it).
    // This is the same blocker documented above on `type Array`.
    //
    // pub trait SlabThing: SlabItem {
    //     fn to_slab(data: Self) -> [u32; Self::SLAB_SIZE];
    // }

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
            arr[0] != 0
        }

        fn array_container() -> Self::Array {
            [0]
        }
    }

    #[derive(Wgsl)]
    pub struct Bar {
        pub value: f32,
        pub flag: bool,
    }

    impl SlabItem for Bar {
        const SLAB_SIZE: usize = f32::SLAB_SIZE + bool::SLAB_SIZE;

        type Array = [u32; f32::SLAB_SIZE + bool::SLAB_SIZE];

        fn array_container() -> Self::Array {
            [0u32; f32::SLAB_SIZE + bool::SLAB_SIZE]
        }

        fn to_array(data: Self) -> [u32; f32::SLAB_SIZE + bool::SLAB_SIZE] {
            let mut dest = Self::array_container();
            let mut i: usize = 0;

            let value_slab = f32::to_array(data.value);
            slab_copy!(value_slab, 0, dest, i, f32::SLAB_SIZE);
            i += f32::SLAB_SIZE;

            let flag_slab = bool::to_array(data.flag);
            slab_copy!(flag_slab, 0, dest, i, bool::SLAB_SIZE);
            //i += bool::SLAB_SIZE;

            dest
        }

        fn from_array(slab: Self::Array) -> Self {
            let mut i: usize = 0;

            let mut value_array = f32::array_container();
            slab_copy!(slab, i, value_array, 0, f32::SLAB_SIZE);
            i += f32::SLAB_SIZE;
            let value = f32::from_array(value_array);

            let mut flag_array = bool::array_container();
            slab_copy!(slab, i, flag_array, 0, bool::SLAB_SIZE);
            //i += bool::SLAB_SIZE;
            let flag = bool::from_array(flag_array);

            Bar { value, flag }
        }
    }

    #[derive(Wgsl)]
    pub struct Foo {
        pub count: u32,
        pub inner: Bar,
    }

    impl SlabItem for Foo {
        const SLAB_SIZE: usize = u32::SLAB_SIZE + Bar::SLAB_SIZE;

        type Array = [u32; u32::SLAB_SIZE + Bar::SLAB_SIZE];

        fn array_container() -> Self::Array {
            [0u32; u32::SLAB_SIZE + Bar::SLAB_SIZE]
        }

        fn to_array(data: Self) -> [u32; u32::SLAB_SIZE + Bar::SLAB_SIZE] {
            let mut dest = Self::array_container();
            let mut i: usize = 0;

            let count_slab = u32::to_array(data.count);
            slab_copy!(count_slab, 0, dest, i, u32::SLAB_SIZE);
            i += u32::SLAB_SIZE;

            let inner_slab = Bar::to_array(data.inner);
            slab_copy!(inner_slab, 0, dest, i, Bar::SLAB_SIZE);
            //i += Bar::SLAB_SIZE;

            dest
        }

        fn from_array(slab: Self::Array) -> Self {
            let mut i: usize = 0;

            let mut count_array = u32::array_container();
            slab_copy!(slab, i, count_array, 0, u32::SLAB_SIZE);
            i += u32::SLAB_SIZE;
            let count = u32::from_array(count_array);

            let mut inner_array = Bar::array_container();
            slab_copy!(slab, i, inner_array, 0, Bar::SLAB_SIZE);
            // i += Bar::SLAB_SIZE;
            let inner = Bar::from_array(inner_array);

            Foo { count, inner }
        }
    }

    storage!(group(0), binding(0), read_write, SLAB: RuntimeArray<u32>);

    #[compute]
    #[workgroup_size(8)]
    pub fn main(#[builtin(local_invocation_index)] _x: u32) {
        let mut foo_array = Foo::array_container();
        slab_copy!(get!(SLAB), 0, foo_array, 0, Foo::SLAB_SIZE);

        let mut d = Foo::from_array(foo_array);
        d.count += 1;
        d.inner.flag = true;

        let foo_slab = Foo::to_array(d);
        slab_copy!(foo_slab, 0, get_mut!(SLAB), 0, Foo::SLAB_SIZE);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod test {
    use super::*;

    //     #[test]
    //     fn primitives_round_trip() {
    //         let mut slab = [0u32; 8];
    //         slab_write(&mut slab, 0, &42u32);
    //         slab_write(&mut slab, 1, &-7i32);
    //         slab_write(&mut slab, 2, &3.14f32);
    //         slab_write(&mut slab, 3, &true);
    //         slab_write(&mut slab, 4, &false);

    //         assert_eq!(42u32, slab_read::<u32>(&slab, 0));
    //         assert_eq!(-7i32, slab_read::<i32>(&slab, 1));
    //         assert_eq!(3.14f32, slab_read::<f32>(&slab, 2));
    //         assert_eq!(true, slab_read::<bool>(&slab, 3));
    //         assert_eq!(false, slab_read::<bool>(&slab, 4));
    //     }

    #[test]
    fn generates_wgsl() {
        let source = slab::WGSL_SOURCE.wgsl_source().unwrap();
        eprintln!("{source}");

        // Primitive impls render with the mangled `Type__1member` names.
        assert!(
            source.contains("const u32__1SLAB_SIZE: u32 = 1u;"),
            "expected u32__1SLAB_SIZE, got:\n{source}"
        );
        assert!(
            source.contains("fn u32__1to_array("),
            "expected u32__1to_array, got:\n{source}"
        );
        assert!(
            source.contains("fn u32__1from_array("),
            "expected u32__1from_array, got:\n{source}"
        );

        // Struct impls render `SLAB_SIZE` as a const-sum expression, not a
        // literal.
        assert!(
            source.contains("const Bar__1SLAB_SIZE: u32 = (f32__1SLAB_SIZE + bool__1SLAB_SIZE);"),
            "expected Bar__1SLAB_SIZE const-sum, got:\n{source}"
        );
        assert!(
            source.contains("const Foo__1SLAB_SIZE: u32 = (u32__1SLAB_SIZE + Bar__1SLAB_SIZE);"),
            "expected Foo__1SLAB_SIZE const-sum, got:\n{source}"
        );
        assert!(
            source.contains("fn Bar__1to_array("),
            "expected Bar__1to_array, got:\n{source}"
        );
        assert!(
            source.contains("fn Bar__1from_array("),
            "expected Bar__1from_array, got:\n{source}"
        );
        assert!(
            source.contains("fn Foo__1to_array("),
            "expected Foo__1to_array, got:\n{source}"
        );
        assert!(
            source.contains("fn Foo__1from_array("),
            "expected Foo__1from_array, got:\n{source}"
        );

        // The offset-corrected lowering: `to_array` writes `dest[i + _i]`
        // from the field slab at `0u`, and `from_array` reads `slab[i + _i]`
        // into the field array at `0u` (the old `slab_read_array!` form put
        // the running offset on the wrong operand).
        assert!(
            source.contains("dest[i + _i] = value_slab[0u + _i];"),
            "expected offset-on-dest to_array copy, got:\n{source}"
        );
        assert!(
            source.contains("value_array[0u + _i] = slab[i + _i];"),
            "expected offset-on-src from_array copy, got:\n{source}"
        );

        // No unlowered macros survive to the rendered source.
        assert!(
            !source.contains("slab_copy!"),
            "slab_copy! should have been lowered, got:\n{source}"
        );
        assert!(
            !source.contains("slab_read!"),
            "slab_read! should have been lowered, got:\n{source}"
        );
        assert!(
            !source.contains("slab_write!"),
            "slab_write! should have been lowered, got:\n{source}"
        );
    }
}
