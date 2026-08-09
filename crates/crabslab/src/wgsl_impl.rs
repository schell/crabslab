use crate::SlabItemExt;
use wgsl_rs::wgsl;

/// CPU-side expansion of `slab_read!`.
///
/// On the CPU, reads a `SlabItem` from an indexable source using direct
/// indexing. Works with `&[u32]`, `RuntimeArray<u32>` guards, etc.
/// On the GPU (inside `#[wgsl]`), the parser passes this through as
/// `Stmt::Macro`, and the `SlabItemExt` extension lowers it.
#[macro_export]
macro_rules! slab_read {
    ($ty:ty, $slab:expr, $off:expr, $dest:ident) => {
        let off = $off as usize;
        let size = <$ty as $crate::SlabItem>::SLAB_SIZE;
        let mut __slab_read_buf = [0u32; 256];
        let mut __i = 0usize;
        while __i < size {
            __slab_read_buf[__i] = $slab[off + __i];
            __i += 1;
        }
        let mut $dest: $ty = <$ty as $crate::SlabItem>::read_slab(0, &__slab_read_buf[..size]);
    };
}

/// CPU-side expansion of `slab_write!`.
#[macro_export]
macro_rules! slab_write {
    ($ty:ty, $slab:expr, $off:expr, $src:expr) => {
        let off = $off as usize;
        let size = <$ty as $crate::SlabItem>::SLAB_SIZE;
        let mut __slab_write_buf = [0u32; 256];
        $crate::SlabItem::write_slab(&$src, 0, &mut __slab_write_buf[..size]);
        let mut __i = 0usize;
        while __i < size {
            $slab[off + __i] = __slab_write_buf[__i];
            __i += 1;
        }
    };
}

/// Marker attribute for slab item structs.
///
/// On the CPU, this is a no-op (the `#[derive(SlabItem)]` derive handles
/// the CPU-side impl). On the GPU (inside `#[wgsl]`), this attribute is
/// preserved in the IR and the `SlabItemExt` extension recognizes it to
/// generate `SLAB_SIZE`, `from_array`, `to_array`, and `array_container`.
pub use crabslab_derive::slab_item;

#[wgsl(crate_path = wgsl_rs, extensions = [crate::SlabItemExt])]
pub mod slab {
    use wgsl_rs::std::*;

    /// A simple slab item struct.
    #[crate::slab_item]
    #[derive(Clone, Copy, Default)]
    pub struct Data {
        pub one: f32,
        pub two: u32,
        pub flag: bool,
    }

    storage!(group(0), binding(0), SLAB: RuntimeArray<u32>);

    #[compute]
    #[workgroup_size(8)]
    pub fn main(#[builtin(local_invocation_index)] i: u32) {
        slab_read!(Data, get!(SLAB), i, d);
        d.one += 1.0;
        slab_write!(Data, get_mut!(SLAB), i, d);
    }
}

// CPU-side SlabItem impl for Data (the #[wgsl] macro strips derives,
// so we implement it manually here).
impl crate::SlabItem for slab::Data {
    const SLAB_SIZE: usize = 3;

    fn read_slab(mut index: usize, slab: &(impl crate::Slab + ?Sized)) -> Self {
        let one = f32::read_slab(index, slab);
        index += f32::SLAB_SIZE;
        let two = u32::read_slab(index, slab);
        index += u32::SLAB_SIZE;
        let flag = bool::read_slab(index, slab);
        Self { one, two, flag }
    }

    fn write_slab(&self, mut index: usize, slab: &mut (impl crate::Slab + ?Sized)) -> usize {
        index = self.one.write_slab(index, slab);
        index = self.two.write_slab(index, slab);
        index = self.flag.write_slab(index, slab);
        index
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn generates_wgsl() {
        let source = slab::WGSL_SOURCE.wgsl_source().unwrap();
        eprintln!("{source}");
        assert!(
            source.contains("const Data__1SLAB_SIZE: u32 = 3u;"),
            "expected Data__1SLAB_SIZE = 3, got:\n{source}"
        );
        assert!(
            source.contains("fn Data__1from_array("),
            "expected Data__1from_array, got:\n{source}"
        );
        assert!(
            source.contains("fn Data__1to_array("),
            "expected Data__1to_array, got:\n{source}"
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
