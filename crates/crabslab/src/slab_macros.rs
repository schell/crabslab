//! CPU-side `slab_read!`/`slab_write!` statement macros — the CPU half
//! of the two-worlds pair owned by [`crate::SlabItemExt`] (GPU lowering).
//!
//! Inside a `#[wgsl(extensions = [crabslab::SlabItemExt])]` module the
//! transpiler captures these invocations as `Stmt::Macro` and the
//! extension lowers them to `array_container`/`SlabCopy`/`from_array` /
//! `to_array` sequences. On the CPU these `macro_rules!` expand to the
//! same shape, so both worlds agree: a value is read from a slab by
//! copying its slots into a zeroed container and calling `from_array`,
//! and written by calling `to_array` and copying the result back out.
//!
//! NOTE: this is the minimal surface required for `#[wgsl]` modules to
//! compile on the CPU. The full CPU API (offset- and `Id`-aware forms)
//! is a separate deliverable.
//!
//! # Example
//!
//! ```
//! use crabslab::{slab_read, slab_write, SlabItem};
//!
//! #[derive(Clone, Copy, Debug, Default, PartialEq, SlabItem)]
//! struct Foo {
//!     count: u32,
//!     is_on: bool,
//! }
//!
//! let foo = Foo { count: 42, is_on: true };
//!
//! // One slab slot per field.
//! let mut slab = [0u32; 2];
//! slab_write!(Foo, slab, 0, foo);
//! assert_eq!([42, 1], slab);
//!
//! // Declare the destination yourself; `slab_read!` reads into it.
//! // Use `let` for a plain read, `let mut` when you will mutate after.
//! let d: Foo;
//! slab_read!(Foo, slab, 0, d);
//! assert_eq!(foo, d);
//! ```

/// Read a `SlabItem` from `$slab` at `$offset`, assigning it into the
/// caller-declared `$dest`.
///
/// The macro does not define `$dest` — declare it first. A plain
/// `let d: Foo;` (no `mut`) works for pure reads; use `let mut` when
/// you will mutate the value after reading it back.
///
/// GPU form (lowered by `SlabItemExt`):
/// `slab_read!(Foo, get!(SLAB), offset, d)` →
/// `array_container` + `SlabCopy` + a `from_array` assignment into `d`.
///
/// The CPU form accepts any indexable slab expression: a `&[u32]`, a
/// `Vec<u32>`, a wgsl-rs storage guard (`get!(SLAB)`), etc.
#[macro_export]
macro_rules! slab_read {
    ($ty:ty, $slab:expr, $offset:expr, $dest:ident) => {
        let mut slab_read_buf = <$ty as $crate::SlabItem>::array_container();
        {
            let slab_offset = $offset as usize;
            for slab_i in 0..<$ty as $crate::SlabItem>::SLAB_SIZE {
                slab_read_buf[slab_i] = $slab[slab_offset + slab_i];
            }
        }
        $dest = <$ty as $crate::SlabItem>::from_array(slab_read_buf);
    };
}

/// Write the `SlabItem` value `$src` into `$slab` at `$offset`.
///
/// GPU form (lowered by `SlabItemExt`):
/// `slab_write!(Foo, get_mut!(SLAB), offset, d)` →
/// `to_array` + `SlabCopy`.
#[macro_export]
macro_rules! slab_write {
    ($ty:ty, $slab:expr, $offset:expr, $src:expr) => {
        let slab_write_buf = <$ty as $crate::SlabItem>::to_array($src);
        {
            let slab_offset = $offset as usize;
            for slab_i in 0..<$ty as $crate::SlabItem>::SLAB_SIZE {
                $slab[slab_offset + slab_i] = slab_write_buf[slab_i];
            }
        }
    };
}
