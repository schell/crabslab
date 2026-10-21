//! The unified CPU/GPU `SlabItem` trait (Option B).
//!
//! This module promotes the trait and its WGSL-expressible primitive impls
//! out of the `wgsl_impl.rs` experiment into the crate's real surface. The
//! module is a `#[wgsl]` module so that it transpiles: the trait definition
//! is Rust-only (WGSL has no traits), while the primitive impls render as
//! mangled functions (`u32__1to_array`, ...) that any `#[wgsl]` module can
//! import via `use crabslab::slab_item::*;`.
//!
//! The dual-world `slab_read!`/`slab_write!` macros are defined here too:
//! on the CPU they expand to `array_container`/copy-loop/`from_array`
//! sequences, and inside `#[wgsl]` modules the transpiler passes them
//! through as `Stmt::Macro` for `SlabItemExt` to lower (see
//! [`crate::SlabItemExt`]).
//!
//! Every slab item is WGSL-expressible; slab I/O (`read_slab`/`write_slab`)
//! lives on the `CpuSlabItem` extension trait, re-exported at the crate root.
#![allow(clippy::approx_constant)]

/// Read a `SlabItem` from `$slab` at `$offset`, assigning it into the
/// caller-declared `$dest`.
///
/// The typed, whole-value convenience API over slab storage — distinct
/// from wgsl-rs's builtin `slab_copy!` (the raw 5-arg array/buffer
/// copy): this macro copies `SLAB_SIZE` slots and deserializes them in
/// one step.
///
/// The macro does not define `$dest` — declare it first. A plain
/// `let d: Foo;` (no `mut`) works for pure reads; use `let mut` when
/// you will mutate the value after reading it back.
///
/// The two worlds:
///
/// - **CPU**: expands to a `SlabItem::array_container` temp, an
///   element-wise copy loop, and a `SlabItem::from_array` assignment
///   into `$dest`. Any indexable slab expression works — a `[u32; N]`,
///   a `Vec<u32>`, a slice, a wgsl-rs storage guard (`get!(SLAB)`), etc.
/// - **GPU** (inside a `#[wgsl(extensions = [crabslab::SlabItemExt])]`
///   module): the transpiler captures the invocation as a `Stmt::Macro`
///   and `SlabItemExt` lowers it to the same shape —
///   `array_container` + copy loop + a `from_array` assignment.
///
/// # Example
///
/// ```
/// use crabslab::{slab_read, slab_write, SlabItem};
///
/// #[derive(Clone, Copy, Debug, Default, PartialEq, SlabItem)]
/// struct Foo {
///     count: u32,
///     is_on: bool,
/// }
///
/// let foo = Foo { count: 42, is_on: true };
///
/// // One slab slot per field.
/// let mut slab = [0u32; 2];
/// slab_write!(Foo, slab, 0, foo);
/// assert_eq!([42, 1], slab);
///
/// let d: Foo;
/// slab_read!(Foo, slab, 0, d);
/// assert_eq!(foo, d);
/// ```
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
/// The write half of the dual-world pair documented on
/// [`slab_read!`](crate::slab_read): CPU expands to a
/// `SlabItem::to_array` temp plus an element-wise copy loop; GPU
/// (`#[wgsl]` modules) is lowered by `SlabItemExt` to the same shape.
///
/// # Example
///
/// ```
/// use crabslab::{slab_write, SlabItem};
///
/// #[derive(Clone, Copy, Debug, Default, PartialEq, SlabItem)]
/// struct Id {
///     gen: u32,
///     index: u32,
/// }
///
/// let mut slab = [0u32; 5];
/// // Write at a nonzero offset; untouched slots stay zero.
/// slab_write!(Id, slab, 3, Id { gen: 7, index: 9 });
/// assert_eq!([0, 0, 0, 7, 9], slab);
/// ```
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
