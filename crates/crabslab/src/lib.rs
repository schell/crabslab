//! Creating and crafting a tasty slab of memory.
#![doc = include_str!("../README.md")]
#![warn(missing_docs)]

// Allow the `#[derive(SlabItem)]` macro to reference `crabslab::SlabItem`
// from within the crate itself.
pub extern crate self as crabslab;

mod array;
mod id;
mod slab;

pub mod bits;
/// CPU-only [`SlabItem`] implementations: `u8` through
/// `u128`, `f64`, tuples, arrays, and `glam` types behind the `glam`
/// feature.
pub mod impl_slab_item;
pub mod offset;
#[macro_use]
pub mod slab_item;

pub use array::*;
pub use id::*;
pub use slab::*;
pub use slab_item::slab::SlabItem;

pub use crabslab_derive::SlabItem;

/// The living experiment fixture for GPU-side slab serialization.
/// Test scaffolding; not intended API, so documentation is not enforced
/// here.
#[allow(missing_docs)]
pub mod wgsl_impl;

pub mod slab_item_ext;

pub use slab_item_ext::SlabItemExt;

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
