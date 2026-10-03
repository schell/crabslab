//! Creating and crafting a tasty slab of memory.
#![doc = include_str!("../README.md")]

// Allow the `#[derive(SlabItem)]` macro to reference `crabslab::SlabItem`
// from within the crate itself.
pub extern crate self as crabslab;

mod array;
mod id;
mod slab;

pub mod bits;
pub mod impl_slab_item;
pub mod offset;

pub use array::*;
pub use id::*;
pub use slab::*;

pub use crabslab_derive::SlabItem;

pub mod wgsl_impl;

mod slab_item_ext;

pub use slab_item_ext::SlabItemExt;

// TODO: See if we need any of this at all.

/// Proxy for `u32::saturating_sub`.
///
/// Used by the derive macro for `SlabItem`.
pub const fn __saturating_sub(a: usize, b: usize) -> usize {
    a.saturating_sub(b)
}

#[inline]
pub fn slice_index<T>(slab: &[T], index: usize) -> &T {
    &slab[index]
}

#[inline]
pub fn array_index<const N: usize, T>(slab: &[T; N], index: usize) -> &T {
    &slab[index]
}

#[inline]
pub fn array_index_mut<const N: usize, T>(slab: &mut [T; N], index: usize) -> &mut T {
    &mut slab[index]
}
