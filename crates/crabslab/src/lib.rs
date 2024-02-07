#![cfg_attr(target_arch = "spirv", no_std)]
//! Creating and crafting a tasty slab of memory.
#![doc = include_str!("../README.md")]

mod array;
pub use array::*;

mod id;
pub use id::*;

mod slab;
pub use slab::*;

#[cfg(feature = "wgpu")]
mod wgpu_slab;
#[cfg(feature = "wgpu")]
pub use wgpu_slab::*;

pub mod impl_slab_item;

pub use crabslab_derive::SlabItem;

#[cfg(not(target_arch = "spirv"))]
/// Proxy for `u32::saturating_sub`.
///
/// Used by the derive macro for `SlabItem`.
pub fn __saturating_sub(a: usize, b: usize) -> usize {
    a.saturating_sub(b)
}

#[cfg(target_arch = "spirv")]
pub fn __saturating_sub(a: usize, b: usize) -> usize {
    if a < b {
        0
    } else {
        a - b
    }
}
