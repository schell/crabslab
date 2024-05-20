#![allow(unexpected_cfgs)]
#![cfg_attr(target_arch = "spirv", no_std)]
//! Creating and crafting a tasty slab of memory.
#![doc = include_str!("../README.md")]

mod array;
mod id;
mod slab;

pub mod impl_slab_item;

pub use array::*;
pub use id::*;
pub use slab::*;

pub use crabslab_derive::SlabItem;

#[cfg(not(target_arch = "spirv"))]
/// Proxy for `u32::saturating_sub`.
///
/// Used by the derive macro for `SlabItem`.
pub const fn __saturating_sub(a: usize, b: usize) -> usize {
    a.saturating_sub(b)
}

#[cfg(target_arch = "spirv")]
pub const fn __saturating_sub(a: usize, b: usize) -> usize {
    if a < b {
        0
    } else {
        a - b
    }
}

#[cfg(not(target_arch = "spirv"))]
#[inline]
pub fn slice_index<T>(slab: &[T], index: usize) -> &T {
    &slab[index]
}

#[cfg(not(target_arch = "spirv"))]
#[inline]
pub fn slice_index_mut<T>(slab: &mut [T], index: usize) -> &mut T {
    &mut slab[index]
}

#[cfg(target_arch = "spirv")]
#[inline]
pub fn slice_index<T>(slab: &[T], index: usize) -> &T {
    unsafe { spirv_std::arch::IndexUnchecked::index_unchecked(slab, index) }
}

#[cfg(target_arch = "spirv")]
#[inline]
pub fn slice_index_mut<T>(slab: &mut [T], index: usize) -> &mut T {
    unsafe { spirv_std::arch::IndexUnchecked::index_unchecked_mut(slab, index) }
}

#[cfg(not(target_arch = "spirv"))]
#[inline]
pub fn array_index<const N: usize, T>(slab: &[T; N], index: usize) -> &T {
    &slab[index]
}

#[cfg(not(target_arch = "spirv"))]
#[inline]
pub fn array_index_mut<const N: usize, T>(slab: &mut [T; N], index: usize) -> &mut T {
    &mut slab[index]
}

#[cfg(target_arch = "spirv")]
#[inline]
pub fn array_index<const N: usize, T>(slab: &[T; N], index: usize) -> &T {
    unsafe { spirv_std::arch::IndexUnchecked::index_unchecked(slab, index) }
}

#[cfg(target_arch = "spirv")]
#[inline]
pub fn array_index_mut<const N: usize, T>(slab: &mut [T; N], index: usize) -> &mut T {
    unsafe { spirv_std::arch::IndexUnchecked::index_unchecked_mut(slab, index) }
}
