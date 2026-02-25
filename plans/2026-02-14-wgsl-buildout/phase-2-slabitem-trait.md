# Phase 2: New `SlabItem` Trait (CPU-only, in `crabslab`)

**Status:** Pending
**Estimated effort:** 1 week
**Prerequisites:** Phase 1

## Overview

Define a new CPU-only `SlabItem` trait in `crates/crabslab`, implement it for
primitives, and remove the old trait-based abstractions (`Slab`, `GrowableSlab`,
generic `Id<T>`, `Array<T>`).

---

## 2.1 Define the new trait

In `crates/crabslab/src/lib.rs` (outside any `#[wgsl]` module):

```rust
/// CPU-only trait for types that can be stored in a u32 slab.
///
/// Auto-implemented by the `#[slab_module]` macro for `#[slab_item]` types.
pub trait SlabItem: Sized {
    /// The concrete ID type for this slab item (e.g. `DataId`).
    type Id: Copy + Default;

    /// The number of `u32` slots this type occupies in a slab.
    const SLAB_SIZE: usize;

    /// Read this type from a `u32` slab at the given index.
    fn slab_read(slab: &[u32], index: usize) -> Self;

    /// Write this type into a `u32` slab at the given index.
    fn slab_write(data: &Self, slab: &mut [u32], index: usize);

    /// Serialize this value into a new `Vec<u32>`.
    fn slab_data(&self) -> Vec<u32> {
        let mut data = vec![0u32; Self::SLAB_SIZE];
        Self::slab_write(self, &mut data, 0);
        data
    }
}
```

---

## 2.2 Implement for primitives

Manual `SlabItem` impls for `u32`, `i32`, `f32`, `bool`, and glam vector/matrix
types. These live in `crabslab` and don't need `#[slab_item]`. Each primitive
gets a simple ID type (or uses a generic `SlabIndex { inner: u32 }`).

---

## 2.3 Remove old types and traits

- Remove old `Slab` trait
- Remove old `SlabItem` trait
- Remove old `GrowableSlab` trait
- Remove generic `Id<T>` and `Array<T>`
- Remove `CpuSlab<B>` wrapper (or repurpose if still useful)
- Remove all `spirv-std` dependencies
- Remove all `#[cfg(target_arch = "spirv")]` / `#[cfg(not(target_arch =
  "spirv"))]` blocks
- Remove `#![cfg_attr(target_arch = "spirv", no_std)]` from all crates
- Remove `spirv-std` from workspace dependencies
- Remove `[patch.crates-io]` spirv-std entry
- Remove helper functions: `slice_index`, `array_index`, `array_index_mut`,
  `__saturating_sub`
