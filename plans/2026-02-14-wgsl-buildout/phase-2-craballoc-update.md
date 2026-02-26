# Phase 2: Update `craballoc` for `crabslab2`

**Status:** Pending
**Estimated effort:** 1 week
**Prerequisites:** Phase 1

## Overview

Update the `craballoc` arena allocator crate in place to depend on `crabslab2`
instead of `crabslab`. Replace `Id<T>`, `Array<T>`, `Slab`, `GrowableSlab`, and
`CpuSlab` usage with `crabslab2`'s `SlabItem` trait and raw `u32` indices.

---

## 2.1 Dependency changes

- Replace `crabslab` dependency with `crabslab2`
- Remove `crabslab-derive` dependency if present
- Add `wgsl-rs` dependency (for wire types in tests)

---

## 2.2 Update `Arena` API

`Arena` allocation returns raw `u32` indices instead of `Id<T>`:

```rust
pub fn new_value<T: SlabItem>(&self, value: T) -> Value<T> {
    let arr = value.to_array();
    // bump allocate T::SLAB_SIZE slots, write arr to the slab
    // ...
}
```

`Value<T>` methods:
- `get()` uses `T::from_array(...)` to read from the CPU cache
- `set()` uses `T::to_array(...)` to write to the CPU cache
- `id()` returns `u32` (the slab index of this value)

---

## 2.3 Update `Range` conversions

Replace `Range::from(Id<T>)` and `Range::from(Array<T>)` with methods that take
a `u32` index and `T::SLAB_SIZE` directly.

---

## 2.4 Remove old imports

- Remove all `use crabslab::{Id, Array, Slab, GrowableSlab, CpuSlab, SlabItem}`
- Replace with `use crabslab2::{SlabItem, slab_read, slab_write}`
- Remove `Id<T>` from all type signatures, replace with `u32`
- Remove `Array<T>` usage, replace with `(u32, u32)` or a simple struct
  `SlabArray { index: u32, len: u32 }`

---

## 2.5 Remove SPIR-V shader loading

- Remove `wgpu::include_spirv!` usage
- Remove `.spv` file and `manifest.json` from `craballoc/src/test/shaders/`
- Remove `wgpu`'s `"spirv"` feature from workspace dependencies
