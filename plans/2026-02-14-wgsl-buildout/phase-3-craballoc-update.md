# Phase 3: Update `craballoc` for New `SlabItem`

**Status:** Pending
**Estimated effort:** 1 week
**Prerequisites:** Phase 2

## Overview

Update the `craballoc` arena allocator crate to use the new `SlabItem` trait,
update `Range` conversions, and remove SPIR-V shader loading.

---

## 3.1 Update `Arena` API

`Arena::new_value<T: SlabItem>` remains generic but uses the new trait:

```rust
pub fn new_value<T: SlabItem>(&self, value: T) -> Value<T> {
    let data = T::slab_data(&value);
    // ... bump allocate, create CpuUpdateSource, etc.
}
```

`Value<T>` methods update similarly:
- `get()` uses `T::slab_read(&data, 0)`
- `set()` uses `T::slab_write(&value, &mut data, 0)`
- `id()` returns `T::Id` constructed from the allocation's first index

---

## 3.2 Update `Range` conversions

Replace `Range::from(Id<T>)` and `Range::from(Array<T>)` with methods that take
a `u32` index and `T::SLAB_SIZE`.

---

## 3.3 Remove SPIR-V shader loading

- Remove `wgpu::include_spirv!` usage from `craballoc/src/test/wgpu.rs`
- Remove `crates/craballoc/src/test/shaders/apply_data_changes.spv`
- Remove `crates/craballoc/src/test/shaders/manifest.json`
- Remove `wgpu`'s `"spirv"` feature from workspace dependencies
