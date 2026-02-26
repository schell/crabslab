# Phase 2: Update `craballoc` for `crabslab2`

**Status:** Complete
**Estimated effort:** 1 week
**Prerequisites:** Phase 1

## Overview

Update the `craballoc` arena allocator crate in place to depend on `crabslab2`
instead of `crabslab`. Replace `Id<T>`, `Array<T>`, `Slab` trait usage with
`crabslab2`'s `SlabItem` trait and raw `u32` indices. `GrowableSlab` and
`CpuSlab` are not used in `craballoc` — only `Slab`, `SlabItem`, `Id<T>`, and
`Array<T>`.

## Steps

| Step | Description | Status |
|---|---|---|
| 1 | Workspace & dependency changes | Complete |
| 2 | Update `lib.rs` prelude | Complete |
| 3 | Update `arena.rs` — remove `Id<T>`/`Array<T>`, use `slab_read`/`slab_write` | Complete |
| 4 | Update `range.rs` — remove `From<Id<T>>` and `From<Array<T>>` | Complete |
| 5 | Disable wire-type-dependent and GPU shader tests | Complete |
| 6 | Verify — `cargo build`, `cargo test`, `cargo clippy` | Complete |

---

## 2.1 Dependency changes

**Workspace root `Cargo.toml`:**
- Add `"crates/craballoc"` to `workspace.members`

**`crates/craballoc/Cargo.toml`:**
- Replace `crabslab = { path = "../crabslab", version = "0.7.0" }` with
  `crabslab2 = { path = "../crabslab2" }`
- Add `wgsl-rs = { workspace = true }`
- Remove `craballoc-test-wire-types` from dev-dependencies (those types will
  be inlined as test modules in Phase 3)

---

## 2.2 Update `lib.rs` prelude

- Remove `pub extern crate crabslab;`
- Remove `pub use crabslab::{Array, Id};`
- Add `pub use crabslab2::{SlabItem, slab_read, slab_write};`

---

## 2.3 Update `arena.rs` — API changes

### `Value<T>` methods

| Old | New |
|---|---|
| `id() -> Id<T>` | `id() -> u32` — raw slab index |
| `array() -> Array<T>` | `array() -> (u32, u32)` — `(index, 1)` |
| `modify()` uses `data.read(Id::<T>::ZERO)` | `slab_read::<T>(&data, 0)` |
| `modify()` uses `data.write(Id::ZERO, &t)` | `slab_write(&mut data, 0, &t)` |
| `get()` uses `data.read_unchecked(Id::ZERO)` | `slab_read::<T>(&data, 0)` |

### `Value<[T]>` methods

| Old | New |
|---|---|
| `array() -> Array<T>` | `array() -> (u32, u32)` — `(first_index, len)` |
| `modify_range()` uses `data.read_vec(array)` | Loop: `slab_read` per item |
| `modify_range()` uses `data.write_array(array, &s)` | Loop: `slab_write` per item |
| `read_range()` uses `data.read_vec(array)` | Loop: `slab_read` per item |

### `Arena` test methods

| Old | New |
|---|---|
| `read_slab(array: Array<T>)` | `read_slab(index: u32, len: u32)` — loop with `slab_read` |

### Helper functions for slab slice operations

Add private helpers to replace `Slab` trait methods on `[u32]`:

```rust
/// Read a Vec<T> from a u32 slice, starting at index 0, reading `count` items.
fn slab_read_vec<T: SlabItem>(data: &[u32], count: usize) -> Vec<T> {
    (0..count)
        .map(|i| slab_read::<T>(data, i * T::SLAB_SIZE))
        .collect()
}

/// Write a slice of T into a u32 slice, starting at index 0.
fn slab_write_slice<T: SlabItem>(data: &mut [u32], items: &[T]) {
    for (i, item) in items.iter().enumerate() {
        slab_write(data, i * T::SLAB_SIZE, item);
    }
}
```

---

## 2.4 Update `range.rs`

- Remove `use crabslab::{Array, Id, SlabItem};`
- Remove `impl<T: SlabItem> From<Id<T>> for Range`
- Remove `impl<T: SlabItem> From<Array<T>> for Range`
- Call sites that used `Range::from(id)` or `Range::from(array)` construct
  `Range` directly from raw `u32` values + `T::SLAB_SIZE`

---

## 2.5 Disable wire-type-dependent tests

Tests that depend on `craballoc-test-wire-types` or the SPIR-V shader are
gated behind `#[cfg(any())]` (compile-time disabled) in a `phase3_tests`
submodule. They cannot use `#[ignore]` because the old types (`Data`,
`DataChange`, `Array<T>`, `Id<T>`) no longer compile. Phase 3 will inline
the wire types using `#[slab_module]` and rewrite the compute shader in WGSL.

**Disabled tests (Phase 3 will fix):**
- `arena_roundtrip_sanity` (Data portion — the u32 portion was kept as
  `arena_roundtrip_u32_sanity`)
- `array_subslice_sanity`
- `gpu_update_test_sanity_on_cpu`
- `gpu_update_test_sanity_on_gpu`
- `gpu_array_update_test_sanity_on_cpu`
- `gpu_array_update_test_sanity_on_gpu`
- `proptest_gpu_updates_checked_on_cpu`
- `proptest_gpu_updates_checked_on_gpu`
- `workgroup_dimensions_to_id_sanity`
- `invocations_sanity`
- `regression`

**Kept working (11 tests passing):**
- `mngr_updates_count_sanity`
- `range_sanity`
- `arena_roundtrip_u32_sanity` (new — extracted u32 portion)
- `slab_manager_sanity`
- `overwrite_sanity`
- `proptest_overlapping_updates` (u32-only, no wire types)
- `range_full` (arena.rs inline test)
- All `range/test.rs` tests (4 tests including proptest)

---

## 2.6 SPIR-V removal is Phase 3

The `.spv` file, `manifest.json`, and `wgpu::include_spirv!` usage in
`test/wgpu.rs` remain on disk but are not compiled (tests are `#[ignore]`d).
Phase 3 will delete these and replace with WGSL via `#[slab_module]` +
`#[wgsl]` + wgsl-rs `linkage-wgpu`.

---

## Key observations from codebase analysis

1. **`IsRuntime` trait** operates on raw `u32` slices and `std::ops::Range<usize>`.
   No crabslab types at the runtime interface level. No changes needed.
2. **`CpuUpdateSource::modify()` and `::read()` closures** receive
   `&mut [u32]` / `&[u32]`. The old code calls `Slab` trait methods on these.
   The new code uses `crabslab2::slab_read`/`slab_write` directly.
3. **`GrowableSlab` and `CpuSlab`** are NOT used in `craballoc`.
4. **`buffer.rs` and `buffer/manager.rs`** have no crabslab imports — they
   operate on raw `u32` buffers. No changes needed.
5. **`update.rs`** has no crabslab imports — it manages `Vec<u32>` data and
   `Range` types. No changes needed (our `Range` stays the same).
6. **`wgsl-rs` depends on `naga v28`** which has a compile error with
   `termcolor`. Moving `wgsl-rs` from `[dependencies]` to `[dev-dependencies]`
   in `crabslab2/Cargo.toml` avoids this — `wgsl-rs` is only used in
   `#[cfg(test)]` code in `crabslab2`.
7. **`glam` with `default-features = false`** fails to compile without a math
   backend. Removed from `craballoc` dev-dependencies since the tests that used
   it are now cfg-gated.
