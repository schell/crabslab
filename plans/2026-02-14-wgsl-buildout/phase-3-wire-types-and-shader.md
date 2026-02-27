# Phase 3: Wire Types and Compute Shader (Test Modules)

**Status:** Complete
**Estimated effort:** 1 week
**Prerequisites:** Phase 2

## Overview

Rewrite the wire types and compute shader as a single `#[wgsl] #[slab_module]`
module inside `craballoc/src/test.rs`, using wgsl-rs for WGSL transpilation and
CPU-side dispatch. The separate `craballoc-test-shaders` and
`craballoc-test-wire-types` crates are no longer needed.

---

## Implementation Notes

### Single module approach

Unlike the original plan which had separate `wire_types` and
`apply_data_changes` modules, the final implementation uses a **single**
`#[wgsl] #[slab_module] mod apply_data_changes` module containing both the wire
types and the compute shader. This avoids cross-module import issues inside
`#[wgsl]` modules.

### Wire types

All `#[slab_item]` types derive `Clone, Copy, Debug, Default` (and `PartialEq`
for `Data`). The `#[slab_module]` macro does NOT auto-derive these on the
annotated types — only on the generated companion types (`*Id`, `*Array`).

- `Data` — `{ i: u32, float_val: f32, ints_0: u32, ints_1: u32 }`
- `DataChangeTy` — `#[repr(u32)]` enum with `#[default]` on `I = 0`
- `DataChange` — `{ ty: DataChangeTy, data_0: u32, data_1: u32, data_2: u32 }`
  with `DataChange::apply(change, data) -> Data` method
- `ArrayChange` — `{ i: u32, change: DataChange }`
- `AnyChangeId` — `{ change_id: ArrayChangeId, data_array: DataArray }`
- `ApplyDataChangeInvocation` — `{ changes_ids: AnyChangeIdArray }`

The `InvocationCount` type from the original plan was dropped. Invocation
counters use separate `Atomic<u32>` storage bindings instead of being embedded
in the slab.

### Compute shader

The `main` entry point uses:
- 4 storage bindings: `DATA_SLAB` (read_write RuntimeArray), `CHANGES_SLAB`
  (read-only RuntimeArray), `INVOCATIONS_RAN` (read_write Atomic), and
  `INVOCATIONS_SKIPPED` (read_write Atomic)
- `slab_read_array!` / `slab_write_array!` macros with explicit `[0u32, ...]`
  array literals (wgsl-rs does not support `[0u32; N]` repeat expressions)
- Statement-form `match` with `#[wgsl_allow(non_literal_match_statement_patterns)]`

### CPU dispatch

The `BackendUpdate` impl for `CpuRuntime` uses:
- `Storage::set()` to populate storage statics before dispatch
- Direct function calls to `apply_data_changes::main(vec3u(i, j, k))` in a
  loop (not `dispatch_workgroups`, to keep the dispatch logic explicit)
- `atomic_load()` to read counters back after dispatch

### wgsl-rs discoveries

- `#[allow(clippy::...)]` attributes inside `#[wgsl]` modules break the parser.
  Place them on the module itself.
- `#[default]` on enum variants works correctly through `#[wgsl]`/`#[slab_module]`.
- Feature unification: when `craballoc` enables `wgsl-rs/linkage-wgpu`, the
  `#[wgsl]` macro emits `wgpu`-referencing code for all crates in the workspace.
  `crabslab2` needed `wgpu` added as a dev-dependency to compile.

---

## Tests

All 18 craballoc tests pass (7 previously gated tests re-enabled):
- `gpu_update_test_sanity_on_cpu`
- `gpu_array_update_test_sanity_on_cpu`
- `invocations_sanity`
- `regression`
- `workgroup_dimensions_to_id_sanity`
- `proptest_gpu_updates_checked_on_cpu`
- Plus all 11 previously passing tests

GPU (wgpu) backend tests are deferred as a follow-up — the `TestBackendWgpu`
struct needs updating to use `linkage-wgpu` generated pipeline code.
