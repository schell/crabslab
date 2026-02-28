# Phase 5: Testing and Verification

**Status:** Complete
**Estimated effort:** 1 week
**Prerequisites:** Phase 4

## Overview

Validate the migration end-to-end: WGSL validation, CPU-side tests, GPU-side
tests via wgpu, and manual inspection of generated WGSL output.

---

## 5.1 WGSL validation

All `#[wgsl]` modules with no imports validate at compile time via naga.
Modules with imports validate at test time via auto-generated
`__validate_wgsl()` tests.

**Status:** Complete. All generated WGSL validates at compile time (naga
validation enabled, `skip_validation` removed).

### Resolved issues

#### Attribute ordering issue

Proc-macro attributes in Rust are applied **top-to-bottom** (outermost first).
The original stacked-attribute pattern had `#[wgsl]` running before
`#[slab_module]`, so companion types were missing from the WGSL output.

**Fix:** `#[slab_module(wgsl(...))]` now handles this internally. The macro
generates companion types first, then emits `#[wgsl_rs::wgsl(...)]` on the
output module. No attribute stacking required.

#### wgsl-rs compatibility issues resolved

- `[0u32; N]` repeat expressions: Fixed in wgsl-rs commit `3e608ec`.
- `[u32; Type::SLAB_SIZE]` array type annotations: Supported.
- `Type::SLAB_SIZE` path expressions in arithmetic: Supported.
- `slab_read_array!`/`slab_write_array!` macros: Supported.
- Struct literals `Type { field: value }`: Supported.
- `#[derive(...)]` on structs: Silently stripped by `#[wgsl]`.
- `impl crabslab::SlabItem for X` trait impls: Silently skipped by `#[wgsl]`.

#### Codegen fixes for WGSL compliance

- **WGSL `switch` is a statement, not an expression.** Enum `to_array` and
  `from_array` use block-form match arms with explicit variable assignment
  (`var v: u32; match d { ... }` -> `var v: u32; switch(d) { case 0u { v = 0u; } ... }`).
- **WGSL does not support block expressions as struct field initializers.**
  Nested type `from_array` uses pre-computation locals (`sub_<field>`,
  `val_<field>`) before the struct literal.
- **WGSL identifiers cannot start with `__`.** Changed generated variable
  prefixes from `__sub_`/`__val_` to `sub_`/`val_`.
- **WGSL `let` must be initialized.** User code uses `let mut result: Data;`
  (-> `var result: Data;` in WGSL) instead of `let result: Data;`.
- **wgsl-rs auto-generated `default` case bug.** Added explicit `_ => {}` arm
  to enum `to_array` match to avoid `default:` (with colon) in output.

---

## 5.2 CPU-side tests

Run the existing property-based tests (proptest) that test
`ApplyDataChangeInvocation` on the CPU backend. These should work identically
since the Rust code inside `#[wgsl]` modules is real, executable Rust.

**Status:** Complete (18 tests passing since Phase 3).

---

## 5.3 GPU-side tests

Run the wgpu-based tests with the new WGSL shader. Compare results with the CPU
backend to verify correctness.

**Status:** Complete. Three GPU tests pass, verifying that the transpiled WGSL
compute shader produces identical results to CPU dispatch.

### Implementation

`TestBackendWgpu` struct holds the wgpu pipeline, bind group layout, and two
atomic counter buffers (invocations_ran, invocations_skipped). Constructed via
`apply_data_changes::linkage::*` auto-generated module.

`BackendUpdate for GpuUpdateTest<WgpuRuntime, TestBackendWgpu>` dispatches
the compute shader on the real GPU via:
1. Zero counter buffers via `queue.write_buffer`
2. Create bind group from arena buffers + counter buffers
3. Encode compute pass with pipeline and bind group
4. Copy counters to staging buffers, submit, poll, map, read back

**Three GPU tests:**
- `gpu_update_test_sanity_on_gpu` — single value, three sequential changes
- `gpu_array_update_test_sanity_on_gpu` — array value, one change
- `proptest_gpu_updates_checked_on_gpu` — property-based (bounds: 8 values, 8
  changes per value, 1-8 entries) with stepwise verification

### Device feature requirement

wgsl-rs linkage sets `ShaderStages::all()` on bind group layout entries. For
`read_write` storage bindings, this requires the `VERTEX_WRITABLE_STORAGE`
device feature. The test device setup (`wgpu_runtime()`) requests this feature.

---

## 5.4 Verify generated WGSL

Inspect the generated WGSL source (from `WGSL_MODULE.wgsl_source()`) to ensure
it is correct and human-readable.

**Status:** Complete. Two inspection tests verify the generated WGSL:
- `wgsl_source_contains_companion_types` — asserts `struct DataId`,
  `struct DataArray`, `struct DataChangeTyId`, `struct DataChange`, and
  `@compute` are present in the output.
- `wgsl_source_validates_with_naga` — calls `WGSL_MODULE.validate()` which
  runs naga validation at test time (belt-and-suspenders with compile-time
  validation).

---

## Test Summary

| Category | Count | Status |
|---|---|---|
| crabslab unit tests | 26 | Passing |
| craballoc arena/range tests | 8 | Passing |
| craballoc CPU dispatch tests | 8 | Passing |
| craballoc GPU dispatch tests | 3 | Passing |
| craballoc WGSL inspection tests | 2 | Passing |
| craballoc proptest (GPU) | 1 | Passing |
| craballoc proptest (CPU) | 2 | Passing |
| **Total** | **50** | **All passing** |
