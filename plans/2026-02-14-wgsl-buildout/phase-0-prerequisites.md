# Phase 0: Prerequisites -- wgsl-rs Changes

**Status:** Complete
**Duration:** Done
**Prerequisites:** None

## Summary

All prerequisite verification and implementation work in the wgsl-rs repository
is complete. The key findings and outcomes are summarized below.

## Verification Results (0.1-0.4)

- **`macro_rules!` inside `#[wgsl]` modules:** Definitions pass through
  unchanged. Invocations in expression position are rejected (only `get!` and
  `get_mut!` are whitelisted). This motivated the switch from per-type macros to
  per-type functions.
- **`as usize` indexing:** Works. `usize` maps to `u32` in WGSL.
- **Cross-module imports:** Glob imports (`use super::module::*`) work. wgsl-rs
  concatenates WGSL source from imported modules via `WGSL_MODULE` constants.
- **Fixed-size arrays:** `[u32; N]` transpiles to `array<u32, N>`. Literals,
  indexing, mutation, and passing as function params/returns all work.
- **`RuntimeArray<u32>`:** Transpiles to `array<u32>` (runtime-sized). Storage
  buffers work. `get!`/`get_mut!` strip to bare identifiers.
- **Atomics:** Full support -- `Atomic<u32>`, `Atomic<i32>`, all 11 WGSL atomic
  builtins, workgroup atomics via `workgroup!(COUNTER: Atomic<u32>)`.
- **`match` as expression:** Rejected. Must use statement form with explicit
  variable assignment.
- **Method receiver syntax:** `self` receiver not supported. Must use
  `Type::method(obj, args)` style.

## `slab_read_array!` / `slab_write_array!` (0.5)

Implemented in wgsl-rs commit `420998a`. Two built-in statement-level macros for
copying data between storage buffers and local fixed-size arrays.

- **`slab_read_array!(slab, offset, dst, size)`** -- emits a WGSL `for` loop
  copying `size` elements from `slab` starting at `offset` into `dst`.
- **`slab_write_array!(slab, offset, src, size)`** -- emits a WGSL `for` loop
  copying `size` elements from `src` into `slab` starting at `offset`.
- **`slab_write_array!(slab, offset, src)`** -- 3-argument form uses
  `arrayLength(&slab)` as the loop bound.

`size` can be a WGSL `const` identifier (e.g., `DATA_SLAB_SIZE`), not just a
literal. GPU drivers are expected to unroll small constant-bounded loops.
