# Phase 7: Testing and Verification

**Status:** Pending
**Estimated effort:** 1 week
**Prerequisites:** Phase 6

## Overview

Validate the migration end-to-end: WGSL validation, CPU-side tests (including
property-based tests), GPU-side tests via wgpu, and manual inspection of
generated WGSL output.

---

## 7.1 WGSL validation

All `#[wgsl]` modules with no imports validate at compile time via naga.
Modules with imports validate at test time via auto-generated
`__validate_wgsl()` tests.

---

## 7.2 CPU-side tests

Run the existing property-based tests (proptest) that test
`ApplyDataChangeInvocation` on the CPU backend. These should work identically
since the Rust code inside `#[wgsl]` modules is real, executable Rust.

---

## 7.3 GPU-side tests

Run the wgpu-based tests with the new WGSL shader. Compare results with the CPU
backend to verify correctness.

---

## 7.4 Verify generated WGSL

Inspect the generated WGSL source (from `WGSL_MODULE.wgsl_source()`) to ensure
it is correct and human-readable.
