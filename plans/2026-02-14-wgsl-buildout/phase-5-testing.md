# Phase 5: Testing and Verification

**Status:** Pending
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

---

## 5.2 CPU-side tests

Run the existing property-based tests (proptest) that test
`ApplyDataChangeInvocation` on the CPU backend. These should work identically
since the Rust code inside `#[wgsl]` modules is real, executable Rust.

---

## 5.3 GPU-side tests

Run the wgpu-based tests with the new WGSL shader. Compare results with the CPU
backend to verify correctness.

---

## 5.4 Verify generated WGSL

Inspect the generated WGSL source (from `WGSL_MODULE.wgsl_source()`) to ensure
it is correct and human-readable.
