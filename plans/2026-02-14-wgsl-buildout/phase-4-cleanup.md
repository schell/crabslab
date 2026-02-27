# Phase 4: Cleanup and Rename

**Status:** Complete
**Estimated effort:** 2-3 days
**Prerequisites:** Phase 3

## Overview

Rename `crabslab2` to `crabslab`, delete old crates, clean up workspace
configuration, and update documentation.

---

## 4.1 Rename crates

- Rename `crates/crabslab2` to `crates/crabslab`
- Rename `crates/crabslab2-macros` to `crates/crabslab-macros`
- Update all `Cargo.toml` references
- Update all `use crabslab2::` imports to `use crabslab::`

---

## 4.2 Delete old crates

- Delete `crates/crabslab` (the old one, already removed from workspace)
- Delete `crates/crabslab-derive`
- Delete `crates/craballoc-test-shaders`
- Delete `crates/craballoc-test-wire-types`

---

## 4.3 Remove SPIR-V infrastructure

- Remove `spirv-std` from workspace `Cargo.toml`
- Remove `[patch.crates-io]` spirv-std entry
- Remove `wgpu`'s `"spirv"` feature
- Remove the `exclude = ["./shaders"]` line
- Remove all `#[cfg(target_arch = "spirv")]` blocks from any remaining code
- Remove `#![cfg_attr(target_arch = "spirv", no_std)]`
- Remove `#![allow(unexpected_cfgs)]` where no longer needed

---

## 4.4 Update documentation

- Update `README.md` files to reference wgsl-rs instead of rust-gpu
- Update `AGENTS.md` guidelines for the new crate structure
- Update crate descriptions in `Cargo.toml`
- Archive or update `SESSION.md`

---

## 4.5 Update CI

- Remove any `cargo-gpu` references
- Ensure `cargo test` runs all WGSL validation tests
- Vulkan drivers still needed for wgpu GPU tests
