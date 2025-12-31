# AGENTS.md

Guidelines for AI agents working in the crabslab repository.

## Project Overview

`crabslab` is a slab allocator focused on marshalling data between CPUs and GPUs,
designed to work with `rust-gpu` for writing shaders in Rust.

### Workspace Structure

- `crates/crabslab` - Core slab traits (`Slab`, `GrowableSlab`, `SlabItem`) and types (`Id`, `Array`)
- `crates/crabslab-derive` - Proc-macro for deriving `SlabItem`
- `crates/craballoc` - Arena allocator with RAII semantics and wgpu support
- `crates/craballoc-test-shaders` - SPIR-V test shaders (rust-gpu)
- `crates/craballoc-test-wire-types` - Wire types for GPU/CPU communication in tests

## Build, Test, and Lint Commands

```bash
# Build
cargo build
cargo build --release

# Test (all tests)
cargo test
cargo nextest run -j 1          # CI uses nextest with single job for GPU tests

# Run a single test
cargo test test_name
cargo test test_name -- --nocapture   # With output
cargo nextest run test_name -j 1

# Run tests in a specific crate
cargo test -p crabslab
cargo test -p craballoc

# Lint and format
cargo fmt
cargo fmt --check               # CI check
cargo clippy
cargo clippy -- -D warnings     # Treat warnings as errors

# Check specific features
cargo check -p crabslab --no-default-features
cargo check -p craballoc --no-default-features
```

### Feature Flags

- `crabslab`: `default = ["glam", "futures-lite"]`
- `craballoc`: `default = ["wgpu"]`

### CI Environment Notes

- GPU tests require Vulkan: `mesa-vulkan-drivers libvulkan1 vulkan-tools`
- Tests run with `RUST_BACKTRACE=1`
- Use `-j 1` with nextest to avoid GPU resource contention

## Code Style Guidelines

### Imports Organization

Order imports as: std, external crates, then crate-local. Group related imports:

```rust
use std::{
    future::Future,
    ops::{Deref, DerefMut},
    sync::{Arc, RwLock},
};

use snafu::prelude::*;
use tracing::Instrument;

use crate::{
    arena::{Arena, Value},
    range::Range,
    runtime::CpuRuntime,
};
```

### Formatting (rustfmt.toml)

```toml
wrap_comments = true
format_code_in_doc_comments = true
normalize_comments = true
format_strings = true
```

### Naming Conventions

| Element              | Convention              | Example                          |
|----------------------|-------------------------|----------------------------------|
| Types/Structs        | PascalCase              | `SlabItem`, `CpuRuntime`         |
| Functions/Methods    | snake_case              | `read_slab`, `write_indexed`     |
| Constants            | SCREAMING_SNAKE_CASE    | `SLAB_SIZE`, `NONE`, `ZERO`      |
| Field offset consts  | `OFFSET_OF_FIELDNAME`   | `OFFSET_OF_POSITION`             |
| Private helpers      | `__prefix`              | `__saturating_sub`               |
| Test functions       | Descriptive snake_case  | `mngr_updates_count_sanity`      |

### Common Derive Patterns

```rust
// Data types for slab storage
#[derive(Clone, Copy, Debug, Default, PartialEq, SlabItem)]

// Enums used in GPU code (require repr for deterministic layout)
#[derive(Default, Debug, Clone, Copy, SlabItem)]
#[repr(u32)]
enum MyEnum { ... }

// Error types
#[derive(Debug, Snafu)]
#[snafu(visibility(pub(crate)))]
```

### Error Handling with snafu

```rust
use snafu::prelude::*;

#[derive(Debug, Snafu)]
#[snafu(visibility(pub(crate)))]
pub enum Error {
    #[snafu(display("Slab has no internal buffer"))]
    NoInternalBuffer,

    #[snafu(display("Async recv error: {source}"))]
    AsyncRecv { source: async_channel::RecvError },

    #[cfg(feature = "wgpu")]
    #[snafu(display("Async error: {source}"))]
    Async { source: wgpu::BufferAsyncError },
}

// Usage with context extension
use snafu::{OptionExt, ResultExt};
let buffer = self.get_buffer().context(NoInternalBufferSnafu)?;
result.context(AsyncRecvSnafu)?;
```

### Conditional Compilation (CPU vs GPU)

```rust
#![allow(unexpected_cfgs)]
#![cfg_attr(target_arch = "spirv", no_std)]

#[cfg(not(target_arch = "spirv"))]
fn cpu_only_function() { ... }

#[cfg(target_arch = "spirv")]
fn gpu_only_function() { ... }

// Feature-gated code
#[cfg(feature = "wgpu")]
impl WgpuRuntime { ... }
```

### Documentation Style

```rust
//! Module-level documentation.
//!
//! More details about the module.
#![doc = include_str!("../README.md")]

/// Brief description of the function.
///
/// More detailed explanation.
///
/// ## Note
/// Important considerations.
///
/// ## Errors
/// When this function can fail.
fn example() { }
```

Use `#[cfg(doc)]` for documentation-only imports:

```rust
#[cfg(doc)]
use crate::prelude::*;
```

### Test Patterns

```rust
#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn my_test_sanity() {
        // Initialize logger for test output
        let _ = env_logger::builder().is_test(true).try_init();

        // Test implementation...
    }
}

// Async tests with futures-lite
let result = futures_lite::future::block_on(async_operation()).unwrap();

// Property-based testing with proptest
proptest! {
    #[test]
    fn proptest_example(value in arb_value()) {
        let _ = env_logger::builder().is_test(true).try_init();
        // Test with generated value
    }
}
```

### Logging and Tracing

```rust
// Logging with log crate
log::trace!("detailed info");
log::debug!("debug info: {:?}", value);
log::info!("important info");

// Tracing for instrumentation
#[tracing::instrument(skip_all)]
async fn instrumented_function() { ... }

let span = tracing::trace_span!("operation-name");
span.in_scope(|| { ... });
```

## Project-Specific Patterns

### SlabItem Trait and Derive

Types that can be stored in a slab implement `SlabItem`:

```rust
use crabslab::{SlabItem, Id, Array, Slab, GrowableSlab, CpuSlab};

#[derive(Debug, Default, PartialEq, SlabItem)]
struct MyData {
    position: glam::Vec4,
    value: u32,
}

// Use #[offsets] to generate field offset constants
#[derive(SlabItem)]
#[offsets]
struct WithOffsets {
    a: u32,  // Generates OFFSET_OF_A, SLAB_SIZE_OF_A
    b: f32,  // Generates OFFSET_OF_B, SLAB_SIZE_OF_B
}
```

### Core Types

- `Id<T>` - Typed index into the slab (`#[repr(transparent)]` over `u32`)
- `Array<T>` - Contiguous elements pointer (starting `Id` + length)
- `Value<T>` - Arena-allocated value with CPU cache and GPU sync
- `Offset<F, T>` - Type-safe field offset within a struct

### Runtime Abstractions

```rust
// CPU-only runtime (for testing)
let arena = Arena::new(&CpuRuntime, "label", None);

// wgpu runtime (for GPU)
let runtime = WgpuRuntime { device, queue };
let arena = Arena::new(&runtime, "label", None);

// Commit changes to GPU buffer
let buffer = arena.commit();

// Read back from GPU (async)
let data = futures_lite::future::block_on(arena.read_slab(array)).unwrap();
```
