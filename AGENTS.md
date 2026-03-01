# AGENTS.md

Guidelines for AI agents working in the crabslab repository.

## Project Overview

`crabslab` is a slab allocator focused on marshalling data between CPUs and GPUs.
The project has migrated from `rust-gpu`/SPIR-V to `wgsl-rs` (Rust-to-WGSL
transpilation). The migration plan lives in `plans/2026-02-14-wgsl-buildout/` -- see
`plans/2026-02-14-wgsl-buildout/README.md` for the overview and index of all
phase documents, which are the source of truth for all migration decisions.

### Workspace Structure

- `crates/crabslab` - Core `SlabItem` trait, `slab_read`/`slab_write` helpers, `slab_read!`/`slab_write!` macros, re-exports `#[slab_module]`/`#[slab_item]`
- `crates/crabslab-macros` - Proc-macros: `#[slab_module]` and `#[slab_item]`, `slab_read!`/`slab_write!` expansion
- `crates/craballoc` - Arena allocator with RAII semantics and wgpu support

## Build, Test, and Lint Commands

**IMPORTANT:** Always use `cargo nextest run` instead of `cargo test` for this
workspace. Several craballoc tests share global wgsl-rs storage statics and will
interfere with each other under `cargo test`'s in-process parallelism.
`cargo nextest run` isolates each test in its own process.

```bash
# Build
cargo build
cargo build --release

# Test (all tests) — always use nextest
cargo nextest run -j 1          # CI uses nextest with single job for GPU tests
cargo nextest run                # Also fine; process isolation prevents conflicts

# Run a single test
cargo nextest run test_name -j 1
cargo nextest run test_name -j 1 --no-capture   # With output

# Run tests in a specific crate
cargo nextest run -p crabslab2
cargo nextest run -p craballoc

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

- `crabslab`: no features (no default features)
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
// Data types for slab storage (inside #[slab_module])
// NOTE: #[slab_item] does NOT auto-derive these — you must add them yourself
#[slab_item]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MyType { ... }

// Enums used in GPU code (require repr for deterministic layout)
#[slab_item]
#[derive(Clone, Copy, Debug, Default)]
#[repr(u32)]
pub enum MyEnum {
    #[default]
    First = 0,
    ...
}

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

### Conditional Compilation

```rust
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
use crabslab::{SlabItem, slab_read, slab_write};

// Inside a #[slab_module], use #[slab_item] to generate:
// - SLAB_SIZE constant
// - from_array / to_array functions
// - TypeId / TypeArray companion types
// - impl SlabItem for Type
#[slab_item]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MyData {
    pub x: u32,
    pub y: f32,
}
```

### Combining `#[slab_module]` with `#[wgsl]`

**Do NOT stack `#[slab_module]` and `#[wgsl]` as separate attributes.** Instead,
pass a `wgsl(...)` parameter group to `#[slab_module]`:

```rust
// Correct — slab_module generates companion types, then emits #[wgsl] on the output
#[crabslab::slab_module(wgsl())]
pub mod my_shader { ... }

// Also correct — bare #[slab_module] (no WGSL transpilation)
#[crabslab::slab_module]
pub mod my_types { ... }

// Custom wgsl crate path
#[crabslab::slab_module(wgsl_crate = my_wgsl, wgsl())]
pub mod my_shader { ... }
```

The `wgsl(...)` group is forwarded as `#[wgsl_rs::wgsl(...)]` on the output
module. This ensures `#[wgsl]` always runs *after* companion types have been
generated, eliminating macro ordering issues.

### `slab_read!` / `slab_write!` Macros

Inside a `#[slab_module]`, use `slab_read!` and `slab_write!` to read/write
`#[slab_item]` types from/to storage buffers. These are expanded by
`#[slab_module]` into `slab_read_array!` + `from_array` / `to_array` +
`slab_write_array!` before `#[wgsl]` runs, so they work on both CPU and GPU.

```rust
// Read a typed value from a storage slab at an offset
let data = slab_read!(Data, get!(DATA_SLAB), offset);

// Write a typed value to a storage slab at an offset
slab_write!(Data, get_mut!(DATA_SLAB), offset, data);
```

Outside a `#[slab_module]`, the `macro_rules!` fallback in the `crabslab`
crate provides CPU-only behavior using the `SlabItem` trait.

### Core Types

- `Value<T>` - Arena-allocated value with CPU cache and GPU sync
- Per-type ID structs (e.g., `DataId { inner: u32 }`) — generated by `#[slab_item]`
- Per-type Array structs (e.g., `DataArray { id: DataId, len: u32 }`) — generated by `#[slab_item]`

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
