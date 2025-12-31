# SESSION.md: crabslab Migration from Rust-GPU to wgsl-rs

This document captures the migration plan for transitioning crabslab from Rust-GPU to wgsl-rs.

## Background

`crabslab` is a slab allocator for marshalling data between CPUs and GPUs. It currently uses
Rust-GPU to compile shared Rust code to SPIR-V for GPU execution. The goal is to migrate to
`wgsl-rs`, which transpiles a subset of Rust to WGSL while keeping the Rust code **operational**
(runs on CPU) and **isomorphic** (types work on both sides).

### Key wgsl-rs Properties

- Code inside `#[wgsl]` modules runs on CPU AND generates WGSL
- Types defined in `#[wgsl]` modules are real Rust types usable on CPU
- Upcoming support for `impl` blocks with explicit receiver syntax: `Light::attenuate(light, amount)`
- No traits, no generics, no mutable references
- `match` support coming (transpiles to WGSL `switch`)
- Bitcast operations supported
- Array indexing for storage buffers supported

## Migration Summary

The migration is primarily a **refactoring exercise**:

1. Replace `SlabItem` trait with explicit `Type::read()`/`Type::write()` methods
2. Replace `Id<T>` generics with concrete newtype IDs
3. Move shared GPU/CPU types into `#[wgsl]` modules
4. Keep CPU-only code (Arena, Value, RAII) outside `#[wgsl]` modules

---

## Architecture Overview

```
crates/
├── crabslab/                    # Core slab types (mostly #[wgsl] modules)
│   └── src/
│       ├── lib.rs               # Re-exports, CPU-only utilities
│       ├── types.rs             # #[wgsl] module with Id, Array, core types
│       └── slab.rs              # #[wgsl] module with slab read/write ops
│
├── crabslab-derive/             # REMOVED or repurposed
│   └── (proc-macro no longer needed for SlabItem)
│
├── craballoc/                   # Arena allocator (CPU-only, unchanged API)
│   └── src/
│       ├── lib.rs
│       ├── arena.rs             # Uses types from crabslab #[wgsl] modules
│       └── runtime.rs
│
├── craballoc-test-wire-types/   # #[wgsl] module for test types
│   └── src/lib.rs               # Data, DataChange, etc. in #[wgsl] module
│
└── craballoc-test-shaders/      # #[wgsl] compute shaders
    └── src/lib.rs               # apply_data_changes shader
```

---

## Phase 1: Core Types Migration

### 1.1 Replace `Id<T>` with Concrete Newtypes

**Current:**
```rust
pub struct Id<T>(pub(crate) u32, PhantomData<T>);
```

**New (inside `#[wgsl]` module):**
```rust
#[wgsl]
pub mod slab {
    use wgsl_rs::std::*;

    pub const ID_NONE: u32 = 4294967295u32;  // u32::MAX

    // Base Id type for when you don't need type safety
    pub struct SlabId {
        pub index: u32,
    }

    impl SlabId {
        pub const NONE: SlabId = SlabId { index: ID_NONE };
        pub const ZERO: SlabId = SlabId { index: 0 };

        pub fn new(index: u32) -> SlabId {
            SlabId { index }
        }

        pub fn is_none(id: SlabId) -> bool {
            id.index == ID_NONE
        }

        pub fn is_some(id: SlabId) -> bool {
            id.index != ID_NONE
        }
    }
}
```

**Concrete newtypes for specific types:**
```rust
#[wgsl]
pub mod data_types {
    use super::slab::*;

    pub struct DataId { pub index: u32 }

    impl DataId {
        pub const NONE: DataId = DataId { index: ID_NONE };

        pub fn new(index: u32) -> DataId {
            DataId { index }
        }
    }
}
```

### 1.2 Replace `Array<T>` Similarly

```rust
#[wgsl]
pub mod slab {
    // ...

    pub struct SlabArray {
        pub id: SlabId,
        pub len: u32,
    }

    impl SlabArray {
        pub const NONE: SlabArray = SlabArray {
            id: SlabId::NONE,
            len: 0
        };

        pub fn new(id: SlabId, len: u32) -> SlabArray {
            SlabArray { id, len }
        }

        pub fn at(arr: SlabArray, index: u32, item_size: u32) -> SlabId {
            if index >= arr.len {
                SlabId::NONE
            } else {
                SlabId::new(arr.id.index + item_size * index)
            }
        }
    }
}
```

**Type-specific arrays:**
```rust
pub struct DataArray { pub id: DataId, pub len: u32 }

impl DataArray {
    pub fn at(arr: DataArray, index: u32) -> DataId {
        // DATA_SLAB_SIZE known at definition time
        DataId::new(arr.id.index + Data::SLAB_SIZE * index)
    }
}
```

### 1.3 Replace `SlabItem` Trait with Explicit Methods

**Current:**
```rust
pub trait SlabItem {
    const SLAB_SIZE: usize;
    fn read_slab(index: usize, slab: &[u32]) -> Self;
    fn write_slab(&self, index: usize, slab: &mut [u32]) -> usize;
}
```

**New pattern (per type):**
```rust
#[wgsl]
pub mod data_types {
    use wgsl_rs::std::*;

    pub struct Data {
        pub i: u32,
        pub float_val: f32,
        pub ints_0: u32,
        pub ints_1: u32,
    }

    impl Data {
        pub const SLAB_SIZE: u32 = 4u32;

        pub fn read(id: DataId, slab: ptr<storage, array<u32>, read>) -> Data {
            let idx = id.index;
            Data {
                i: slab[idx],
                float_val: bitcast<f32>(slab[idx + 1u32]),
                ints_0: slab[idx + 2u32],
                ints_1: slab[idx + 3u32],
            }
        }

        pub fn write(data: Data, id: DataId, slab: ptr<storage, array<u32>, read_write>) {
            let idx = id.index;
            slab[idx] = data.i;
            slab[idx + 1u32] = bitcast<u32>(data.float_val);
            slab[idx + 2u32] = data.ints_0;
            slab[idx + 3u32] = data.ints_1;
        }
    }
}
```

---

## Phase 2: Derive Macro Decision

Given that each type now needs explicit `read()`/`write()` implementations, options:

**Option A: Remove `crabslab-derive` entirely**
- Users write `read()`/`write()` manually
- More explicit, less magic
- Works for types with special serialization needs

**Option B: Create a new `#[wgsl_slab_item]` attribute macro**
- Generates the `impl` block with `SLAB_SIZE`, `read()`, `write()`
- Works within `#[wgsl]` modules
- Less boilerplate

**Option C: Hybrid approach**
- Provide a helper macro for simple structs
- Manual implementation for complex cases

**Recommendation:** Start with manual implementations (Option A) to validate the pattern,
then add a macro (Option C) if boilerplate becomes burdensome.

---

## Phase 3: Wire Types Migration

Migrate `craballoc-test-wire-types` to `#[wgsl]` module:

```rust
#[wgsl]
pub mod wire_types {
    use wgsl_rs::std::*;
    use crabslab::slab::*;

    pub struct Data {
        pub i: u32,
        pub float_val: f32,
        pub ints_0: u32,
        pub ints_1: u32,
    }

    // Enum as u32 discriminant (until match is supported)
    pub const DATA_CHANGE_TY_I: u32 = 0u32;
    pub const DATA_CHANGE_TY_FLOAT: u32 = 1u32;
    pub const DATA_CHANGE_TY_INTS: u32 = 2u32;

    pub struct DataChange {
        pub ty: u32,
        pub data_0: u32,
        pub data_1: u32,
        pub data_2: u32,
    }

    impl DataChange {
        pub fn apply(change: DataChange, data: Data) -> Data {
            // When match is available:
            // match change.ty {
            //     DATA_CHANGE_TY_I => Data { i: change.data_0, ..data },
            //     ...
            // }

            // For now, use if/else:
            if change.ty == DATA_CHANGE_TY_I {
                Data { i: change.data_0, float_val: data.float_val, ints_0: data.ints_0, ints_1: data.ints_1 }
            } else if change.ty == DATA_CHANGE_TY_FLOAT {
                Data { i: data.i, float_val: bitcast<f32>(change.data_0), ints_0: data.ints_0, ints_1: data.ints_1 }
            } else {
                Data { i: data.i, float_val: data.float_val, ints_0: change.data_0, ints_1: change.data_1 }
            }
        }
    }
}
```

---

## Phase 4: Compute Shader Migration

```rust
#[wgsl]
pub mod apply_data_changes {
    use wgsl_rs::std::*;
    use crate::wire_types::*;

    storage!(group(0), binding(0), read_write, DATA_SLAB: [u32; 65536]);
    storage!(group(0), binding(1), CHANGES_SLAB: [u32; 65536]);

    #[compute]
    #[workgroup_size(16, 1, 1)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let invocation = ApplyDataChangeInvocation::read(
            SlabId::ZERO,
            &CHANGES_SLAB
        );
        let index = global_id.x;

        if index >= invocation.changes_ids.len {
            return;
        }

        let change_info = AnyChangeId::read(
            DataArray::at(invocation.changes_ids, index),
            &CHANGES_SLAB
        );

        let change = ArrayChange::read(change_info.change_id, &CHANGES_SLAB);
        let data_id = DataArray::at(change_info.data_array, change.i);
        let data = Data::read(data_id, &DATA_SLAB);
        let new_data = DataChange::apply(change.change, data);
        Data::write(new_data, data_id, &DATA_SLAB);
    }
}
```

---

## Phase 5: CPU-Side Adapter Layer

The `craballoc` crate's `Arena`, `Value<T>`, etc. remain CPU-only Rust code.
They use types from `#[wgsl]` modules:

```rust
// In craballoc, NOT in a #[wgsl] module
use crabslab::wire_types::{Data, DataId};

impl Arena {
    pub fn new_data(&self, data: Data) -> Value<Data> {
        // Allocates space, returns wrapper
    }
}
```

---

## Open Questions

### 1. Struct Update Syntax

Does wgsl-rs support `..data` struct update syntax? If not, full struct construction is needed:
```rust
// With struct update:
Data { i: change.data_0, ..data }

// Without:
Data { i: change.data_0, float_val: data.float_val, ints_0: data.ints_0, ints_1: data.ints_1 }
```

### 2. Storage Buffer Sizing

The `[u32; 65536]` fixed size in storage declarations. Options:
- Use a large fixed size (current approach in examples)
- Generate shaders with configurable sizes at build time
- Wait for runtime-sized array support in wgsl-rs

### 3. Atomic Operations

The current code uses `spirv_std::arch::atomic_i_increment` for invocation counting.
Does wgsl-rs support WGSL atomics (`atomicAdd`, etc.), or is this planned?

### 4. `craballoc` Generics

Should the arena remain generic (`Arena::new_value<T>`) or become concrete?
If generic, we need a CPU-side trait that wraps wgsl types:

```rust
// Option A: Concrete methods per type
impl Arena {
    pub fn new_data(&self, data: Data) -> Value<Data> { ... }
    pub fn new_light(&self, light: Light) -> Value<Light> { ... }
}

// Option B: CPU-side trait wrapper
pub trait CpuSlabItem {
    const SLAB_SIZE: usize;
    fn write_to_slab(&self, slab: &mut [u32], index: usize);
    fn read_from_slab(slab: &[u32], index: usize) -> Self;
}

// Implement for wgsl types
impl CpuSlabItem for Data {
    const SLAB_SIZE: usize = Data::SLAB_SIZE as usize;
    // ...
}
```

### 5. Timeline/Priority

Is this migration blocking other work, or can it be done incrementally alongside
new development?

### 6. Tuple Flattening Convention

Current crabslab uses tuples like `(u32, u32)`. WGSL has no tuples.
Proposed convention: flatten to `fieldname_0`, `fieldname_1`, etc.
Is this acceptable?

---

## Estimated Timeline

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Phase 1: Core types | 1 week | wgsl-rs impl block support |
| Phase 2: Derive macro decision | 2-3 days | Phase 1 complete |
| Phase 3: Wire types | 3-4 days | Phase 1 complete |
| Phase 4: Compute shader | 3-4 days | Phases 1, 3 complete |
| Phase 5: CPU adapter | 1 week | Phases 1-4 complete |
| Testing & polish | 1 week | All phases complete |

**Total: ~4-5 weeks**

---

## Dependencies on wgsl-rs Development

The migration depends on these wgsl-rs features:

| Feature | Status | Required By |
|---------|--------|-------------|
| `impl` blocks with explicit receiver | In progress | Phase 1 |
| Bitcast operations | Supported | Phase 1 |
| Array indexing for storage | Supported | Phase 1 |
| `match` on enums | Planned | Phase 3 (workaround: if/else) |
| Atomics | Unknown | Phase 4 (for invocation counting) |

---

## Success Criteria

1. All existing `craballoc` tests pass with wgsl-rs backend
2. Generated WGSL validates via naga
3. CPU-side types from `#[wgsl]` modules work identically to current `SlabItem` types
4. No Rust-GPU/SPIR-V dependencies remain
5. Human-readable WGSL output for debugging
