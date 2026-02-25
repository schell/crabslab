# Phase 4: Migrate Wire Types (`craballoc-test-wire-types`)

**Status:** Pending
**Estimated effort:** 3-4 days
**Prerequisites:** Phases 1, 2

## Overview

Rewrite the `craballoc-test-wire-types` crate to use `#[slab_module]` /
`#[slab_item]`, flatten tuples to individual fields, move CPU-only helpers
outside the WGSL module, and handle atomic operations.

---

## 4.1 Rewrite as `#[slab_module]`

```rust
#[slab_module]
pub mod wire_types {
    use wgsl_rs::std::*;

    #[slab_item]
    pub struct Data {
        pub i: u32,
        pub float_val: f32,
        pub ints_0: u32,
        pub ints_1: u32,
    }

    #[slab_item]
    #[repr(u32)]
    pub enum DataChangeTy {
        I = 0,
        Float = 1,
        Ints = 2,
    }

    #[slab_item]
    pub struct DataChange {
        pub ty: DataChangeTy,
        pub data_0: u32,
        pub data_1: u32,
        pub data_2: u32,
    }

    impl DataChange {
        #[wgsl_allow(non_literal_match_statement_patterns)]
        pub fn apply(change: DataChange, data: Data) -> Data {
            // NOTE: match must be a statement, not an expression.
            // wgsl-rs rejects match in expression context.
            let mut result = data;
            match change.ty {
                DataChangeTy::I => {
                    result = Data {
                        i: change.data_0,
                        float_val: data.float_val,
                        ints_0: data.ints_0,
                        ints_1: data.ints_1,
                    };
                },
                DataChangeTy::Float => {
                    result = Data {
                        i: data.i,
                        float_val: bitcast_f32(change.data_0),
                        ints_0: data.ints_0,
                        ints_1: data.ints_1,
                    };
                },
                _ => {
                    result = Data {
                        i: data.i,
                        float_val: data.float_val,
                        ints_0: change.data_0,
                        ints_1: change.data_1,
                    };
                },
            }
            result
        }
    }

    #[slab_item]
    pub struct ArrayChange {
        pub i: u32,
        pub change: DataChange,
    }

    #[slab_item]
    pub struct AnyChangeId {
        pub change_id: ArrayChangeId,
        pub data_array: DataArray,
    }

    #[slab_item]
    pub struct InvocationCount(pub u32);

    #[slab_item]
    pub struct ApplyDataChangeInvocation {
        pub changes_ids: AnyChangeIdArray,
        pub invocations_id: InvocationCountId,
        pub invocations_skipped_id: InvocationCountId,
    }
}
```

---

## 4.2 Tuple flattening

Current code uses tuples like `(u32, u32)` for `Data.ints`. WGSL has no tuples.
These are flattened to `ints_0: u32, ints_1: u32`.

---

## 4.3 CPU-only helpers

CPU-only code (like `Display` impls, `DataChange::new()` constructors that use
`SlabItem` generics) lives **outside** the `#[slab_module]` module:

```rust
// Outside the #[wgsl] module -- CPU-only
impl core::fmt::Display for wire_types::DataChange {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // ...
    }
}
```

---

## 4.4 Atomic operations

The current `spirv_std::arch::atomic_i_increment` for invocation counting is
replaced with wgsl-rs's `atomic_add`. This requires a separate `Atomic<u32>`
storage variable, since atomics on arbitrary indices within a
`RuntimeArray<u32>` are not directly supported in WGSL.

Options:
- Use a separate `storage!(group(0), binding(2), read_write, COUNTERS: Counters)`
  where `Counters` contains `Atomic<u32>` fields
- Use `workgroup!` variables for workgroup-scoped atomic counters
- Drop atomic counting if it's only used for test validation

wgsl-rs has full atomic support: `Atomic<u32>`, `Atomic<i32>`, all 11 WGSL
atomic builtins (`atomic_add`, `atomic_load`, `atomic_store`, etc.), and
workgroup atomics via `workgroup!(COUNTER: Atomic<u32>)`.
