# Phase 3: Wire Types and Compute Shader (Test Modules)

**Status:** Pending
**Estimated effort:** 1 week
**Prerequisites:** Phase 2

## Overview

Rewrite the wire types and compute shader as `#[cfg(test)]` modules inside
`craballoc`, using `#[slab_module]` / `#[slab_item]` and wgsl-rs's
`linkage-wgpu` feature. The separate `craballoc-test-shaders` and
`craballoc-test-wire-types` crates are no longer needed.

---

## 3.1 Wire types as a test module

Move the wire type definitions into a `#[cfg(test)]` module inside `craballoc`:

```rust
#[cfg(test)]
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
        pub fn apply(change: DataChange, data: Data) -> Data {
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

## 3.2 Compute shader as a test module

```rust
#[cfg(test)]
#[slab_module]
pub mod apply_data_changes {
    use wgsl_rs::std::*;
    use super::wire_types::*;

    storage!(group(0), binding(0), read_write, DATA_SLAB: RuntimeArray<u32>);
    storage!(group(0), binding(1), CHANGES_SLAB: RuntimeArray<u32>);

    #[compute]
    #[workgroup_size(16, 1, 1)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        // Read invocation descriptor, apply changes, write back
        // (same logic as Phase 5 in the original plan)
    }
}
```

---

## 3.3 Tuple flattening

Current code uses tuples like `(u32, u32)` for `Data.ints`. WGSL has no tuples.
These are flattened to `ints_0: u32, ints_1: u32`.

---

## 3.4 CPU-only helpers

CPU-only code (like `Display` impls) lives **outside** the `#[slab_module]`
module, gated behind `#[cfg(test)]`.

---

## 3.5 Atomic operations

Atomic counters for invocation counting use wgsl-rs's `atomic_add` with a
separate `Atomic<u32>` storage binding.

---

## 3.6 Use `linkage-wgpu` for wgpu integration

Enable wgsl-rs's `linkage-wgpu` feature. The `#[wgsl]` macro generates:
- `apply_data_changes::linkage::shader_module(device)`
- `apply_data_changes::linkage::bind_group_0::layout(device)`
- `apply_data_changes::linkage::main::WORKGROUP_SIZE`

Update `TestBackendWgpu` to use the generated linkage instead of manual pipeline
setup.
