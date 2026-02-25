# Phase 5: Migrate Compute Shader (`craballoc-test-shaders`)

**Status:** Pending
**Estimated effort:** 3-4 days
**Prerequisites:** Phases 3, 4

## Overview

Rewrite the `craballoc-test-shaders` crate as a regular Rust crate (no
`#![no_std]`, no `spirv-std`) using `#[slab_module]` and `#[wgsl]` for compute
shader generation. Use wgsl-rs's `linkage-wgpu` feature for wgpu integration.

---

## 5.1 Rewrite as `#[wgsl]` compute shader

The `craballoc-test-shaders` crate becomes a regular Rust crate:

```rust
#[slab_module]
pub mod apply_data_changes {
    use wgsl_rs::std::*;
    use super::wire_types::*;

    storage!(group(0), binding(0), read_write, DATA_SLAB: RuntimeArray<u32>);
    storage!(group(0), binding(1), CHANGES_SLAB: RuntimeArray<u32>);

    #[compute]
    #[workgroup_size(16, 1, 1)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        // Read the invocation descriptor from the changes slab
        let mut inv_raw = [0u32; APPLY_DATA_CHANGE_INVOCATION_SLAB_SIZE];
        slab_read_array!(
            get!(CHANGES_SLAB),
            ApplyDataChangeInvocationId::ZERO.inner,
            inv_raw,
            APPLY_DATA_CHANGE_INVOCATION_SLAB_SIZE
        );
        let invocation = apply_data_change_invocation_from_array(inv_raw);

        let index = global_id.x();
        if index >= invocation.changes_ids.len {
            return;
        }

        // Read which change to apply and which data array it targets
        let info_id = AnyChangeIdArray::at(invocation.changes_ids, index);
        let mut info_raw = [0u32; ANY_CHANGE_ID_SLAB_SIZE];
        slab_read_array!(
            get!(CHANGES_SLAB),
            info_id.inner,
            info_raw,
            ANY_CHANGE_ID_SLAB_SIZE
        );
        let change_info = any_change_id_from_array(info_raw);

        // Read the change itself
        let mut change_raw = [0u32; ARRAY_CHANGE_SLAB_SIZE];
        slab_read_array!(
            get!(CHANGES_SLAB),
            change_info.change_id.inner,
            change_raw,
            ARRAY_CHANGE_SLAB_SIZE
        );
        let change = array_change_from_array(change_raw);

        // Read the target data element
        let data_id = DataArray::at(change_info.data_array, change.i);
        let mut data_raw = [0u32; DATA_SLAB_SIZE];
        slab_read_array!(
            get!(DATA_SLAB),
            data_id.inner,
            data_raw,
            DATA_SLAB_SIZE
        );
        let data = data_from_array(data_raw);

        // Apply the change and write back
        let new_data = DataChange::apply(change.change, data);
        // Write back using 4-argument form (explicit size).
        // Alternatively, the 3-argument form `slab_write_array!(get_mut!(DATA_SLAB),
        // data_id.inner, out)` omits the size and uses `arrayLength(&DATA_SLAB)`
        // as the loop bound in WGSL, but this copies more than necessary.
        let out = data_to_array(new_data);
        slab_write_array!(
            get_mut!(DATA_SLAB),
            data_id.inner,
            out,
            DATA_SLAB_SIZE
        );
    }
}
```

---

## 5.2 Use `linkage-wgpu` for wgpu integration

Enable wgsl-rs's `linkage-wgpu` feature. The `#[wgsl]` macro then generates:
- `apply_data_changes::linkage::shader_module(device)` -- creates
  `wgpu::ShaderModule`
- `apply_data_changes::linkage::bind_group_0::layout(device)` -- creates
  `wgpu::BindGroupLayout`
- `apply_data_changes::linkage::main::WORKGROUP_SIZE` -- the workgroup size
  constant
- Per-binding buffer descriptor helpers

---

## 5.3 Update `TestBackendWgpu`

Replace the manual pipeline setup in `craballoc/src/test/wgpu.rs` with the
generated linkage:

```rust
use craballoc_test_shaders::apply_data_changes;

pub struct TestBackendWgpu {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl TestBackendWgpu {
    pub fn new(runtime: WgpuRuntime) -> Self {
        let module = apply_data_changes::linkage::shader_module(&runtime.device);
        let bind_group_layout =
            apply_data_changes::linkage::bind_group_0::layout(&runtime.device);
        let pipeline_layout =
            runtime
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("test"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });
        let pipeline =
            runtime
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("test"),
                    layout: Some(&pipeline_layout),
                    module: &module,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });
        Self {
            pipeline,
            bind_group_layout,
        }
    }
}
```
