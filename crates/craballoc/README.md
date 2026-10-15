<div style="float: right; padding: 1em;">
    <img src="https://github.com/schell/crabslab/blob/main/crates/crabslab/crabslab.png?raw=true" alt="slabcraft for crabs" width="256" />
</div>

## What

`craballoc` is a slab allocator built on top of
[`crabslab`](https://github.com/schell/crabslab) that provides RAII primitives
for synchronizing values across CPU and GPU code.

## But Why?

Opinion: working with _shaders_ is much easier using a slab.

Shader code can be written in Rust with
[`wgsl-rs`](https://github.com/schell/wgsl-rs), and
[`crabslab`](https://github.com/schell/crabslab) lets you use the same types on
both sides: one `#[derive(SlabItem)]` covers the CPU-side serialization and,
through the `crabslab::SlabItemExt` extension, the GPU side too.

`craballoc` adds the allocator half of the story: it hands out values backed by
a slab buffer, tracks which of them changed, and synchronizes CPU edits to the
GPU - and GPU edits back to the CPU - with RAII. No manual buffer management,
no per-value bind points to keep straight.

## And How

The idea is simple - `craballoc` provides [`Arena<R>`], where `R` is a runtime
implementing [`IsRuntime`]. The `wgpu` feature (**on** by default) provides
[`WgpuRuntime`], which drives real `wgpu::Buffer`s; [`CpuRuntime`] is a CPU-only
reference implementation, mostly useful in tests.

Your types derive [`SlabItem`](crabslab::SlabItem) so they can be written to
and read from the slab on both sides.

On the CPU, allocate with [`Arena::new_value`] or [`Arena::new_array`], receiving
a [`Value<T>`] or a `Value<[T]>`:

* read and edit the value locally with [`Value::get`], [`Value::set`], and
  [`Value::modify`]; edits are queued automatically
* choose a synchronization direction with [`Value::into_bidirectional`] (the
  default), [`Value::into_one_way_from_cpu`], [`Value::into_one_way_from_gpu`],
  or [`Value::into_sync_none`]

Synchronize once per frame (or more, or less):

* [`Arena::commit`] uploads queued CPU changes to the GPU
* [`Arena::synchronize`] copies GPU changes back into the CPU caches,
  where [`Value::get`] reads them

On the GPU, write your shader as a `wgsl-rs` `#[wgsl]` module that runs the
`crabslab::SlabItemExt` extension and reads and writes values with
`crabslab`'s `slab_read!`/`slab_write!` macros - the same derive generates the
GPU-side serialization. See the
[crabslab docs](https://docs.rs/crabslab/latest/crabslab/) for the GPU-side
details, and the `craballoc-test-shaders` crate in this repository for a
complete shader.

## Example

One definition, two worlds: the CPU allocates a `Lamp` with
[`Arena::new_value`], edits it with [`Value::modify`], and uploads it with
[`Arena::commit`]. Then a `wgsl-rs` compute shader reads the lamp with
`slab_read!`, adds `1.0` to its intensity, and writes it back with
`slab_write!`. The wgpu pipeline objects are generated from the shader by
[`wgsl-rs-linkage-wgpu`](https://crates.io/crates/wgsl-rs-linkage-wgpu).

```rust
use craballoc::prelude::*;
use wgsl_rs::wgsl;
use wgsl_rs_linkage_wgpu::v29 as wgpu_linkage;

#[wgsl(extensions = [crabslab::SlabItemExt])]
mod lamp_shader {
    use crabslab::slab_item::*;
    use wgsl_rs::std::*;

    /// A lamp in the scene, shared between CPU and GPU code. The derive
    /// covers both sides: CPU serialization here on the CPU, and the
    /// GPU-side serialization in the assembled WGSL.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Wgsl, SlabItem)]
    pub struct Lamp {
        pub intensity: f32,
        pub on: bool,
    }

    storage!(group(0), binding(0), read_write, SLAB: RuntimeArray<u32>);

    #[compute]
    #[workgroup_size(1)]
    pub fn main(#[builtin(local_invocation_index)] _i: u32) {
        let mut lamp: Lamp;
        slab_read!(Lamp, get!(SLAB), 0u32, lamp);
        lamp.intensity += 1.0;
        slab_write!(Lamp, get_mut!(SLAB), 0u32, lamp);
    }
}

# // WGPU device setup, elided from the rendered docs.
# let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
#     backends: wgpu::Backends::PRIMARY,
#     ..wgpu::InstanceDescriptor::new_without_display_handle()
# });
# let adapter = futures_lite::future::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
#     power_preference: wgpu::PowerPreference::default(),
#     compatible_surface: None,
#     force_fallback_adapter: false,
# }))
# .unwrap();
# let (device, queue) = futures_lite::future::block_on(
#     adapter.request_device(&wgpu::DeviceDescriptor::default()),
# )
# .unwrap();
let runtime = WgpuRuntime { device: device.into(), queue: queue.into() };

let arena = Arena::new(&runtime, "lamp-arena", None);
let lamp = arena.new_value(lamp_shader::Lamp { intensity: 2.0, on: true });

// A CPU-side edit; queued, not yet uploaded.
lamp.modify(|lamp| lamp.intensity *= 2.0);

// Upload the slab to a GPU buffer.
let buffer = arena.commit();

// Analyze the shader's linkage and build its wgpu pipeline objects,
// binding the arena's buffer by the storage declaration's name.
let mut linkage = wgpu_linkage::analyze_wgsl_module(&lamp_shader::WGSL_SOURCE).unwrap();
let module = linkage.shader_module(&runtime.device);
let pipeline = runtime.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
    label: Some("lamp"),
    layout: Some(&linkage.pipeline_layout(&runtime.device, Some("lamp"))),
    module: &module,
    entry_point: Some("main"),
    compilation_options: Default::default(),
    cache: None,
});
let bind_group = linkage
    .create_bind_group_named(0, &runtime.device, &[("SLAB", buffer.as_entire_binding())])
    .unwrap();
let mut encoder = runtime.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
    label: Some("lamp"),
});
let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
    label: Some("lamp"),
    timestamp_writes: None,
});
pass.set_pipeline(&pipeline);
pass.set_bind_group(0, &bind_group, &[]);
pass.dispatch_workgroups(1, 1, 1);
drop(pass);
runtime.queue.submit(Some(encoder.finish()));
runtime.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();

// The GPU read the lamp, bumped its intensity by one, and wrote it
// back. Pull GPU changes into the CPU caches...
futures_lite::future::block_on(arena.synchronize()).unwrap();
// ...and the RAII `Value` reflects them.
assert_eq!(
    lamp_shader::Lamp { intensity: 5.0, on: true },
    lamp.get()
);
```

## Features

* `wgpu` (**on** by default) - provides [`WgpuRuntime`], an [`IsRuntime`]
  implementation that drives `wgpu::Buffer`s. Disable it to bring your own
  runtime.