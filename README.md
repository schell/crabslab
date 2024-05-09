<div style="float: right; padding: 1em;">
    <img src="https://github.com/schell/crabslab/blob/main/crates/crabslab/crabslab.png?raw=true" alt="slabcraft for crabs" width="256" />
</div>

## What

`crabslab` is a slab implementation focused on marshalling data between CPUs and GPUs.

[See the example below](#example).

## But Why?
It's hard to get data onto GPUs in the form you expect.

To marshall your data correctly you must know about the alignment and sizes of the underlying representation of your data.
This will often surprise you!

Working with a slab on the other hand, only requires that your types can be written into an array and read from an array.

### Opinion
Working with _shaders_ is much easier using a slab.

Shader code can be written in Rust with [`rust-gpu`](https://github.com/EmbarkStudios/rust-gpu),
which will enable you to use this crate on both CPU and GPU code.

### rust-gpu
This crate was made to work with [`rust-gpu`](https://github.com/EmbarkStudios/rust-gpu/).
Specifically, with this crate it is possible to pack your types into a buffer on the CPU
and then read your types from the slab on the GPU (in Rust).

### Other no-std platforms
Even though this crate was written with `rust-gpu` in mind, it should work in other `no-std`
contexts.

## And How
The idea is simple - `crabslab` helps you manage a heap of contiguous `u32`s (roughly in the form of `Vec<u32>`). 
Types implement the trait `SlabItem` which writes the type into an index of the slab as contiguous `u32`s and also 
reads them out symmetrically. 

`crabslab` includes:
* a few traits:
  - `Slab`
  - `GrowableSlab`
  - `SlabItem`
* a derive macro for `SlabItem` for your types
* a few new structs for working with slabs
  - `Id`
  - `Array`
  - `Offset`
* a helper struct `CpuSlab` which wraps anything implementing `GrowableSlab`
* a feature-gated helper for using slabs with `wgpu` - `WgpuBuffer`
  - [example](https://github.com/schell/crabslab/blob/main/crates/crabslab/src/wgpu_slab.rs#L356)
* feature for deriving `SlabItem` for `glam` types

# Example
```rust
use crabslab::{CpuSlab, Slab, GrowableSlab, SlabItem, Id};
use glam::{Vec3, Vec4};

#[derive(Debug, Default, SlabItem, PartialEq)]
struct Light {
    direction: Vec3,
    color: Vec4,
    inner_cutoff: f32,
    outer_cutoff: f32,
    is_on: bool
}

impl Light {
    fn standard() -> Self {
        Light {
            direction: Vec3::NEG_Z, // pointing down
            color: Vec4::ONE, // white
            inner_cutoff: 0.5,
            outer_cutoff: 2.6,
            is_on: true
        }
    }
}

fn cpu_code() -> (Id<Light>, Vec<u32>) {
    let light = Light::standard();
    // Create a new slab on the CPU-side.
    // Using CpuSlab make `append` unambiguous, as `Vec` has its own `append` function.
    let mut slab = CpuSlab::new(vec![]);
    let id = slab.append(&light);
    (id, slab.into_inner())
}

fn shader_code(light_id: Id<Light>, slab: &[u32]) {
    let light = slab.read(light_id);
    assert_eq!(Light::standard(), light);
}

let (light_id, slab) = cpu_code();
// marshalling your data depends on which GPU library you are using...
shader_code(light_id, &slab);
```
