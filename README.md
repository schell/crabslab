<div style="float: right; padding: 1em;">
    <img src="https://github.com/schell/crabslab/blob/main/crates/crabslab/crabslab.png?raw=true" alt="slabcraft for crabs" width="256" />
</div>

## What

`crabslab` is a slab implementation focused on marshalling data between CPUs and GPUs.

[See the examples below](#examples).

This project is funded through [NGI Zero Commons](https://nlnet.nl/commonsfund/), a fund established by [NLnet](https://nlnet.nl) 
with financial support from the European Commission's [Next Generation Internet](https://ngi.eu) program. 
Learn more at the [2025 NLnet project page](https://nlnet.nl/project/Renderling-Ecosystem/).

[<img src="https://nlnet.nl/logo/banner.png" alt="NLnet foundation logo" width="20%" />](https://nlnet.nl)

[<img src="https://nlnet.nl/image/logos/NGI0_tag.svg" alt="NGI Zero Logo" width="20%" />](https://nlnet.nl/core)

## But Why?
It's hard to get data onto GPUs in the form you expect.

To marshall your data correctly you must know about the alignment and sizes of the underlying representation of your data.
This will often surprise you!

Working with a slab on the other hand, only requires that your types can be written into an array and read from an array.

### Opinion
Working with _shaders_ is much easier using a slab.

Shader code can be written in Rust with [`wgsl-rs`](https://github.com/schell/wgsl-rs),
which lets you use your Rust types in both CPU and GPU code. A single
`#[derive(SlabItem)]` covers both worlds: the derive generates the CPU-side
serialization, and the [`SlabItemExt`] extension generates the same serialization
for `wgsl-rs` shader modules on the GPU.

### Other no-std platforms
Even though this crate was written with `wgsl-rs` in mind, it should work in other `no-std`
contexts.

## And How
The idea is simple - `crabslab` helps you manage a heap of contiguous `u32`s (roughly in the form of `Vec<u32>`). 
Types implement the trait [`SlabItem`] which writes the type into an index of the slab as contiguous `u32`s and also 
reads them out symmetrically.

`crabslab` includes:
* a few traits:
  - [`Slab`]
  - [`GrowableSlab`]
  - [`SlabItem`]
* a derive macro for [`SlabItem`] for your types
* a few new structs for working with slabs
  - [`Id`]
  - [`Array`]
  - [`Offset`](offset::Offset)
* a helper struct [`CpuSlab`] which wraps anything implementing [`GrowableSlab`]
* the [`slab_item`] module, which holds the [`SlabItem`] trait definition, the
  WGSL-expressible primitive impls, and the dual-world [`slab_read!`] /
  [`slab_write!`] macros
* [`SlabItemExt`], a `wgsl-rs` extension that generates the GPU-side serialization
  from the same `#[derive(SlabItem)]`
* the [`bits`] module, with utilities for tightly packing fields in manual
  [`SlabItem`] implementations
* a feature for deriving [`SlabItem`] for `glam` types

For a slab allocator with RAII values that keep themselves synchronized with the
GPU, see the [`craballoc`](https://github.com/schell/crabslab/tree/main/crates/craballoc)
crate, which is built on this one.

## Features
* `glam` (**on** by default) - implements [`SlabItem`] for [`glam`](https://docs.rs/glam) types
* `futures-lite` (**on** by default) - enables the `futures-lite` dependency

## Examples

### CPU

```rust
use crabslab::{CpuSlab, Slab, GrowableSlab, SlabItem, Id};
use glam::{Vec3, Vec4};

#[derive(Clone, Copy, Debug, Default, SlabItem, PartialEq)]
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

### One definition, two worlds

Define your type once and derive `SlabItem` once. The CPU gets its serialization
from the derive; inside a `wgsl-rs` module, [`SlabItemExt`] generates the same
serialization for the GPU. The `#[wgsl]` module is plain Rust too, so this
example runs on the CPU as-is:

```rust
use crabslab::{slab_read, slab_write, SlabItem};
use wgsl_rs::wgsl;

#[wgsl(extensions = [crabslab::SlabItemExt])]
mod lamp_shader {
    use crabslab::slab_item::*;
    use wgsl_rs::std::*;

    /// A lamp in the scene, shared between CPU and GPU code.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Wgsl, SlabItem)]
    pub struct Lamp {
        pub intensity: f32,
        pub on: bool,
    }

    storage!(group(0), binding(0), read_write, SLAB: RuntimeArray<u32>);

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(local_invocation_index)] _i: u32) {
        let mut lamp: Lamp;
        slab_read!(Lamp, get!(SLAB), 0u32, lamp);
        lamp.intensity += 1.0;
        slab_write!(Lamp, get_mut!(SLAB), 0u32, lamp);
    }
}

// The `#[wgsl]` module is plain Rust, so the very same serialization
// round-trips on the CPU, no GPU required:
let lamp = lamp_shader::Lamp { intensity: 2.0, on: true };
let mut slab = [0u32; lamp_shader::Lamp::SLAB_SIZE];
slab_write!(lamp_shader::Lamp, slab, 0, lamp);
let round_tripped: lamp_shader::Lamp;
slab_read!(lamp_shader::Lamp, slab, 0, round_tripped);
assert_eq!(lamp, round_tripped);

// `WGSL_SOURCE` is the assembled shader source, complete with the
// `SlabItemExt`-generated GPU-side serialization for `Lamp`:
let source = lamp_shader::WGSL_SOURCE.wgsl_source().unwrap();
assert!(source.contains("fn Lamp__1to_array("));
```

## 🫶 Sponsor this!

This work will always be free and open source. 
If you use it (outright or for inspiration), please consider donating.

[💰 Sponsor 💝](https://github.com/sponsors/schell)
