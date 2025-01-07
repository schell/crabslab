<div style="float: right; padding: 1em;">
    <img src="https://github.com/schell/crabslab/blob/main/crates/crabslab/crabslab.png?raw=true" alt="slabcraft for crabs" width="256" />
</div>

## What

`craballoc` is a slab allocator built on top of [`crabslab`](github.com/schell/crabslab).

## But Why?

Opinion: working with _shaders_ is much easier using a slab.

Shader code can be written in Rust with [`rust-gpu`](https://github.com/rust-gpu/rust-gpu),
which will enable you to use your Rust types in CPU and GPU code, and this library helps 
marshal your types to the GPU.

Using a slab makes it pretty easy to marshal data to the GPU, and `craballoc` does the heavy
lifting, with (almost) automatic synchronization and RAII.

## And How

The idea is simple - `craballoc` provides [`SlabAllocator<T>`], where `T` is the runtime of your 
choice. Enabling the feature `wgpu` (**on** by default) provides a [`WgpuRuntime`] to fill that `T`.

Your local types derive the trait [`SlabItem`] which allows them to be written to and read
from to the slab.

You use [`SlabAllocator::new_value`] or [`SlabAllocator::new_array`] on the CPU to allocate 
new values and arrays on the slab, receiving a [`Hybrid<T>`] or [`HybridArray<T>`], respectively.

To modify values, use [`Hybrid::modify`].

To forget the CPU side of values, use [`Hybrid::into_gpu_only`].

Finally, synchronize the slab once per frame (or more, or less) using [`SlabAllocator::get_updated_buffer`].

On the GPU (using a shader written with [`spirv-std`](https://crates.io/crates/spirv-std)) use 
`crabslab` to read values in a no_std context. 
See the [crabslab docs](https://docs.rs/crabslab/latest/crabslab/) for more info.
