//! Crafting a tasty slab.
//!
//! This crate provides [`Arena`] allocation backed by a `u32` slab.
#![doc = include_str!("../README.md")]

use snafu::prelude::*;

pub mod arena;
mod buffer;
pub mod range;
pub mod runtime;
// pub mod slab;
mod update;
// pub mod value;

pub mod prelude {
    //! Easy-include prelude module.
    pub extern crate crabslab;
    pub use super::arena::{Arena, Value};
    pub use super::runtime::CpuRuntime;
    #[cfg(feature = "wgpu")]
    pub use super::runtime::WgpuRuntime;
    pub use crabslab::{Array, Id};
}

#[cfg(doc)]
use prelude::crabslab::SlabItem;
#[cfg(doc)]
use prelude::*;

#[derive(Debug, Snafu)]
#[snafu(visibility(pub(crate)))]
pub enum Error {
    #[snafu(display("Slab has no internal buffer. Please call Arena::commit first"))]
    NoInternalBuffer,

    #[snafu(display("Async recv error: {source}"))]
    AsyncRecv { source: async_channel::RecvError },

    #[cfg(feature = "wgpu")]
    #[snafu(display("Async error: {source}"))]
    Async { source: wgpu::BufferAsyncError },

    #[cfg(feature = "wgpu")]
    #[snafu(display("Poll error: {source}"))]
    Poll { source: wgpu::PollError },

    #[snafu(display("{source}"))]
    Other { source: Box<dyn std::error::Error> },
}

#[cfg(all(test, feature = "wgpu"))]
fn wgpu_runtime() -> crate::runtime::WgpuRuntime {
    let backends = wgpu::Backends::PRIMARY;
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends,
        ..Default::default()
    });
    let adapter =
        futures_lite::future::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .unwrap();
    let (device, queue) =
        futures_lite::future::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
            .unwrap();
    crate::runtime::WgpuRuntime {
        device: device.into(),
        queue: queue.into(),
    }
}

#[cfg(test)]
mod test;
