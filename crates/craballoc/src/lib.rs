//! Crafting a tasty slab.
//!
//! This crate provides [`Arena`] allocation backed by a `u32` slab.
#![doc = include_str!("../README.md")]
#![warn(missing_docs)]

use snafu::prelude::*;

pub mod arena;
pub mod buffer;
pub mod range;
pub mod runtime;
mod update;

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
#[cfg(doc)]
use runtime::IsRuntime;

/// Errors that can occur while synchronizing slab data.
#[derive(Debug, Snafu)]
#[snafu(visibility(pub(crate)))]
pub enum Error {
    /// The slab has no internal buffer. Please call [`Arena::commit`] first.
    #[snafu(display("Slab has no internal buffer. Please call Arena::commit first"))]
    NoInternalBuffer,

    /// A `Value`'s sync direction cannot be changed while other handles
    /// to it exist.
    #[snafu(display("cannot change the sync direction of a Value while clones exist"))]
    SyncChangeWithClones,

    /// An async channel receive error.
    #[snafu(display("Async recv error: {source}"))]
    AsyncRecv {
        /// The underlying receive error.
        source: async_channel::RecvError,
    },

    #[cfg(feature = "wgpu")]
    /// An async buffer mapping error.
    #[snafu(display("Async error: {source}"))]
    Async {
        /// The underlying wgpu error.
        source: wgpu::BufferAsyncError,
    },

    #[cfg(feature = "wgpu")]
    /// A device poll error.
    #[snafu(display("Poll error: {source}"))]
    Poll {
        /// The underlying wgpu error.
        source: wgpu::PollError,
    },

    /// Another error.
    #[snafu(display("{source}"))]
    Other {
        /// The underlying error.
        source: Box<dyn std::error::Error>,
    },
}

#[cfg(all(test, feature = "wgpu"))]
fn wgpu_runtime() -> crate::runtime::WgpuRuntime {
    let backends = wgpu::Backends::PRIMARY;
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
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
