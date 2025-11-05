//! An arena allocator built on a `u32` slab.
//!
//! The [`Arena`] type provided by this module is the successor to [`SlabAllocator`].
//! Much like [`SlabAllocator`], [`Arena`] provides an API to dynamically allocate types that
//! implement [`SlabItem`] from the CPU, and then synchronize those changes to the GPU.
//! Unlike [`SlabAllocator`], [`Arena`] also provides synchronization _back_ from the GPU.
//! Only values that are explicitly set to sync will be synchronized.
use std::{
    borrow::Cow,
    marker::PhantomData,
    num::NonZeroU32,
    sync::{Arc, RwLock},
};

use crabslab::{Id, Slab, SlabItem};

#[cfg(doc)]
use crate::slab::SlabAllocator;

use crate::{
    buffer::{manager::BumpAllocator, SlabBuffer},
    range::{Range, RangeManager},
    runtime::IsRuntime,
    update::{CpuUpdateSource, SourceId, UpdateManager},
};

/// Value is synchronized on both CPU and GPU.
pub struct Bidirectional;

// Value is synchronized from CPU to the GPU, but any changes on the GPU will
// leave the value out of sync.
pub struct OneWayFromCpu;

// Value is synchronized from GPU to CPU but cannot be changed from the CPU.
pub struct OneWayFromGpu;

pub struct Value<T: SlabItem, Sync = Bidirectional> {
    update_source: CpuUpdateSource,
    /// A channell to send updates into.
    _phantom: PhantomData<(T, Sync)>,
}

/// An arena allocator backed by a `u32` slab.
///
/// TODO: write about how values allocated from the arena can conditionally be
/// synchronized from the CPU to the GPU and back.
#[derive(Debug)]
pub struct Arena<R: IsRuntime> {
    /// Manages bump allocation on the slab.
    buffer_manager: BumpAllocator<R>,
    /// Manages CPU side updates to the slab after allocation.
    update_manager: UpdateManager,
}

impl<Runtime: IsRuntime> Clone for Arena<Runtime> {
    fn clone(&self) -> Self {
        Self {
            buffer_manager: self.buffer_manager.clone(),
            update_manager: self.update_manager.clone(),
        }
    }
}

impl<R: IsRuntime> Arena<R> {
    /// Create a new arena.
    pub fn new(
        runtime: impl AsRef<R>,
        label: impl Into<Cow<'static, str>>,
        default_buffer_usages: R::BufferUsages,
    ) -> Self {
        Self {
            buffer_manager: BumpAllocator::new(runtime, label, default_buffer_usages),
            update_manager: UpdateManager::default(),
        }
    }

    /// Allocate a new value from the arena.
    ///
    /// This value has bidirectional synchonization.
    pub fn new_value<T: SlabItem>(&self, value: T) -> Value<T> {
        if let Some(spaces) = NonZeroU32::new(T::SLAB_SIZE as u32) {
            let range = self.buffer_manager.alloc(spaces);
            let update_source = self.update_manager.new_update_source(SourceId {
                range,
                type_is: std::any::type_name::<T>(),
            });
            update_source.modify(|data| {
                data.write(Id::ZERO, &value);
            });
            Value {
                update_source,
                _phantom: PhantomData,
            }
        } else {
            Value {
                update_source: CpuUpdateSource::null(),
                _phantom: PhantomData,
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn sanity() {
        let arena = Arena::new(crate::wgpu_runtime(), "tests", wgpu::BufferUsages::empty());
        let u0 = arena.new_value(0u32);
        let u1 = arena.new_value(0u32);
        let u2 = arena.new_value(0u32);
        let u3 = arena.new_value(0u32);
    }
}
