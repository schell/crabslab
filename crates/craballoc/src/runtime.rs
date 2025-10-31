//! The CPU side fo slab allocation.

use std::{
    future::Future,
    ops::{Deref, DerefMut},
    sync::{Arc, Mutex, RwLock},
};

use crabslab::Array;
use snafu::ResultExt;
use tracing::Instrument;

use crate::slab::{AsyncRecvSnafu, AsyncSnafu, PollSnafu, SlabAllocatorError};

/// An update to a slab.
///
/// This is a write that can be serialized for later syncronization.
#[derive(Clone, Debug)]
pub struct SlabUpdate {
    pub array: Array<u32>,
    pub elements: Vec<u32>,
}

impl SlabUpdate {
    pub fn intersects(&self, other: &Self) -> bool {
        let here_start = self.array.index;
        let there_start = other.array.index;
        let here_end = self.array.index + self.array.len;
        let there_end = other.array.index + other.array.len;
        !(here_start >= there_end || there_start >= here_end)
    }
}

/// Represents the runtime that provides the interface to the GPU buffer.
///
/// For example, this could be a struct that contains `wgpu::Device` and `wgpu::Queue`,
/// or it could be a struct that contains Vulkan types, etc.
pub trait IsRuntime: Clone {
    /// The type of buffer this runtime engages with.
    type Buffer;

    /// The type used to denote the configuration of the buffer.
    type BufferUsages: Clone;

    /// Create a new buffer with the given `capacity`, where `capacity` is the number of `u32`s
    /// that can be stored in the buffer.
    fn buffer_create(
        &self,
        capacity: usize,
        label: Option<&str>,
        usages: Self::BufferUsages,
    ) -> Self::Buffer;

    /// Copy the contents of one buffer into another at index 0.
    fn buffer_copy(
        &self,
        source_buffer: &Self::Buffer,
        destination_buffer: &Self::Buffer,
        label: Option<&str>,
    );

    /// Write the updates into the given buffer.
    fn buffer_write<U: Iterator<Item = SlabUpdate>>(&self, updates: U, buffer: &Self::Buffer);

    /// Read the range from the given buffer.
    ///
    /// ## Note
    /// This function is async.
    fn buffer_read(
        &self,
        buffer: &Self::Buffer,
        buffer_len: usize,
        range: impl std::ops::RangeBounds<usize>,
    ) -> impl Future<Output = Result<Vec<u32>, SlabAllocatorError>>;
}

pub(crate) fn range_to_indices_and_len(
    // Used in the case the range is unbounded
    max_len: usize,
    range: impl std::ops::RangeBounds<usize>,
) -> (usize, usize, usize) {
    let start = match range.start_bound() {
        core::ops::Bound::Included(start) => *start,
        core::ops::Bound::Excluded(start) => *start + 1,
        core::ops::Bound::Unbounded => 0,
    };
    let end = match range.end_bound() {
        core::ops::Bound::Included(end) => *end + 1,
        core::ops::Bound::Excluded(end) => *end,
        core::ops::Bound::Unbounded => max_len,
    };
    let len = end - start;
    (start, end, len)
}

/// A runtime that only operates on the CPU.
///
/// `CpuRuntime` manages [`VecSlab`]s, which are used as a reference
/// implementation, mostly for testing.
#[derive(Clone, Copy)]
pub struct CpuRuntime;

impl AsRef<CpuRuntime> for CpuRuntime {
    fn as_ref(&self) -> &CpuRuntime {
        self
    }
}

/// A slab buffer used _only_ on the CPU.
pub struct VecSlab {
    inner: RwLock<Vec<u32>>,
}

impl VecSlab {
    pub fn as_vec(&self) -> impl Deref<Target = Vec<u32>> + '_ {
        self.inner.read().unwrap()
    }

    pub fn as_mut_vec(&self) -> impl DerefMut<Target = Vec<u32>> + '_ {
        self.inner.write().unwrap()
    }
}

impl IsRuntime for CpuRuntime {
    type Buffer = VecSlab;
    type BufferUsages = ();

    fn buffer_create(&self, capacity: usize, label: Option<&str>, _usages: ()) -> VecSlab {
        log::trace!(
            "creating vec buffer '{}' with capacity {capacity}",
            label.unwrap_or("unknown")
        );
        VecSlab {
            inner: RwLock::new(vec![0; capacity]),
        }
    }

    fn buffer_copy(
        &self,
        source_buffer: &VecSlab,
        destination_buffer: &VecSlab,
        label: Option<&str>,
    ) {
        log::trace!("performing copy '{}'", label.unwrap_or("unknown"));
        let this = &destination_buffer;
        let source = source_buffer.inner.read().unwrap();
        let mut destination = this.inner.write().unwrap();
        let destination_slice = &mut destination[0..source.len()];
        destination_slice.copy_from_slice(source.as_slice());
    }

    fn buffer_write<U: Iterator<Item = SlabUpdate>>(&self, updates: U, buffer: &Self::Buffer) {
        let mut guard = buffer.inner.write().unwrap();
        log::trace!("writing to vec len:{}", guard.len());
        for SlabUpdate { array, elements } in updates {
            log::trace!("array: {array:?} elements: {elements:?}");
            let slice = &mut guard[array.starting_index()..array.starting_index() + array.len()];
            slice.copy_from_slice(&elements);
        }
    }

    async fn buffer_read(
        &self,
        buffer: &Self::Buffer,
        buffer_len: usize,
        range: impl std::ops::RangeBounds<usize>,
    ) -> Result<Vec<u32>, SlabAllocatorError> {
        let v = buffer.inner.read().unwrap();
        debug_assert_eq!(v.len(), buffer_len);
        let (start, end, len) = range_to_indices_and_len(v.len(), range);
        let mut output = vec![0; len];
        let slice = &v[start..end];
        output.copy_from_slice(slice);
        Ok(output)
    }
}

#[cfg(feature = "wgpu")]
/// A slab allocation runtime that creates and updates [`wgpu::Buffer`]s.
#[derive(Clone)]
pub struct WgpuRuntime {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
}

impl AsRef<WgpuRuntime> for WgpuRuntime {
    fn as_ref(&self) -> &WgpuRuntime {
        self
    }
}

#[cfg(feature = "wgpu")]
impl IsRuntime for WgpuRuntime {
    type Buffer = wgpu::Buffer;
    type BufferUsages = wgpu::BufferUsages;

    fn buffer_write<U: Iterator<Item = SlabUpdate>>(&self, updates: U, buffer: &Self::Buffer) {
        for SlabUpdate { array, elements } in updates {
            let offset = array.starting_index() as u64 * std::mem::size_of::<u32>() as u64;
            self.queue
                .write_buffer(buffer, offset, bytemuck::cast_slice(&elements));
        }
        self.queue.submit(std::iter::empty());
    }

    fn buffer_create(
        &self,
        capacity: usize,
        label: Option<&str>,
        usages: wgpu::BufferUsages,
    ) -> Self::Buffer {
        let size = (capacity * std::mem::size_of::<u32>()) as u64;
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label,
            size,
            usage: usages
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    fn buffer_copy(
        &self,
        source_buffer: &Self::Buffer,
        destination_buffer: &Self::Buffer,
        label: Option<&str>,
    ) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label });
        encoder.copy_buffer_to_buffer(
            source_buffer,
            0,
            destination_buffer,
            0,
            source_buffer.size(),
        );
        self.queue.submit(std::iter::once(encoder.finish()));
    }

    #[tracing::instrument(skip_all)]
    async fn buffer_read(
        &self,
        buffer: &Self::Buffer,
        buffer_len: usize,
        range: impl std::ops::RangeBounds<usize>,
    ) -> Result<Vec<u32>, SlabAllocatorError> {
        let (start, _end, len) = crate::runtime::range_to_indices_and_len(buffer_len, range);
        let byte_offset = start * std::mem::size_of::<u32>();
        let length = len * std::mem::size_of::<u32>();
        let output_buffer_size = length as u64;
        let output_buffer = tracing::trace_span!("create-buffer").in_scope(|| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: output_buffer_size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            })
        });

        let submission_index = tracing::trace_span!("copy_buffer").in_scope(|| {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            log::trace!(
                "copy_buffer_to_buffer byte_offset:{byte_offset}, \
             output_buffer_size:{output_buffer_size}",
            );
            encoder.copy_buffer_to_buffer(
                buffer,
                byte_offset as u64,
                &output_buffer,
                0,
                output_buffer_size,
            );
            self.queue.submit(std::iter::once(encoder.finish()))
        });

        let buffer_slice = output_buffer.slice(..);
        let (tx, rx) = async_channel::bounded(1);
        tracing::trace_span!("map_async").in_scope(|| {
            buffer_slice.map_async(wgpu::MapMode::Read, move |res| tx.try_send(res).unwrap());
        });
        tracing::trace_span!("poll").in_scope(|| {
            self.device
                .poll(wgpu::PollType::WaitForSubmissionIndex(submission_index))
                .context(PollSnafu)
        })?;
        rx.recv()
            .instrument(tracing::info_span!("recv"))
            .await
            .context(AsyncRecvSnafu)?
            .context(AsyncSnafu)?;
        let output = tracing::trace_span!("get_mapped").in_scope(|| {
            let bytes = buffer_slice.get_mapped_range();
            bytemuck::cast_slice(bytes.deref()).to_vec()
        });
        Ok(output)
    }
}
