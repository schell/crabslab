//! Slab allocation of WebGPU buffers.
use crabslab::{Array, Slab, SlabItem};
use snafu::OptionExt;

use crate::{
    runtime::{IsRuntime, WgpuRuntime},
    slab::NoInternalBufferSnafu,
};

use super::{SlabAllocator, SlabAllocatorError};

impl SlabAllocator<WgpuRuntime> {
    /// Read the slab range from the GPU.
    #[tracing::instrument(skip_all)]
    pub async fn read(
        &self,
        range: impl std::ops::RangeBounds<usize>,
    ) -> Result<Vec<u32>, SlabAllocatorError> {
        let internal_buffer = self.get_buffer().context(NoInternalBufferSnafu)?;
        self.runtime
            .buffer_read(&internal_buffer, self.len(), range)
            .await
    }

    /// Read an array of typed values from the GPU.
    #[tracing::instrument(skip_all)]
    pub async fn read_array<T: SlabItem + Default>(
        &self,
        array: Array<T>,
    ) -> Result<Vec<T>, SlabAllocatorError> {
        let arr = array.into_u32_array();
        let range = array.index as usize..(arr.index + arr.len) as usize;
        let data = self.read(range).await?;
        let t_array = Array::new(0, array.len() as u32);
        Ok(data.read_vec(t_array))
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.runtime.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.runtime.queue
    }
}
