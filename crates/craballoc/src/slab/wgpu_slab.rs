//! Slab allocation of WebGPU buffers.
use crabslab::{Array, Id, Slab, SlabItem};
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
        self.runtime()
            .buffer_read(&internal_buffer, self.len() as usize, range)
            .await
    }

    /// Read on value from the GPU.
    #[tracing::instrument(skip_all)]
    pub async fn read_one<T: SlabItem + Default>(
        &self,
        id: Id<T>,
    ) -> Result<T, SlabAllocatorError> {
        let index = id.index();
        let range = index..(index + T::SLAB_SIZE);
        let data = self.read(range).await?;
        Ok(data.read_unchecked(Id::<T>::new(0)))
    }

    /// Read an array of typed values from the GPU.
    #[tracing::instrument(skip_all)]
    pub async fn read_array<T: SlabItem + Default>(
        &self,
        array: Array<T>,
    ) -> Result<Vec<T>, SlabAllocatorError> {
        let arr = array.into_u32_array();
        let range = array.id.index()..arr.id.index() + arr.len();
        let data = self.read(range).await?;
        let t_array = Array::new(Id::ZERO, array.len() as u32);
        Ok(data.read_vec(t_array))
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.runtime().device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.runtime().queue
    }
}

#[cfg(test)]
mod test {
    use crabslab::{Array, Id};

    use crate::prelude::*;

    #[test]
    fn roundtrips() {
        let slab = SlabAllocator::new(crate::wgpu_runtime(), "tests", wgpu::BufferUsages::empty());
        let _a = slab.new_value(0u32);
        let _b = slab.new_value(1u32);
        let _c = slab.new_value(2u32);
        slab.commit();

        let vals = futures_lite::future::block_on(slab.read_array(Array::<u32>::new(Id::ZERO, 3)))
            .unwrap();
        assert_eq!(&[0, 1, 2], vals.as_slice());
    }
}
