#![allow(unexpected_cfgs)]
#![no_std]
use crabslab::{Array, Id, Slab, SlabItem};
use glam::UVec3;

#[derive(Clone, Copy, Debug, Default, PartialEq, SlabItem)]
pub struct Data {
    pub i: u32,
    pub float: f32,
    pub int: i64,
}

#[derive(Debug, Clone, Copy, SlabItem)]
pub enum DataChange {
    I(u32),
    Float(f32),
    Int(i64),
}

impl Default for DataChange {
    fn default() -> Self {
        DataChange::I(0)
    }
}

impl DataChange {
    pub fn apply(&self, data: &mut Data) {
        match self {
            DataChange::I(i) => data.i = *i,
            DataChange::Float(f) => data.float = *f,
            DataChange::Int(i) => data.int = *i,
        }
    }
}

#[derive(Debug, Clone, Copy, SlabItem)]
pub struct ArrayChange {
    pub i: u32,
    pub change: DataChange,
}

impl ArrayChange {
    pub fn apply(&self, data: &mut [Data]) {
        let data = &mut data[self.i as usize];
        self.change.apply(data);
    }
}

#[derive(Default, Debug, Clone, Copy, SlabItem)]
pub enum AnyChangeId {
    #[default]
    None,
    Data {
        change_id: Id<DataChange>,
        data_id: Id<Data>,
    },
    Array {
        change_id: Id<ArrayChange>,
        data_array: Array<Data>,
    },
}

/// Helper for applying data changes during the GPU updates proptest.
#[derive(Default, Debug, Clone, Copy, SlabItem)]
pub struct ApplyDataChangeInvocation {
    changes_ids: Array<AnyChangeId>,
}

impl ApplyDataChangeInvocation {
    pub const WORKGROUP_SIZE: u32 = 16;
    pub const WORKGROUP_VOLUME: u32 =
        Self::WORKGROUP_SIZE * Self::WORKGROUP_SIZE * Self::WORKGROUP_SIZE;

    fn any_change_id(&self, global_id: UVec3, changes_slab: &[u32]) -> AnyChangeId {
        let index = global_id.z * Self::WORKGROUP_SIZE * Self::WORKGROUP_SIZE
            + global_id.y * Self::WORKGROUP_SIZE
            + global_id.x;
        let id_id = self.changes_ids.at(index as usize);
        changes_slab.read_unchecked(id_id)
    }

    /// Returns the total number of invocations required.
    fn total_invocations_required(&self) -> u32 {
        self.changes_ids.len
    }

    /// Returns the workgroup dimensions required to invoke this shader.
    pub fn workgroup_dimensions(&self) -> UVec3 {
        let count = self.total_invocations_required();
        let volume = Self::WORKGROUP_VOLUME;
        let mut dims = UVec3::ZERO;

        while dims.x * dims.y * dims.z * volume < count {
            if dims.x <= dims.y && dims.x <= dims.z {
                dims.x += 1;
            } else if dims.y <= dims.x && dims.y <= dims.z {
                dims.y += 1;
            } else {
                dims.z += 1;
            }
        }

        dims
    }

    pub fn run(&self, data_slab: &mut [u32], changes_slab: &[u32], global_id: glam::UVec3) {
        let any_change_id = self.any_change_id(global_id, changes_slab);
        match any_change_id {
            AnyChangeId::None => {}
            AnyChangeId::Data { change_id, data_id } => {
                let change = changes_slab.read_unchecked(change_id);
                let mut data = data_slab.read_unchecked(data_id);
                change.apply(&mut data);
                data_slab.write(data_id, &data);
            }
            AnyChangeId::Array {
                change_id,
                data_array,
            } => {
                let change = changes_slab.read_unchecked(change_id);
                let data_id = data_array.at(change.i as usize);
                let mut data = data_slab.read_unchecked(data_id);
                change.change.apply(&mut data);
                data_slab.write(data_id, &data);
            }
        }
    }
}
