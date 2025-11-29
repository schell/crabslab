#![allow(unexpected_cfgs)]
#![cfg_attr(target_arch = "spirv", no_std)]
use crabslab::{Array, Id, Slab, SlabItem};
use glam::UVec3;

#[macro_export]
/// A wrapper around `std::println` that is a noop on the GPU.
macro_rules! println {
    ($($arg:tt)*) => {
        #[cfg(not(target_arch = "spirv"))]
        {
            std::println!($($arg)*);
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, SlabItem)]
pub struct Data {
    pub i: u32,
    pub float: f32,
    pub ints: (u32, u32),
}

#[derive(Default, Debug, Clone, Copy, SlabItem)]
#[repr(u32)]
pub enum DataChangeTy {
    #[default]
    I,
    Float,
    Ints,
}

#[derive(Default, Debug, Clone, Copy, SlabItem)]
pub struct DataChange {
    pub ty: DataChangeTy,
    pub data: [u32; 3],
}

#[cfg(not(target_arch = "spirv"))]
impl core::fmt::Display for DataChange {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let (field, value) = match self.ty {
            DataChangeTy::I => ("i", self.data.read_unchecked(Id::<u32>::ZERO).to_string()),
            DataChangeTy::Float => (
                "float",
                self.data.read_unchecked(Id::<f32>::ZERO).to_string(),
            ),
            DataChangeTy::Ints => (
                "ints",
                format!("{:?}", self.data.read_unchecked(Id::<(u32, u32)>::ZERO)),
            ),
        };
        f.write_str(&format!("change {field} to {value}"))
    }
}

impl DataChange {
    fn new(ty: DataChangeTy, value: impl SlabItem) -> Self {
        let mut s = Self {
            ty,
            ..Default::default()
        };
        s.data.write(Id::ZERO, &value);
        s
    }
    pub fn i(i: u32) -> Self {
        Self::new(DataChangeTy::I, i)
    }

    pub fn float(f: f32) -> Self {
        Self::new(DataChangeTy::Float, f)
    }

    pub fn ints(i: u32, j: u32) -> Self {
        Self::new(DataChangeTy::Ints, (i, j))
    }

    pub fn apply(&self, data: &mut Data) {
        match self.ty {
            DataChangeTy::I => {
                data.i = self.data.read_unchecked(Id::<u32>::ZERO);
            }
            DataChangeTy::Float => {
                data.float = self.data.read_unchecked(Id::ZERO);
            }
            DataChangeTy::Ints => {
                data.ints = self.data.read_unchecked(Id::ZERO);
            }
        }
    }
}

#[derive(Default, Debug, Clone, Copy, SlabItem)]
pub struct ArrayChange {
    pub i: u32,
    pub change: DataChange,
}

#[cfg(not(target_arch = "spirv"))]
impl core::fmt::Display for ArrayChange {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&format!("{} of data at index {}", self.change, self.i))
    }
}

impl From<DataChange> for ArrayChange {
    fn from(change: DataChange) -> Self {
        ArrayChange { i: 0, change }
    }
}

impl ArrayChange {
    pub fn apply(&self, data: &mut [Data]) {
        let data = &mut data[self.i as usize];
        self.change.apply(data);
    }
}

#[derive(Debug, Clone, Copy, SlabItem)]
pub struct AnyChangeId {
    pub change_id: Id<ArrayChange>,
    pub data_array: Array<Data>,
}

impl Default for AnyChangeId {
    fn default() -> Self {
        AnyChangeId {
            change_id: Id::NONE,
            data_array: Array::NONE,
        }
    }
}

/// Helper for applying data changes during the GPU updates proptest.
#[derive(Default, Debug, Clone, Copy, SlabItem)]
pub struct ApplyDataChangeInvocation {
    pub changes_ids: Array<AnyChangeId>,
    // Written atomically from the shader side to keep track of how many
    // invocations have successfully run.
    //
    // This points to a `u32` hosted on the **`data_slab`**.
    pub invocations_id: Id<u32>,
    pub invocations_skipped_id: Id<u32>,
}

impl ApplyDataChangeInvocation {
    pub const WORKGROUP_SIZE: UVec3 = UVec3::new(16, 1, 1);
    pub const WORKGROUP_VOLUME: u32 =
        Self::WORKGROUP_SIZE.x * Self::WORKGROUP_SIZE.y * Self::WORKGROUP_SIZE.z;

    pub fn index(global_id: UVec3) -> u32 {
        global_id.z * Self::WORKGROUP_SIZE.x * Self::WORKGROUP_SIZE.y
            + global_id.y * Self::WORKGROUP_SIZE.x
            + global_id.x
    }

    fn any_change_id(&self, index: u32, changes_slab: &[u32]) -> AnyChangeId {
        let id_id = self.changes_ids.at(index as usize);
        changes_slab.read_unchecked(id_id)
    }

    /// Returns the total number of invocations required.
    pub fn total_invocations_required(&self) -> u32 {
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

    fn atomic_i_increment(&self, data_slab: &mut [u32], index: usize) {
        #[cfg(target_arch = "spirv")]
        unsafe {
            spirv_std::arch::atomic_i_increment::<
                u32,
                { spirv_std::memory::Scope::Workgroup as u32 },
                { spirv_std::memory::Semantics::UNIFORM_MEMORY.bits() },
            >(&mut data_slab[index]);
        }
        #[cfg(not(target_arch = "spirv"))]
        {
            // Does not do atomically
            data_slab[index] += 1;
        }
    }

    fn increment_invocation(&self, data_slab: &mut [u32]) {
        let index = self.invocations_id.index();
        self.atomic_i_increment(data_slab, index);
    }

    fn increment_invocations_skipped(&self, data_slab: &mut [u32]) {
        let index = self.invocations_skipped_id.index();
        self.atomic_i_increment(data_slab, index);
    }

    pub fn run(data_slab: &mut [u32], changes_slab: &[u32], global_id: glam::UVec3) {
        let invocation: ApplyDataChangeInvocation = changes_slab.read_unchecked(Id::ZERO);
        let index = Self::index(global_id);
        if index >= invocation.changes_ids.len() as u32 {
            invocation.increment_invocations_skipped(data_slab);
            return;
        }

        invocation.increment_invocation(data_slab);

        let AnyChangeId {
            change_id,
            data_array,
        } = invocation.any_change_id(index, changes_slab);

        let change = changes_slab.read_unchecked(change_id);
        let data_id = data_array.at(change.i as usize);
        let mut data = data_slab.read_unchecked(data_id);
        change.change.apply(&mut data);
        data_slab.write(data_id, &data);
    }
}
