#![no_std]
#![allow(unexpected_cfgs)]

use craballoc_test_wire_types::*;
use crabslab::{Id, Slab};
use spirv_std::spirv;

#[spirv(compute(threads(16)))]
pub fn apply_data_changes(
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] data_slab: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] changes_slab: &[u32],
    #[spirv(global_invocation_id)] global_id: glam::UVec3,
) {
    let invocation: ApplyDataChangeInvocation = changes_slab.read_unchecked(Id::ZERO);
    invocation.run(data_slab, changes_slab, global_id);
}
