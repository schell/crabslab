//! Automatic rename resolution: the runtime is renamed in `Cargo.toml`
//! and the derive below sets NO `#[slab]` override. The rename-aware
//! derive must discover `crabslab_wgsl` by itself, via `proc-macro-crate`,
//! and bind the generated impl to it.
//!
//! The explicit `#[slab(crate = "...")]` override is covered by
//! `crabslab-derive`'s own tests; binding against the correct runtime
//! *version* is covered by `crabslab-test-two-versions`.

use crabslab_wgsl::{CpuSlab, GrowableSlab, Slab, SlabItem};

#[derive(Clone, Copy, Debug, Default, PartialEq, SlabItem)]
struct GpuCamera {
    position: [f32; 3],
    fov_y: f32,
}

#[test]
fn renamed_dep_resolves_without_override() {
    assert_eq!(4, GpuCamera::SLAB_SIZE);
    let camera = GpuCamera {
        position: [1.0, 2.0, 3.0],
        fov_y: 0.5,
    };
    let mut slab = CpuSlab::new(vec![]);
    let id = slab.append(&camera);
    assert_eq!(camera, slab.read(id));
}
