//! Two crabslab versions in one dependency graph: published 0.6.6 for
//! the rust-gpu path (renamed `crabslab_old`) beside this workspace's
//! wgsl-rs-integrated 0.7.x (renamed `crabslab_wgsl`).
//!
//! This is the end-to-end proof of the version split:
//!
//! - The graph holds TWO distinct crabslab-derive nodes: 0.4.6 (pinned via
//!   a direct dependency, resolved through `crabslab_old`) and 0.7.0
//!   (reached through `crabslab_wgsl`). Before the split, both runtimes
//!   pulled one derive node and its expansions bound to whichever crate
//!   was named `crabslab`.
//! - New-path types derive via the rename-aware 0.7.0 macro with the
//!   `#[slab(crate = "crabslab_wgsl")]` override and round-trip through
//!   the new runtime's slab.
//! - Old-path types derive via the published 0.4.6 macro, which emits
//!   paths through the name `crabslab`; the `use crabslab_old as crabslab`
//!   alias inside the `legacy` module routes those to the old runtime.
//!   That alias is the documented workaround, since the published 0.4.x
//!   derive cannot be retargeted.

use crabslab_wgsl::{CpuSlab, GrowableSlab, Slab, SlabItem};

/// New-path type: rename-aware derive (0.7.0) with the explicit override.
#[derive(Clone, Copy, Debug, Default, PartialEq, SlabItem)]
#[slab(crate = "crabslab_wgsl")]
struct Camera {
    view_transform: [f32; 16],
    fov_y: f32,
    near: f32,
    far: f32,
}

/// Old-path types: the published 0.4.x derive is not rename-aware; its
/// expansions go through the name `crabslab`, aliased here to the old
/// runtime.
mod legacy {
    pub use crabslab_old as crabslab;
    use crabslab_old::SlabItem;

    #[derive(Clone, Copy, Debug, Default, PartialEq, SlabItem)]
    pub struct Light {
        pub position: [u32; 3],
        pub intensity: u32,
    }

    pub fn light_slab_size() -> usize {
        Light::SLAB_SIZE
    }

    pub fn round_trip_light(light: Light) -> Light {
        use crabslab_old::{CpuSlab, GrowableSlab, Slab};
        let mut slab = CpuSlab::new(vec![]);
        let id = slab.append(&light);
        slab.read(id)
    }
}

#[test]
fn new_path_derive_binds_to_new_runtime() {
    assert_eq!(19, Camera::SLAB_SIZE);
    let camera = Camera {
        view_transform: [1.0; 16],
        fov_y: 0.5,
        near: 0.1,
        far: 100.0,
    };
    let mut slab = CpuSlab::new(vec![]);
    let id = slab.append(&camera);
    assert_eq!(camera, slab.read(id));
}

#[test]
fn old_path_derive_binds_to_old_runtime() {
    let light = legacy::Light {
        position: [1, 2, 3],
        intensity: 42,
    };
    assert_eq!(4, legacy::light_slab_size());
    assert_eq!(light, legacy::round_trip_light(light));
}

#[test]
fn both_runtimes_dispatch_builtin_types() {
    // Both slabs coexist in this crate; each round-trips its own
    // primitives through its own runtime.
    let mut new_slab = CpuSlab::new(vec![]);
    let new_id = new_slab.append(&7u32);
    assert_eq!(7, new_slab.read(new_id));

    use crabslab_old::{CpuSlab, GrowableSlab, Slab};
    let mut old_slab = CpuSlab::new(vec![]);
    let old_id = old_slab.append(&9u32);
    assert_eq!(9, old_slab.read(old_id));
}
