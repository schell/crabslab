//! `#[slab(crate = "...")]`: retargeting generated code at a path other
//! than the default `crabslab` name.
//!
//! Cargo rejects listing one package twice under two names in a single
//! manifest, so a true rename cannot be simulated with the workspace's
//! own crabslab. Instead this test retargets at a re-export module: the
//! generated impl must bind to the exact path the override names, not to
//! the default `crabslab`. Binding against the wrong runtime *version*
//! (two crabslabs in one graph) is proven by the dedicated two-version
//! test crate, `crabslab-test-two-versions`.

mod crabslab_wgsl {
    pub use crabslab::*;
}

use crabslab_wgsl::{CpuSlab, GrowableSlab, Slab, SlabItem};

#[derive(Clone, Copy, Debug, Default, PartialEq, SlabItem)]
#[slab(crate = "crabslab_wgsl")]
struct Camera {
    view_transform: [f32; 16],
    fov_y: f32,
    near: f32,
    far: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, SlabItem)]
#[slab(crate = "crabslab_wgsl")]
enum Shape {
    #[default]
    Empty,
    Sphere {
        center: [f32; 3],
        radius: f32,
    },
    Cube {
        min: [f32; 3],
        max: [f32; 3],
    },
}

#[test]
fn retargeted_struct_round_trips() {
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
fn retargeted_enum_round_trips() {
    assert_eq!(7, Shape::SLAB_SIZE);
    let shape = Shape::Sphere {
        center: [1.0, 2.0, 3.0],
        radius: 4.0,
    };
    let mut slab = CpuSlab::new(vec![]);
    let id = slab.append(&shape);
    assert_eq!(shape, slab.read(id));

    let cube = Shape::Cube {
        min: [-1.0, -1.0, -1.0],
        max: [1.0, 1.0, 1.0],
    };
    let id = slab.append(&cube);
    assert_eq!(cube, slab.read(id));
}
