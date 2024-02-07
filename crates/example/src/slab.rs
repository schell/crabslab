use crabslab::*;
use glam::{Vec3, Vec4};
use spirv_std::spirv;

#[derive(Default, Clone, Copy, SlabItem)]
pub struct Sphere {
    pub radius: f32,
}

impl Sphere {
    pub fn distance(self, position: Vec3) -> f32 {
        position.length() - self.radius
    }
}

#[derive(Default, Clone, Copy, SlabItem)]
pub struct Plane {
    pub normal: Vec3,
    pub height: f32,
}

impl Plane {
    pub fn distance(self, position: Vec3) -> f32 {
        position.dot(self.normal) + self.height
    }
}

#[derive(Clone, Copy, SlabItem)]
pub enum Sdf {
    Sphere(Id<Sphere>),
    Plane(Id<Plane>),
}

impl Default for Sdf {
    fn default() -> Self {
        Sdf::Sphere(Id::NONE)
    }
}

impl Sdf {
    pub fn distance(self, slab: &[u32], position: Vec3) -> f32 {
        match self {
            Sdf::Sphere(id) => {
                let sdf = slab.read(id);
                sdf.distance(position)
            }
            Sdf::Plane(id) => {
                let sdf = slab.read(id);
                sdf.distance(position)
            }
        }
    }
}

#[derive(Default, Clone, Copy, SlabItem)]
pub struct Scene {
    pub sdfs: Array<Sdf>,
}

impl Scene {
    pub fn distance(self, slab: &[u32], position: Vec3) -> f32 {
        let mut distance = f32::MAX;
        for id in self.sdfs.iter() {
            let sdf: Sdf = slab.read(id);
            distance = distance.min(sdf.distance(slab, position));
        }
        distance
    }
}

#[spirv(fragment)]
pub fn fragment(
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] slab: &[u32],
    #[spirv(flat)] scene_id: Id<Scene>,
    position: Vec3,
    frag_color: &mut Vec4,
) {
    let scene = slab.read(scene_id);
    let distance = scene.distance(slab, position);
    *frag_color = Vec4::new(distance, distance, distance, 1.0);
}
