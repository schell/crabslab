use glam::{Vec3, Vec4};
use spirv_std::spirv;

fn read_u32(slab: &[u32], id: u32) -> u32 {
    slab[id as usize]
}

fn read_f32(slab: &[u32], id: u32) -> f32 {
    f32::from_bits(slab[id as usize])
}

fn read_vec3(slab: &[u32], id: u32) -> Vec3 {
    let x = read_f32(slab, id);
    let y = read_f32(slab, id + 1);
    let z = read_f32(slab, id + 2);
    Vec3::new(x, y, z)
}

#[derive(Default, Clone, Copy)]
pub struct Sphere {
    pub radius: f32,
}

impl Sphere {
    pub fn distance(self, position: Vec3) -> f32 {
        position.length() - self.radius
    }
}

fn read_sphere(slab: &[u32], id: u32) -> Sphere {
    Sphere {
        radius: read_f32(slab, id),
    }
}

#[derive(Default, Clone, Copy)]
pub struct Plane {
    pub normal: Vec3,
    pub height: f32,
}

impl Plane {
    pub fn distance(self, position: Vec3) -> f32 {
        position.dot(self.normal) + self.height
    }
}

fn read_plane(slab: &[u32], id: u32) -> Plane {
    let normal = read_vec3(slab, id);
    let height = read_f32(slab, id + 3);
    Plane { normal, height }
}

#[derive(Clone, Copy)]
pub enum Sdf {
    Sphere(u32),
    Plane(u32),
}

impl Default for Sdf {
    fn default() -> Self {
        Sdf::Sphere(u32::MAX)
    }
}

impl Sdf {
    pub fn distance(self, slab: &[u32], position: Vec3) -> f32 {
        match self {
            Sdf::Sphere(id) => {
                let sdf = read_sphere(slab, id);
                sdf.distance(position)
            }
            Sdf::Plane(id) => {
                let sdf = read_plane(slab, id);
                sdf.distance(position)
            }
        }
    }
}

fn len_sdf() -> u32 {
    2
}

fn read_sdf(slab: &[u32], id: u32) -> Sdf {
    let hash = read_u32(slab, id);
    let next_id = read_u32(slab, id + 1);
    match hash {
        0 => Sdf::Sphere(next_id),
        1 => Sdf::Plane(next_id),
        _ => Sdf::default(),
    }
}

#[derive(Default, Clone, Copy)]
pub struct Scene {
    pub sdf_starting_id: u32,
    pub sdf_len: u32,
}

impl Scene {
    pub fn distance(self, slab: &[u32], position: Vec3) -> f32 {
        let starting_id = self.sdf_starting_id;
        let mut distance = f32::MAX;
        for i in 0..self.sdf_len {
            let id = starting_id + i * len_sdf();
            let sdf: Sdf = read_sdf(slab, id);
            distance = distance.min(sdf.distance(slab, position));
        }
        distance
    }
}

fn read_scene(slab: &[u32], id: u32) -> Scene {
    let sdf_starting_id = read_u32(slab, id);
    let sdf_len = read_u32(slab, id + 1);
    Scene {
        sdf_starting_id,
        sdf_len,
    }
}

#[spirv(fragment)]
pub fn fragment(
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] slab: &[u32],
    #[spirv(flat)] scene_id: u32,
    position: Vec3,
    frag_color: &mut Vec4,
) {
    let scene = read_scene(slab, scene_id);
    let distance = scene.distance(slab, position);
    *frag_color = Vec4::new(distance, distance, distance, 1.0);
}
