use core::marker::PhantomData;

use glam::{Vec3, Vec4};
use spirv_std::spirv;

pub trait SlabItem {
    fn slab_size() -> usize;

    fn read_slab(id: u32, slab: &[u32]) -> Self;
}

impl SlabItem for f32 {
    fn slab_size() -> usize {
        1
    }

    fn read_slab(id: u32, slab: &[u32]) -> Self {
        f32::from_bits(slab[id as usize])
    }
}

impl SlabItem for u32 {
    fn slab_size() -> usize {
        1
    }

    fn read_slab(id: u32, slab: &[u32]) -> Self {
        slab[id as usize]
    }
}

#[repr(transparent)]
pub struct Id<T>(pub u32, PhantomData<T>);

impl<T> SlabItem for Id<T> {
    fn slab_size() -> usize {
        1
    }

    fn read_slab(id: u32, slab: &[u32]) -> Self {
        Id::new(slab[id as usize])
    }
}

impl<T> PartialOrd for Id<T> {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.0.cmp(&other.0))
    }
}

impl<T> Ord for Id<T> {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.0.cmp(&other.0)
    }
}

impl<T> Copy for Id<T> {}

impl<T> Clone for Id<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> core::hash::Hash for Id<T> {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl<T> PartialEq for Id<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<T> Eq for Id<T> {}

impl<T> From<Id<T>> for u32 {
    fn from(value: Id<T>) -> Self {
        value.0
    }
}

impl<T> From<usize> for Id<T> {
    fn from(value: usize) -> Self {
        Id::new(value as u32)
    }
}

impl<T> From<u32> for Id<T> {
    fn from(value: u32) -> Self {
        Id::new(value)
    }
}

/// `Id::NONE` is the default.
impl<T> Default for Id<T> {
    fn default() -> Self {
        Id::NONE
    }
}

impl<T> Id<T> {
    pub const NONE: Self = Id::new(ID_NONE);

    pub const fn new(i: u32) -> Self {
        Id(i, PhantomData)
    }

    /// Convert this id into a usize for use as an index.
    pub fn index(&self) -> usize {
        self.0 as usize
    }

    /// The raw u32 value of this id.
    pub fn inner(&self) -> u32 {
        self.0
    }

    pub fn is_none(&self) -> bool {
        *self == Id::NONE
    }

    pub fn is_some(&self) -> bool {
        !self.is_none()
    }
}

/// `u32` value of an [`Id`] that does not point to any item.
pub const ID_NONE: u32 = u32::MAX;

#[derive(Default, Clone, Copy)]
pub struct Sphere {
    pub radius: f32,
}

impl Sphere {
    pub fn distance(self, position: Vec3) -> f32 {
        position.length() - self.radius
    }
}

impl SlabItem for Sphere {
    fn slab_size() -> usize {
        1
    }

    fn read_slab(id: u32, slab: &[u32]) -> Self {
        let radius = f32::from_bits(slab[id as usize]);
        Sphere { radius }
    }
}

impl SlabItem for Vec3 {
    fn slab_size() -> usize {
        3
    }

    fn read_slab(id: u32, slab: &[u32]) -> Self {
        let id = id as usize;
        let x = f32::from_bits(slab[id]);
        let y = f32::from_bits(slab[id + 1]);
        let z = f32::from_bits(slab[id + 2]);
        Vec3::new(x, y, z)
    }
}

#[derive(Default, Clone, Copy)]
pub struct Plane {
    pub normal: Vec3,
    pub height: f32,
}

impl SlabItem for Plane {
    fn slab_size() -> usize {
        4
    }

    fn read_slab(id: u32, slab: &[u32]) -> Self {
        let normal = Vec3::read_slab(id, slab);
        let height = f32::read_slab(id + 3, slab);
        Plane { normal, height }
    }
}

impl Plane {
    pub fn distance(self, position: Vec3) -> f32 {
        position.dot(self.normal) + self.height
    }
}

#[derive(Clone, Copy)]
pub enum Sdf {
    Sphere(Id<Sphere>),
    Plane(Id<Plane>),
}

impl Default for Sdf {
    fn default() -> Self {
        Sdf::Sphere(Id::NONE)
    }
}

impl SlabItem for Sdf {
    fn slab_size() -> usize {
        2
    }

    fn read_slab(id: u32, slab: &[u32]) -> Self {
        let hash = u32::read_slab(id, slab);
        match hash {
            0 => {
                let sphere_id = Id::read_slab(id + 1, slab);
                Sdf::Sphere(sphere_id)
            }
            1 => {
                let plane_id = Id::read_slab(id + 1, slab);
                Sdf::Plane(plane_id)
            }
            _ => Self::default(),
        }
    }
}

impl Sdf {
    pub fn distance(self, slab: &[u32], position: Vec3) -> f32 {
        match self {
            Sdf::Sphere(id) => {
                let sphere = Sphere::read_slab(id.0, slab);
                sphere.distance(position)
            }
            Sdf::Plane(id) => {
                let plane = Plane::read_slab(id.0, slab);
                plane.distance(position)
            }
        }
    }
}

#[repr(C)]
pub struct Array<T> {
    // u32 offset in the slab
    index: u32,
    // number of `T` elements in the array
    len: u32,
    _phantom: PhantomData<T>,
}

impl<T> Clone for Array<T> {
    fn clone(&self) -> Self {
        Self {
            index: self.index,
            len: self.len,
            _phantom: PhantomData,
        }
    }
}

impl<T> Copy for Array<T> {}

/// An `Id<T>` is an `Array<T>` with a length of 1.
impl<T> From<Id<T>> for Array<T> {
    fn from(id: Id<T>) -> Self {
        Self {
            index: id.inner(),
            len: 1,
            _phantom: PhantomData,
        }
    }
}

impl<T> Default for Array<T> {
    fn default() -> Self {
        Self {
            index: u32::MAX,
            len: 0,
            _phantom: PhantomData,
        }
    }
}

impl<T> Array<T> {
    pub fn new(index: u32, len: u32) -> Self {
        Self {
            index,
            len,
            _phantom: PhantomData,
        }
    }

    pub fn len(&self) -> usize {
        self.len as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn is_null(&self) -> bool {
        self.index == u32::MAX
    }

    pub fn contains_index(&self, index: usize) -> bool {
        index >= self.index as usize && index < (self.index + self.len) as usize
    }

    pub fn at(&self, index: usize) -> Id<T>
    where
        T: SlabItem,
    {
        Id::new(self.index + (T::slab_size() * index) as u32)
    }

    pub fn starting_index(&self) -> usize {
        self.index as usize
    }

    /// Convert this array into a `u32` array.
    pub fn into_u32_array(self) -> Array<u32>
    where
        T: SlabItem,
    {
        Array {
            index: self.index,
            len: self.len * T::slab_size() as u32,
            _phantom: PhantomData,
        }
    }
}

impl<T> SlabItem for Array<T> {
    fn slab_size() -> usize {
        2
    }

    fn read_slab(id: u32, slab: &[u32]) -> Self {
        let index = u32::read_slab(id, slab);
        let len = u32::read_slab(id, slab);
        Array::new(index, len)
    }
}

#[derive(Default, Clone, Copy)]
pub struct Scene {
    pub sdfs: Array<Sdf>,
}

impl SlabItem for Scene {
    fn slab_size() -> usize {
        2
    }

    fn read_slab(id: u32, slab: &[u32]) -> Self {
        let sdfs = Array::<Sdf>::read_slab(id, slab);
        Scene { sdfs }
    }
}

impl Scene {
    pub fn distance(self, slab: &[u32], position: Vec3) -> f32 {
        let mut distance = f32::MAX;
        for i in 0..self.sdfs.len() {
            let id = self.sdfs.at(i);
            let sdf = Sdf::read_slab(id.0, slab);
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
    let scene = Scene::read_slab(scene_id.0, slab);
    let distance = scene.distance(slab, position);
    *frag_color = Vec4::new(distance, distance, distance, 1.0);
}
