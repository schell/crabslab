use glam::{Mat4, Quat, UVec2, UVec3, UVec4, Vec2, Vec3, Vec4};

use crate::SlabItem;

impl SlabItem for glam::Mat4 {
    fn read_slab(index: usize, slab: &[u32]) -> Self {
        let x_axis = Vec4::read_slab(index, slab);
        let y_axis = Vec4::read_slab(index + 4, slab);
        let z_axis = Vec4::read_slab(index + 8, slab);
        let w_axis = Vec4::read_slab(index + 12, slab);
        Mat4::from_cols(x_axis, y_axis, z_axis, w_axis)
    }

    const SLAB_SIZE: usize = { 16 };

    fn write_slab(&self, index: usize, slab: &mut [u32]) -> usize {
        let Self {
            x_axis,
            y_axis,
            z_axis,
            w_axis,
        } = self;
        let index = x_axis.write_slab(index, slab);
        let index = y_axis.write_slab(index, slab);
        let index = z_axis.write_slab(index, slab);
        w_axis.write_slab(index, slab)
    }
}

impl SlabItem for glam::Vec2 {
    const SLAB_SIZE: usize = { 2 };

    fn read_slab(index: usize, slab: &[u32]) -> Self {
        let x = f32::read_slab(index, slab);
        let y = f32::read_slab(index + 1, slab);
        Vec2::new(x, y)
    }

    fn write_slab(&self, index: usize, slab: &mut [u32]) -> usize {
        if slab.len() < index + 2 {
            return index;
        }
        let index = self.x.write_slab(index, slab);
        let index = self.y.write_slab(index, slab);
        index
    }
}

impl SlabItem for glam::Vec3 {
    const SLAB_SIZE: usize = { 3 };

    fn read_slab(index: usize, slab: &[u32]) -> Self {
        let x = f32::read_slab(index, slab);
        let y = f32::read_slab(index + 1, slab);
        let z = f32::read_slab(index + 2, slab);
        Vec3::new(x, y, z)
    }

    fn write_slab(&self, index: usize, slab: &mut [u32]) -> usize {
        let Self { x, y, z } = self;
        let index = x.write_slab(index, slab);
        let index = y.write_slab(index, slab);
        let index = z.write_slab(index, slab);
        index
    }
}

impl SlabItem for glam::Vec4 {
    const SLAB_SIZE: usize = { 4 };

    fn read_slab(index: usize, slab: &[u32]) -> Self {
        Vec4::new(
            f32::read_slab(index, slab),
            f32::read_slab(index + 1, slab),
            f32::read_slab(index + 2, slab),
            f32::read_slab(index + 3, slab),
        )
    }

    fn write_slab(&self, index: usize, slab: &mut [u32]) -> usize {
        let index = self.x.write_slab(index, slab);
        let index = self.y.write_slab(index, slab);
        let index = self.z.write_slab(index, slab);
        self.w.write_slab(index, slab)
    }
}

impl SlabItem for glam::Quat {
    const SLAB_SIZE: usize = { 16 };

    fn read_slab(index: usize, slab: &[u32]) -> Self {
        Quat::from_xyzw(
            f32::read_slab(index, slab),
            f32::read_slab(index + 1, slab),
            f32::read_slab(index + 2, slab),
            f32::read_slab(index + 3, slab),
        )
    }

    fn write_slab(&self, index: usize, slab: &mut [u32]) -> usize {
        let index = self.x.write_slab(index, slab);
        let index = self.y.write_slab(index, slab);
        let index = self.z.write_slab(index, slab);
        self.w.write_slab(index, slab)
    }
}

impl SlabItem for glam::UVec2 {
    const SLAB_SIZE: usize = { 2 };

    fn read_slab(index: usize, slab: &[u32]) -> Self {
        UVec2::new(u32::read_slab(index, slab), u32::read_slab(index + 1, slab))
    }

    fn write_slab(&self, index: usize, slab: &mut [u32]) -> usize {
        let index = self.x.write_slab(index, slab);
        let index = self.y.write_slab(index, slab);
        index
    }
}

impl SlabItem for glam::UVec3 {
    const SLAB_SIZE: usize = { 3 };

    fn read_slab(index: usize, slab: &[u32]) -> Self {
        UVec3::new(
            u32::read_slab(index, slab),
            u32::read_slab(index + 1, slab),
            u32::read_slab(index + 2, slab),
        )
    }

    fn write_slab(&self, index: usize, slab: &mut [u32]) -> usize {
        let index = self.x.write_slab(index, slab);
        let index = self.y.write_slab(index, slab);
        let index = self.z.write_slab(index, slab);
        index
    }
}

impl SlabItem for glam::UVec4 {
    const SLAB_SIZE: usize = { 4 };

    fn read_slab(index: usize, slab: &[u32]) -> Self {
        UVec4::new(
            u32::read_slab(index, slab),
            u32::read_slab(index + 1, slab),
            u32::read_slab(index + 2, slab),
            u32::read_slab(index + 3, slab),
        )
    }

    fn write_slab(&self, index: usize, slab: &mut [u32]) -> usize {
        let index = self.x.write_slab(index, slab);
        let index = self.y.write_slab(index, slab);
        let index = self.z.write_slab(index, slab);
        let index = self.w.write_slab(index, slab);
        index
    }
}
