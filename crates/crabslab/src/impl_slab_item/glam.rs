use crate::SlabItem;

impl SlabItem for glam::Mat4 {
    fn read_slab(&mut self, index: usize, slab: &[u32]) -> usize {
        let Self {
            x_axis,
            y_axis,
            z_axis,
            w_axis,
        } = self;
        let index = x_axis.read_slab(index, slab);
        let index = y_axis.read_slab(index, slab);
        let index = z_axis.read_slab(index, slab);
        w_axis.read_slab(index, slab)
    }

    fn slab_size() -> usize {
        16
    }

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
    fn slab_size() -> usize {
        2
    }

    fn read_slab(&mut self, index: usize, slab: &[u32]) -> usize {
        let index = self.x.read_slab(index, slab);
        let index = self.y.read_slab(index, slab);
        index
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
    fn slab_size() -> usize {
        3
    }

    fn read_slab(&mut self, index: usize, slab: &[u32]) -> usize {
        let Self { x, y, z } = self;
        let index = x.read_slab(index, slab);
        let index = y.read_slab(index, slab);
        let index = z.read_slab(index, slab);
        index
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
    fn slab_size() -> usize {
        4
    }

    fn read_slab(&mut self, index: usize, slab: &[u32]) -> usize {
        let index = self.x.read_slab(index, slab);
        let index = self.y.read_slab(index, slab);
        let index = self.z.read_slab(index, slab);
        self.w.read_slab(index, slab)
    }

    fn write_slab(&self, index: usize, slab: &mut [u32]) -> usize {
        let index = self.x.write_slab(index, slab);
        let index = self.y.write_slab(index, slab);
        let index = self.z.write_slab(index, slab);
        self.w.write_slab(index, slab)
    }
}

impl SlabItem for glam::Quat {
    fn slab_size() -> usize {
        16
    }

    fn read_slab(&mut self, index: usize, slab: &[u32]) -> usize {
        let index = self.x.read_slab(index, slab);
        let index = self.y.read_slab(index, slab);
        let index = self.z.read_slab(index, slab);
        self.w.read_slab(index, slab)
    }

    fn write_slab(&self, index: usize, slab: &mut [u32]) -> usize {
        let index = self.x.write_slab(index, slab);
        let index = self.y.write_slab(index, slab);
        let index = self.z.write_slab(index, slab);
        self.w.write_slab(index, slab)
    }
}

impl SlabItem for glam::UVec2 {
    fn slab_size() -> usize {
        2
    }

    fn read_slab(&mut self, index: usize, slab: &[u32]) -> usize {
        let index = self.x.read_slab(index, slab);
        let index = self.y.read_slab(index, slab);
        index
    }

    fn write_slab(&self, index: usize, slab: &mut [u32]) -> usize {
        let index = self.x.write_slab(index, slab);
        let index = self.y.write_slab(index, slab);
        index
    }
}

impl SlabItem for glam::UVec3 {
    fn slab_size() -> usize {
        3
    }

    fn read_slab(&mut self, index: usize, slab: &[u32]) -> usize {
        let index = self.x.read_slab(index, slab);
        let index = self.y.read_slab(index, slab);
        let index = self.z.read_slab(index, slab);
        index
    }

    fn write_slab(&self, index: usize, slab: &mut [u32]) -> usize {
        let index = self.x.write_slab(index, slab);
        let index = self.y.write_slab(index, slab);
        let index = self.z.write_slab(index, slab);
        index
    }
}

impl SlabItem for glam::UVec4 {
    fn slab_size() -> usize {
        4
    }

    fn read_slab(&mut self, index: usize, slab: &[u32]) -> usize {
        let index = self.x.read_slab(index, slab);
        let index = self.y.read_slab(index, slab);
        let index = self.z.read_slab(index, slab);
        let index = self.w.read_slab(index, slab);
        index
    }

    fn write_slab(&self, index: usize, slab: &mut [u32]) -> usize {
        if slab.len() < index + 4 {
            return index;
        }
        let index = self.x.write_slab(index, slab);
        let index = self.y.write_slab(index, slab);
        let index = self.z.write_slab(index, slab);
        let index = self.w.write_slab(index, slab);
        index
    }
}
