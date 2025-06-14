//! Projections of values into a slab.
//!
//! Holograms allow the programmer to use an index into a slab, represented as an
//! `Id` as the value itself.

pub trait IsHologram<T> {
    type Target<'a>;
    type TargetMut<'a>;

    fn hologram<'a>(&self, slab: &'a [u32]) -> Self::Target<'a>;
    fn hologram_mut<'a>(&self, slab: &'a mut [u32]) -> Self::TargetMut<'a>;
}

#[cfg(test)]
mod test {
    use crate as crabslab;
    use crate::SlabItem;
    use crate::{Id, Slab};

    use super::*;

    #[test]
    fn hologram_idea() {
        #[derive(Clone, Copy, SlabItem)]
        #[offsets]
        struct MyData {
            a: f32,
            b: u32,
            c: f32,
        }

        impl IsHologram<MyData> for Id<MyData> {
            type Target<'a> = MyDataHolo<'a>;
            type TargetMut<'a> = MyDataHoloMut<'a>;

            fn hologram<'a>(&self, slab: &'a [u32]) -> Self::Target<'a> {
                MyDataHolo {
                    slice: &slab[self.index()..self.index() + MyData::SLAB_SIZE],
                }
            }

            fn hologram_mut<'a>(&self, slab: &'a mut [u32]) -> Self::TargetMut<'a> {
                MyDataHoloMut {
                    slice: &mut slab[self.index()..self.index() + MyData::SLAB_SIZE],
                }
            }
        }

        #[repr(transparent)]
        struct MyDataHolo<'a> {
            slice: &'a [u32],
        }

        impl<'a> MyDataHolo<'a> {
            pub fn read_a(&self) -> f32 {
                self.slice
                    .read_unchecked(Id::<MyData>::new(0) + MyData::OFFSET_OF_A)
            }

            pub fn read_b(&self) -> u32 {
                self.slice
                    .read_unchecked(Id::<MyData>::new(0) + MyData::OFFSET_OF_B)
            }
            pub fn read_c(&self) -> f32 {
                self.slice
                    .read_unchecked(Id::<MyData>::new(0) + MyData::OFFSET_OF_C)
            }
        }

        #[repr(transparent)]
        struct MyDataHoloMut<'a> {
            slice: &'a mut [u32],
        }

        impl<'a> MyDataHoloMut<'a> {
            pub fn read_a(&self) -> f32 {
                self.slice
                    .read_unchecked(Id::<MyData>::new(0) + MyData::OFFSET_OF_A)
            }

            pub fn read_b(&self) -> u32 {
                self.slice
                    .read_unchecked(Id::<MyData>::new(0) + MyData::OFFSET_OF_B)
            }
            pub fn read_c(&self) -> f32 {
                self.slice
                    .read_unchecked(Id::<MyData>::new(0) + MyData::OFFSET_OF_C)
            }

            pub fn write_a(&mut self, a: &f32) {
                let _ = self.slice.write_indexed(a, MyData::OFFSET_OF_A.index());
            }

            pub fn write_b(&mut self, b: &u32) {
                let _ = self.slice.write_indexed(b, MyData::OFFSET_OF_B.index());
            }

            pub fn write_c(&mut self, c: &f32) {
                let _ = self.slice.write_indexed(c, MyData::OFFSET_OF_C.index());
            }
        }

        let mut slab = [0u32; 16];
        let id = Id::<MyData>::new(5);
        {
            let mut holo = id.hologram_mut(&mut slab);
            holo.write_a(&23.0);
        }

        {
            let holo = id.hologram(&slab);
            let a = holo.read_a();
            assert_eq!(23.0, a);
        }
    }
}
