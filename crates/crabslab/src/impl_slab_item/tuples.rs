

use crate::SlabItem;


macro_rules! impl_tuples {

    ($($generic: tt),+) => {
        impl<$($generic),+,> SlabItem for ($($generic),+,)
        where
            $($generic: SlabItem),+,
        {
            fn slab_size() -> usize {
                $($generic::slab_size() +)+ 0
            }
            fn read_slab(&mut self, index: usize, slab: &[u32]) -> usize {
                $(${ignore(generic)} let index = self.${index()}.read_slab(index, slab);)+
                index
            }
            fn write_slab(&self, index: usize, slab: &mut [u32]) -> usize {
                $(${ignore(generic)} let index = self.${index()}.write_slab(index, slab);)+
                index
            }
        }

        impl_tuples!(@pop $($generic),+);
    };

    (@pop $_: tt, $($generic: tt),+) => {
        impl_tuples!($($generic),+);
    };
    (@pop $_: tt) => {};
}

//Rust tuples only implement Default for up to 12 elements, as of now
impl_tuples!(M, L, J, I, H, G, F, E, D, C, B, A);