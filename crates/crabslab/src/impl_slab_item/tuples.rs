use crate as crabslab;

// Rust tuples only implement Default for up to 12 elements, as of now
crabslab_derive::impl_slabitem_tuples!((A,));
crabslab_derive::impl_slabitem_tuples!((A, B));
crabslab_derive::impl_slabitem_tuples!((A, B, C));
crabslab_derive::impl_slabitem_tuples!((A, B, C, D));
crabslab_derive::impl_slabitem_tuples!((A, B, C, D, E));
crabslab_derive::impl_slabitem_tuples!((A, B, C, D, E, F));
crabslab_derive::impl_slabitem_tuples!((A, B, C, D, E, F, G));
crabslab_derive::impl_slabitem_tuples!((A, B, C, D, E, F, G, H));
crabslab_derive::impl_slabitem_tuples!((A, B, C, D, E, F, G, H, I));
crabslab_derive::impl_slabitem_tuples!((A, B, C, D, E, F, G, H, I, J));
crabslab_derive::impl_slabitem_tuples!((A, B, C, D, E, F, G, H, I, J, K));
crabslab_derive::impl_slabitem_tuples!((A, B, C, D, E, F, G, H, I, J, K, L));
crabslab_derive::impl_slabitem_tuples!((A, B, C, D, E, F, G, H, I, J, K, L, M));
