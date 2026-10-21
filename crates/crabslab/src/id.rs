//! Typed identifiers that can also be used as indices.
use crate::wgsl::{Id, SlabItem};

impl<T> PartialOrd for Id<T> {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for Id<T> {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.index.cmp(&other.index)
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
        self.index.hash(state);
    }
}

impl<T> PartialEq for Id<T> {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index
    }
}

impl<T> Eq for Id<T> {}

impl<T> From<Id<T>> for u32 {
    fn from(value: Id<T>) -> Self {
        value.index
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
        Id::none()
    }
}

impl<T> core::fmt::Debug for Id<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if self.is_none() {
            f.write_fmt(core::format_args!(
                "Id<{}>(null)",
                core::any::type_name::<T>(),
            ))
        } else {
            f.write_fmt(core::format_args!(
                "Id<{}>({})",
                core::any::type_name::<T>(),
                self.index
            ))
        }
    }
}

impl<T> core::ops::Add<usize> for Id<T> {
    type Output = Self;

    fn add(self, rhs: usize) -> Self::Output {
        Id::new(self.index + rhs as u32)
    }
}

impl<T> core::ops::Add<Id<T>> for usize {
    type Output = Id<T>;

    fn add(self, rhs: Id<T>) -> Self::Output {
        Id::new(self as u32 + rhs.index)
    }
}

impl<T> core::ops::Add<u32> for Id<T> {
    type Output = Self;

    fn add(self, rhs: u32) -> Self::Output {
        Id::new(self.index + rhs)
    }
}

impl<T> core::ops::Add<Id<T>> for u32 {
    type Output = Id<T>;

    fn add(self, rhs: Id<T>) -> Self::Output {
        Id::new(self + rhs.index)
    }
}

impl<T: SlabItem> core::ops::Index<Id<T>> for [u32] {
    type Output = T;

    fn index(&self, id: Id<T>) -> &Self::Output {
        let index = id.index();
        let s = &self[index..index + T::SLAB_SIZE];
        unsafe { &*(s.as_ptr() as *const T) }
    }
}

impl<T: SlabItem> core::ops::IndexMut<Id<T>> for [u32] {
    fn index_mut(&mut self, id: Id<T>) -> &mut Self::Output {
        let index = id.index();
        let s = &mut self[index..index + T::SLAB_SIZE];
        unsafe { &mut *(s.as_mut_ptr() as *mut T) }
    }
}

impl<T> Id<T> {
    // /// The non-id. Analogous to `None::<Id>`.
    // pub const NONE: Self = Id::new(ID_NONE);

    // /// An `Id` that points to the first slot on a slab.
    // pub const ZERO: Self = Id::new(0);

    // /// Create an `Id` from a raw slot index.
    // pub const fn new(i: u32) -> Self {
    //     Id {
    //         index: i,
    //         _phantom: PhantomData,
    //     }
    // }

    /// Convert this id into a usize for use as an index.
    pub fn index(&self) -> usize {
        self.index as usize
    }

    /// The raw u32 value of this id.
    pub fn inner(&self) -> u32 {
        self.index
    }

    /// Returns `true` if this is the non-id ([`Id::NONE`]).
    pub fn is_none(&self) -> bool {
        *self == Id::none()
    }

    /// Returns `true` if this id points at an item.
    pub fn is_some(&self) -> bool {
        !self.is_none()
    }
}

impl<T: SlabItem> Id<T> {
    /// Returns the u32 slab range that this `Id` occupies.
    pub fn range_u32(&self) -> core::ops::Range<usize> {
        self.index()..(self.index() + T::SLAB_SIZE)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::slab::{CpuSlabItem, SlabItem};
    use wgsl_rs::std::{vec3f, Vec3f};

    #[test]
    fn id_size() {
        #[derive(SlabItem)]
        #[slab(crate = "crate")]
        struct MyEntity {
            name: u32,
            age: f32,
            destiny: [u32; 3],
        }

        assert_eq!(
            std::mem::size_of::<u32>(),
            std::mem::size_of::<Id<MyEntity>>(),
            "id is not u32"
        );
    }

    #[test]
    fn write_id() {
        let mut slab = [0u32; 4];
        let id = Id::<u32>::new(666);
        let index = 0;
        let index = id.write_slab(index, &mut slab);
        assert_eq!(1, index);
        let index = id.write_slab(index, &mut slab);
        assert_eq!(2, index);
    }

    #[test]
    fn index() {
        let mut slab = [0u32; 4];
        let id = Id::<Vec3f>::new(1);
        {
            let zero = &slab[id];
            assert_eq!(&vec3f(0.0, 0.0, 0.0), zero);
        }

        {
            let one = &mut slab[id];
            one.y = 1.0;
        }

        let one = &slab[id];
        assert_eq!(&vec3f(0.0, 1.0, 0.0), one);
    }

    #[test]
    fn slice_representation_sanity() {
        #[derive(Debug)]
        struct V3 {
            x: f32,
            y: f32,
            z: f32,
            __padding: f32,
        }

        fn get_mut<T>(slab: &mut [u32], index: usize) -> &mut T {
            let slice = &mut slab[index..];
            unsafe { &mut *(slice.as_mut_ptr() as *mut T) }
        }

        let mut slab = [0.0f32, 1.0, 2.0, 3.0].map(|f| f.to_bits());

        {
            let vec3: &mut V3 = get_mut(&mut slab, 0);
            vec3.x *= 2.0;
            vec3.y *= 2.0;
            vec3.z *= 2.0;
        }

        {
            let f: &mut f32 = get_mut(&mut slab, 3);
            *f = 666.0;
        }

        let vec3: &V3 = unsafe { &*(slab.as_ptr() as *const V3) };

        println!("vec3: {vec3:?}");
    }
}
