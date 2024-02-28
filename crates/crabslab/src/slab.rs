//! Slab traits.
use core::default::Default;
pub use crabslab_derive::SlabItem;

use crate::{array::Array, id::Id};

/// Determines the "size" of a type when stored in a slab of `&[u32]`,
/// and how to read/write it from/to the slab.
///
/// `SlabItem` can be automatically derived for struct and tuple types,
/// so long as those types' fields implement `SlabItem`.
pub trait SlabItem: core::any::Any + Sized {
    /// The number of `u32`s this type occupies in a slab of `&[u32]`.
    const SLAB_SIZE: usize;

    /// Read the type out of the slab at starting `index` and return
    /// the new index.
    ///
    /// If the type cannot be read, the returned index will be equal
    /// to `index`.
    fn read_slab(index: usize, slab: &[u32]) -> Self;

    /// Write the type into the slab at starting `index` and return
    /// the new index.
    ///
    /// If the type cannot be written, the returned index will be equal
    /// to `index`.
    fn write_slab(&self, index: usize, slab: &mut [u32]) -> usize;
}

/// Trait for slabs of `u32`s that can store many types.
pub trait Slab {
    /// Return the number of u32 elements in the slab.
    fn len(&self) -> usize;

    /// Returns whether the slab may contain the value with the given id.
    fn contains<T: SlabItem>(&self, id: Id<T>) -> bool {
        id.index() + T::SLAB_SIZE <= self.len()
    }

    /// Read the type from the slab using the Id as the index.
    fn read<T: SlabItem + Default>(&self, id: Id<T>) -> T;

    #[cfg(not(target_arch = "spirv"))]
    fn read_vec<T: SlabItem + Default>(&self, array: crate::array::Array<T>) -> Vec<T> {
        let mut vec = Vec::with_capacity(array.len());
        for i in 0..array.len() {
            let id = array.at(i);
            vec.push(self.read(id));
        }
        vec
    }

    /// Write the type into the slab at the index.
    ///
    /// Return the next index, or the same index if writing would overlap the
    /// slab.
    fn write_indexed<T: SlabItem>(&mut self, t: &T, index: usize) -> usize;

    /// Write a slice of the type into the slab at the index.
    ///
    /// Return the next index, or the same index if writing would overlap the
    /// slab.
    fn write_indexed_slice<T: SlabItem>(&mut self, t: &[T], index: usize) -> usize;

    /// Write the type into the slab at the position of the given `Id`.
    ///
    /// This likely performs a partial write if the given `Id` is out of bounds.
    fn write<T: SlabItem>(&mut self, id: Id<T>, t: &T) {
        let _ = self.write_indexed(t, id.index());
    }

    /// Write contiguous elements into the slab at the position of the given
    /// `Array`.
    ///
    /// ## NOTE
    /// This does nothing if the length of `Array` is greater than the length of
    /// `data`.
    fn write_array<T: SlabItem>(&mut self, array: Array<T>, data: &[T]) {
        if array.len() > data.len() {
            return;
        }
        let _ = self.write_indexed_slice(data, array.starting_index());
    }
}

impl Slab for [u32] {
    fn len(&self) -> usize {
        self.len()
    }

    fn read<T: SlabItem>(&self, id: Id<T>) -> T {
        T::read_slab(id.0 as usize, self)
    }

    fn write_indexed<T: SlabItem>(&mut self, t: &T, index: usize) -> usize {
        t.write_slab(index, self)
    }

    fn write_indexed_slice<T: SlabItem>(&mut self, t: &[T], index: usize) -> usize {
        let mut index = index;
        for item in t {
            index = item.write_slab(index, self);
        }
        index
    }
}

#[cfg(not(target_arch = "spirv"))]
impl Slab for Vec<u32> {
    fn len(&self) -> usize {
        self.len()
    }

    fn read<T: SlabItem + Default>(&self, id: Id<T>) -> T {
        self.as_slice().read(id)
    }

    fn write_indexed<T: SlabItem>(&mut self, t: &T, index: usize) -> usize {
        self.as_mut_slice().write_indexed(t, index)
    }

    fn write_indexed_slice<T: SlabItem>(&mut self, t: &[T], index: usize) -> usize {
        self.as_mut_slice().write_indexed_slice(t, index)
    }
}

/// Trait for slabs of `u32`s that can store many types, and can grow to fit.
pub trait GrowableSlab: Slab {
    /// Return the current capacity of the slab.
    fn capacity(&self) -> usize;

    /// Reserve enough space on the slab to fit the given capacity.
    fn reserve_capacity(&mut self, capacity: usize);

    /// Increment the length of the slab by `n` u32s.
    ///
    /// Returns the previous length.
    fn increment_len(&mut self, n: usize) -> usize;

    /// Expands the slab to fit the given number of `T`s, if necessary.
    fn maybe_expand_to_fit<T: SlabItem>(&mut self, len: usize) {
        let capacity = self.capacity();
        // log::trace!(
        //    "append_slice: {size} * {ts_len} + {len} ({}) >= {capacity}",
        //    size * ts_len + len
        //);
        let capacity_needed = self.len() + T::SLAB_SIZE * len;
        if capacity_needed > capacity {
            let mut new_capacity = capacity * 2;
            while new_capacity < capacity_needed {
                new_capacity = (new_capacity * 2).max(2);
            }
            self.reserve_capacity(new_capacity);
        }
    }

    /// Preallocate space for one `T` element, but don't write anything to the
    /// buffer.
    ///
    /// The returned `Id` can be used to write later with [`Slab::write`].
    ///
    /// NOTE: This changes the next available buffer index and may change the
    /// buffer capacity.
    fn allocate<T: SlabItem>(&mut self) -> Id<T> {
        self.maybe_expand_to_fit::<T>(1);
        let index = self.increment_len(T::SLAB_SIZE);
        Id::from(index)
    }

    /// Preallocate space for `len` `T` elements, but don't write to
    /// the buffer.
    ///
    /// This can be used to allocate space for a bunch of elements that get
    /// written later with [`Slab::write_array`].
    ///
    /// NOTE: This changes the length of the buffer and may change the capacity.
    fn allocate_array<T: SlabItem>(&mut self, len: usize) -> Array<T> {
        if len == 0 {
            return Array::default();
        }
        self.maybe_expand_to_fit::<T>(len);
        let index = self.increment_len(T::SLAB_SIZE * len);
        Array::new(index as u32, len as u32)
    }

    /// Append to the end of the buffer.
    ///
    /// Returns the `Id` of the written element.
    fn append<T: SlabItem + Default>(&mut self, t: &T) -> Id<T> {
        let id = self.allocate::<T>();
        // IGNORED: safe because we just allocated the id
        let _ = self.write(id, t);
        id
    }

    /// Append a slice to the end of the buffer, resizing if necessary
    /// and returning a slabbed array.
    ///
    /// Returns the `Array` of the written elements.
    fn append_array<T: SlabItem + Default>(&mut self, ts: &[T]) -> Array<T> {
        let array = self.allocate_array::<T>(ts.len());
        // IGNORED: safe because we just allocated the array
        let _ = self.write_array(array, ts);
        array
    }
}

/// A wrapper around a [`GrowableSlab`] that provides convenience methods for
/// working with CPU-side slabs.
///
/// Working with slabs on the CPU is much more convenient because the underlying
/// buffer `B` is often a growable type, like `Vec<u32>`. This wrapper provides
/// methods for appending to the end of the buffer with automatic resizing and
/// for preallocating space for elements that will be written later.
pub struct CpuSlab<B> {
    slab: B,
}

impl<B> AsRef<B> for CpuSlab<B> {
    fn as_ref(&self) -> &B {
        &self.slab
    }
}

impl<B> AsMut<B> for CpuSlab<B> {
    fn as_mut(&mut self) -> &mut B {
        &mut self.slab
    }
}

impl<B: Slab> Slab for CpuSlab<B> {
    fn len(&self) -> usize {
        self.slab.len()
    }

    fn read<T: SlabItem + Default>(&self, id: Id<T>) -> T {
        self.slab.read(id)
    }

    fn write_indexed<T: SlabItem>(&mut self, t: &T, index: usize) -> usize {
        self.slab.write_indexed(t, index)
    }

    fn write_indexed_slice<T: SlabItem>(&mut self, t: &[T], index: usize) -> usize {
        self.slab.write_indexed_slice(t, index)
    }
}

impl<B: GrowableSlab> GrowableSlab for CpuSlab<B> {
    fn capacity(&self) -> usize {
        self.slab.capacity()
    }

    fn reserve_capacity(&mut self, capacity: usize) {
        self.slab.reserve_capacity(capacity);
    }

    fn increment_len(&mut self, n: usize) -> usize {
        self.slab.increment_len(n)
    }
}

impl<B: GrowableSlab> CpuSlab<B> {
    /// Create a new `SlabBuffer` with the given slab.
    pub fn new(slab: B) -> Self {
        Self { slab }
    }

    /// Consume the [`CpuSlab`], converting it into the underlying buffer.
    pub fn into_inner(self) -> B {
        self.slab
    }
}

#[cfg(not(target_arch = "spirv"))]
impl GrowableSlab for Vec<u32> {
    fn capacity(&self) -> usize {
        Vec::capacity(self)
    }

    fn reserve_capacity(&mut self, capacity: usize) {
        Vec::reserve(self, capacity - self.capacity());
    }

    fn increment_len(&mut self, n: usize) -> usize {
        let index = self.len();
        self.extend(core::iter::repeat(0).take(n));
        index
    }
}

#[cfg(test)]
mod test {
    use glam::Vec4;

    use crate::{self as crabslab, Array, CpuSlab, SlabItem};

    use super::*;

    #[derive(Debug, Default, PartialEq, SlabItem)]
    struct Vertex {
        position: Vec4,
        color: Vec4,
        uv: glam::Vec2,
    }

    #[test]
    fn slab_array_readwrite() {
        let mut slab = [0u32; 16];
        slab.write_indexed(&42, 0);
        slab.write_indexed(&666, 1);
        let t = slab.read(Id::<[u32; 2]>::new(0));
        assert_eq!([42, 666], t);
        let t: Vec<u32> = slab.read_vec(Array::new(0, 2));
        assert_eq!([42, 666], t[..]);
        slab.write_indexed_slice(&[1, 2, 3, 4], 2);
        let t: Vec<u32> = slab.read_vec(Array::new(2, 4));
        assert_eq!([1, 2, 3, 4], t[..]);

        // use _f32 explicit, otherwise it fails
        slab.write_indexed_slice(&[[1.0_f32, 2.0, 3.0, 4.0], [5.5, 6.5, 7.5, 8.5]], 0);

        let arr = Array::<[f32; 4]>::new(0, 2);
        assert_eq!(Id::new(0), arr.at(0));
        assert_eq!(Id::new(4), arr.at(1));
        assert_eq!([1.0, 2.0, 3.0, 4.0], slab.read(arr.at(0)));
        assert_eq!([5.5, 6.5, 7.5, 8.5], slab.read(arr.at(1)));

        let geometry = vec![
            Vertex {
                position: Vec4::new(0.5, -0.5, 0.0, 1.0),
                color: Vec4::new(1.0, 0.0, 0.0, 1.0),
                ..Default::default()
            },
            Vertex {
                position: Vec4::new(0.0, 0.5, 0.0, 1.0),
                color: Vec4::new(0.0, 1.0, 0.0, 1.0),
                ..Default::default()
            },
            Vertex {
                position: Vec4::new(-0.5, -0.5, 0.0, 1.0),
                color: Vec4::new(0.0, 0.0, 1.0, 1.0),
                ..Default::default()
            },
            Vertex {
                position: Vec4::new(-1.0, 1.0, 0.0, 1.0),
                color: Vec4::new(1.0, 0.0, 0.0, 1.0),
                ..Default::default()
            },
            Vertex {
                position: Vec4::new(-1.0, 0.0, 0.0, 1.0),
                color: Vec4::new(0.0, 1.0, 0.0, 1.0),
                ..Default::default()
            },
            Vertex {
                position: Vec4::new(0.0, 1.0, 0.0, 1.0),
                color: Vec4::new(0.0, 0.0, 1.0, 1.0),
                ..Default::default()
            },
        ];
        let geometry_slab_size = Vertex::SLAB_SIZE * geometry.len();
        let mut slab = vec![0u32; geometry_slab_size + Array::<Vertex>::SLAB_SIZE];
        let index = 0usize;
        let vertices = Array::<Vertex>::new(index as u32, geometry.len() as u32);
        let index = slab.write_indexed_slice(&geometry, index);
        assert_eq!(geometry_slab_size, index);
        let vertices_id = Id::<Array<Vertex>>::from(index);
        let index = slab.write_indexed(&vertices, index);
        assert_eq!(geometry_slab_size + Array::<Vertex>::SLAB_SIZE, index);
        assert_eq!(Vertex::SLAB_SIZE * 6, vertices_id.index());
        assert!(slab.contains(vertices_id),);

        let array = slab.read(vertices_id);
        assert_eq!(vertices, array);
    }

    #[test]
    fn cpuslab_sanity() {
        let mut slab = CpuSlab::new(vec![]);
        let v = Vertex {
            position: Vec4::new(0.5, -0.5, 0.0, 1.0),
            color: Vec4::new(1.0, 0.0, 0.0, 1.0),
            ..Default::default()
        };
        let id = slab.append(&v);
        assert_eq!(Id::new(0), id);
        assert_eq!(v, slab.read(id));

        let f32s = [1.1, 2.2, 3.3, 4.4f32];
        let array = slab.append_array(&f32s);
        assert_eq!(1.1, slab.read(array.at(0)));
        assert_eq!(2.2, slab.read(array.at(1)));
        assert_eq!(3.3, slab.read(array.at(2)));
        assert_eq!(4.4, slab.read(array.at(3)));

        let f32_vec = slab.read_vec(array);
        assert_eq!(f32s, f32_vec[..]);
    }

    #[test]
    fn tuples_and_all_primitives() {
        let mut slab = CpuSlab::new(vec![]);
        let buffer1 = (-5_i8, 5u8, -5_i16, 5u16, -5_i32, 5u32);
        let buffer2 = (-5_i64, 5u64, -5_i128, 5u128, false, 1.0_f32, 1.0_f64);

        let id1 = slab.append(&buffer1);
        let id2 = slab.append(&buffer2);

        assert_eq!(buffer1, slab.read(id1));
        assert_eq!(buffer2, slab.read(id2));
    }
}

#[cfg(test)]
mod blah {
    use crate as crabslab;
    use crate::*;

    #[test]
    fn derive_baz_sanity() {
        #[derive(Debug, Default, PartialEq, SlabItem)]
        pub struct Bar {
            a: u32,
        }

        #[derive(Debug, Default, PartialEq, SlabItem)]
        enum Baz {
            #[default]
            One,
            Two {
                a: u32,
                b: u32,
            },
            Three(u32, u32),
            Four(Bar),
        }

        assert_eq!(3, Baz::SLAB_SIZE);

        let mut slab = CpuSlab::new(vec![]);

        let one_id = slab.append(&Baz::One);
        let two_id = slab.append(&Baz::Two { a: 1, b: 2 });
        let three_id = slab.append(&Baz::Three(3, 4));
        let four_id = slab.append(&Baz::Four(Bar { a: 5 }));

        assert_eq!(Baz::One, slab.read(one_id));
        assert_eq!(Baz::Two { a: 1, b: 2 }, slab.read(two_id));
        assert_eq!(Baz::Three(3, 4), slab.read(three_id));
        assert_eq!(Baz::Four(Bar { a: 5 }), slab.read(four_id));
    }
}
