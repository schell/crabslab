//! Managing ranges of values.

use crabslab::{Array, Id, SlabItem};

use crate::{
    runtime::SlabUpdate,
    update::{SourceId, Update},
};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Range {
    pub first_index: u32,
    pub last_index: u32,
}

impl core::fmt::Debug for Range {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&format!("{}..={}", self.first_index, self.last_index))
    }
}

impl<T: SlabItem> From<Array<T>> for Range {
    fn from(array: Array<T>) -> Self {
        let array = array.into_u32_array();
        let first_index = array.starting_index() as u32;
        Range {
            first_index,
            last_index: first_index + array.len() as u32 - 1,
        }
    }
}

impl<T: SlabItem> From<Id<T>> for Range {
    fn from(id: Id<T>) -> Self {
        Range {
            first_index: id.inner(),
            last_index: id.inner() + T::SLAB_SIZE as u32 - 1,
        }
    }
}

impl Range {
    pub fn len(&self) -> u32 {
        1 + self.last_index - self.first_index
    }

    pub fn is_empty(&self) -> bool {
        self.last_index == self.first_index
    }

    pub fn intersects(&self, other: &Range) -> bool {
        !(self.first_index > other.last_index || self.last_index < other.first_index)
    }
}

/// Represents a block of contiguous numbers.
pub trait IsRange {
    /// Returns `true` if `self` is the left neighbor of `rhs`.
    fn is_left_neighbor_of(&self, other: &Self) -> bool;

    /// Returns the union of two ranges.
    fn merge_with_right_neighbor(&mut self, rhs: Self);
}

impl IsRange for Range {
    fn is_left_neighbor_of(&self, other: &Self) -> bool {
        debug_assert!(
            !self.intersects(other),
            "{self:?} intersects existing {other:?}, should never happen with Range"
        );

        self.last_index + 1 == other.first_index
    }

    fn merge_with_right_neighbor(&mut self, rhs: Self) {
        self.last_index = rhs.last_index;
    }
}

impl IsRange for SourceId {
    fn is_left_neighbor_of(&self, other: &Self) -> bool {
        self.range.is_left_neighbor_of(&other.range)
    }

    fn merge_with_right_neighbor(&mut self, rhs: Self) {
        self.range.merge_with_right_neighbor(rhs.range);
        if self.type_is != rhs.type_is {
            self.type_is = "_";
        }
    }
}

impl IsRange for Update {
    fn is_left_neighbor_of(&self, other: &Self) -> bool {
        self.range.is_left_neighbor_of(&other.range)
    }

    fn merge_with_right_neighbor(&mut self, rhs: Self) {
        self.range.merge_with_right_neighbor(rhs.range);
        self.data.extend(rhs.data);
    }
}

impl IsRange for SlabUpdate {
    fn is_left_neighbor_of(&self, other: &Self) -> bool {
        self.intersects(other)
    }

    fn merge_with_right_neighbor(&mut self, rhs: Self) {
        if self.array == rhs.array {
            *self = rhs;
            return;
        }

        self.array.union(&rhs.array);
        self.elements.extend(rhs.elements);
    }
}

/// Manages contiguous ranges.
pub struct RangeManager<R> {
    /// The contiguous ranges managed by this manager.
    ///
    /// ## Note
    /// Keep in mind that the ranges within this vector are not guaranteed to
    /// be sorted.
    pub ranges: Vec<R>,
}

impl<R> Default for RangeManager<R> {
    fn default() -> Self {
        Self { ranges: vec![] }
    }
}

impl<R: core::fmt::Debug> core::fmt::Debug for RangeManager<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct(&format!("RangeManager<{}>", std::any::type_name::<R>()))
            .field("ranges", &self.ranges)
            .finish()
    }
}

impl<R: IsRange> RangeManager<R> {
    /// Return the number of distinct ranges being managed.
    pub fn len(&self) -> usize {
        self.ranges.len()
    }

    /// Return whether this manager is managing any ranges.
    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }

    pub fn add_range(&mut self, input_range: R) {
        for range in self.ranges.iter_mut() {
            if range.is_left_neighbor_of(&input_range) {
                range.merge_with_right_neighbor(input_range);
                return;
            }
            if input_range.is_left_neighbor_of(range) {
                let rhs = std::mem::replace(range, input_range);
                range.merge_with_right_neighbor(rhs);
                return;
            }
        }
        self.ranges.push(input_range);
    }

    pub fn defrag(self) -> Self {
        let mut defragged = Self::default();
        for range in self.ranges.into_iter() {
            defragged.add_range(range);
        }
        defragged
    }
}

impl RangeManager<Range> {
    /// Removes a range of `count` elements, if possible.
    ///
    /// If `Some` was returned, either a `Range` was found that was
    /// exactly of size `count`, or a `count` spaces was removed from an existing `Range`.
    pub fn remove(&mut self, count: u32) -> Option<Range> {
        let mut remove_index = usize::MAX;
        for (i, range) in self.ranges.iter_mut().enumerate() {
            // This is potentially a hot path, so use the `if` even
            // though clippy complains (because using match is slower)
            #[allow(clippy::comparison_chain)]
            if range.len() > count {
                let first_index = range.first_index;
                let last_index = range.first_index + count - 1;
                range.first_index += count;
                return Some(Range {
                    first_index,
                    last_index,
                });
            } else if range.len() == count {
                remove_index = i;
                break;
            }
        }

        if remove_index == usize::MAX {
            None
        } else {
            Some(self.ranges.swap_remove(remove_index))
        }
    }
}
