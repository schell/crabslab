//! Managing ranges of values.

use std::{
    ops::{Add, Index, IndexMut},
    slice::SliceIndex,
};

use crabslab::{Array, Id, SlabItem};

use crate::update::{SourceId, Update};

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

impl From<Range> for std::ops::Range<usize> {
    fn from(value: Range) -> Self {
        value.first_index as usize..value.last_index as usize + 1
    }
}

macro_rules! from_unsigned_ranges_impl {
    ($ty:ty) => {
        impl From<std::ops::Range<$ty>> for Range {
            fn from(value: std::ops::Range<$ty>) -> Self {
                Range {
                    first_index: value.start as u32,
                    last_index: value.end as u32 - 1,
                }
            }
        }

        impl From<std::ops::RangeInclusive<$ty>> for Range {
            fn from(value: std::ops::RangeInclusive<$ty>) -> Self {
                Range {
                    first_index: *value.start() as u32,
                    last_index: *value.end() as u32,
                }
            }
        }
    };
}
from_unsigned_ranges_impl!(usize);
from_unsigned_ranges_impl!(u32);
from_unsigned_ranges_impl!(u16);
from_unsigned_ranges_impl!(u8);

impl From<std::ops::RangeFull> for Range {
    fn from(_: std::ops::RangeFull) -> Self {
        Range {
            first_index: 0,
            last_index: u32::MAX,
        }
    }
}

impl<T> Index<Range> for [T] {
    type Output = [T];

    fn index(&self, index: Range) -> &Self::Output {
        &self[index.first_index as usize..=index.last_index as usize]
    }
}

impl<T> IndexMut<Range> for [T] {
    fn index_mut(&mut self, index: Range) -> &mut Self::Output {
        &mut self[index.first_index as usize..=index.last_index as usize]
    }
}

impl<T> Index<Range> for Vec<T> {
    type Output = [T];

    fn index(&self, index: Range) -> &Self::Output {
        &self[index.first_index as usize..=index.last_index as usize]
    }
}

impl<T> IndexMut<Range> for Vec<T> {
    fn index_mut(&mut self, index: Range) -> &mut Self::Output {
        &mut self[index.first_index as usize..=index.last_index as usize]
    }
}

impl Range {
    pub const DEGENERATE: Range = Range {
        first_index: u32::MAX,
        last_index: 0,
    };

    pub fn len(&self) -> u32 {
        if self.is_degenerate() {
            0
        } else {
            1 + (self.last_index - self.first_index)
        }
    }

    pub fn is_empty(&self) -> bool {
        self.is_degenerate()
    }

    pub fn is_degenerate(&self) -> bool {
        self.last_index < self.first_index
    }

    pub fn intersects(&self, other: &Range) -> bool {
        !self.is_degenerate()
            && self.first_index <= other.last_index
            && self.last_index >= other.first_index
    }

    /// Nullifies this range, making it a zero range.
    pub fn nullify(&mut self) {
        self.first_index = u32::MAX;
        self.last_index = 0;
    }

    /// Offset this range by the given range.
    ///
    /// This returns a `Range` that is effectively `self`, but
    /// local to `rhs`.
    pub fn offset(&self, rhs: &Self) -> Range {
        Range {
            first_index: self.first_index - rhs.first_index,
            last_index: self.last_index - rhs.first_index,
        }
    }
}

/// Represents a block of contiguous numbers.
pub trait IsRange: Sized {
    /// Returns the inner `Range`.
    fn range(&self) -> &Range;

    /// Returns the inner `Range`, mutably.
    fn range_mut(&mut self) -> &mut Range;

    /// Returns `true` if `self` is the left neighbor of `rhs`.
    fn is_left_neighbor_of(&self, other: &Self) -> bool;

    /// Returns `true` if `self` contains `other`.
    fn contains(&self, other: &Self) -> bool;

    /// Returns the union of two contiguous ranges.
    fn merge_with_right_neighbor(&mut self, rhs: Self);

    /// Returns `Some` containing the overlapping range
    /// if `self` overlaps `rhs`, but returns `false` if they are contiguous or
    /// separated or either are degenerate.
    fn overlap(&self, rhs: &Self) -> Option<Range> {
        let here = self.range();
        let there = rhs.range();
        (!here.is_degenerate()
            && !there.is_degenerate()
            && here.first_index <= there.last_index
            && here.last_index >= there.first_index)
            .then_some(())?;
        Some(if here.first_index < there.first_index {
            Range {
                first_index: there.first_index,
                last_index: here.last_index,
            }
        } else {
            Range {
                first_index: here.first_index,
                last_index: there.last_index,
            }
        })
    }

    /// Returns `true` if `self` is contiguous with `rhs`, but returns `false` if they
    /// overlap or are separated or either are degenerate.
    fn contiguous(&self, rhs: &Self) -> bool {
        !self.range().is_degenerate()
            && !rhs.range().is_degenerate()
            && (self.range().last_index == rhs.range().first_index + 1
                || self.range().first_index == rhs.range().last_index + 1)
    }

    fn split_off(&mut self, n: u32) -> Self;

    /// Take the first `n` spaces from `self`, returning it as a new range.
    /// The remainder stays in place in `self`.
    ///
    /// `self` must contain at least `n + 1` spaces or this will leave `self`
    /// degenerate.
    fn take(&mut self, n: u32) -> Self {
        let len = self.range().len();
        let right = self.split_off(len - n);
        std::mem::replace(self, right)
    }

    /// Drop the last `n` spaces from the right of `self`, returning them.
    ///
    /// `self` must contain at least `n + 1` spaces or this will leave `self`
    /// degenerate.
    fn drop(&mut self, n: u32) -> Self {
        self.split_off(n)
    }

    fn merge_contained(&mut self, contained: Self);
}

impl IsRange for Range {
    fn range(&self) -> &Range {
        self
    }

    fn range_mut(&mut self) -> &mut Range {
        self
    }

    fn is_left_neighbor_of(&self, other: &Self) -> bool {
        self.last_index + 1 == other.first_index
    }

    fn contains(&self, other: &Self) -> bool {
        self.first_index <= other.first_index && self.last_index >= other.last_index
    }

    fn merge_with_right_neighbor(&mut self, rhs: Self) {
        self.last_index = rhs.last_index;
    }

    fn split_off(&mut self, n: u32) -> Self {
        let last_index = self.last_index;
        self.last_index -= n;
        Range {
            first_index: self.last_index + 1,
            last_index,
        }
    }

    fn merge_contained(&mut self, contained: Self) {
        debug_assert!(self.contains(&contained));
    }
}

impl IsRange for SourceId {
    fn range(&self) -> &Range {
        &self.range
    }

    fn range_mut(&mut self) -> &mut Range {
        &mut self.range
    }

    fn is_left_neighbor_of(&self, other: &Self) -> bool {
        self.range.is_left_neighbor_of(&other.range)
    }

    fn contains(&self, other: &Self) -> bool {
        self.range.contains(&other.range)
    }

    fn merge_with_right_neighbor(&mut self, rhs: Self) {
        self.range.merge_with_right_neighbor(rhs.range);
        if self.type_is != rhs.type_is {
            self.type_is = "_";
        }
    }

    fn split_off(&mut self, n: u32) -> Self {
        let range = self.range_mut().split_off(n);
        SourceId {
            range,
            type_is: self.type_is,
        }
    }

    fn merge_contained(&mut self, contained: Self) {
        debug_assert!(self.contains(&contained));
    }
}

impl IsRange for Update {
    fn range(&self) -> &Range {
        &self.range
    }

    fn range_mut(&mut self) -> &mut Range {
        &mut self.range
    }

    fn is_left_neighbor_of(&self, other: &Self) -> bool {
        self.range.is_left_neighbor_of(&other.range)
    }

    fn contains(&self, other: &Self) -> bool {
        self.range.contains(&other.range)
    }

    fn merge_with_right_neighbor(&mut self, rhs: Self) {
        self.range.merge_with_right_neighbor(rhs.range);
        self.data.extend(rhs.data);
    }

    fn merge_contained(&mut self, contained: Self) {
        debug_assert!(self.contains(&contained));
        let contained_local_to_self = contained.range().offset(self.range());
        self.data[contained_local_to_self].copy_from_slice(&contained.data);
    }

    fn split_off(&mut self, n: u32) -> Self {
        let range = self.range.split_off(n);
        let data = self.data.split_off(n as usize);
        Self { range, data }
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

impl<R: IsRange + std::fmt::Debug> RangeManager<R> {
    /// Return the number of distinct ranges being managed.
    pub fn len(&self) -> usize {
        self.ranges.len()
    }

    /// Return whether this manager is managing any ranges.
    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }

    pub fn add_range(&mut self, input_range: R) {
        println!("  adding range: {input_range:?}");
        let mut maybe_input_range = Some(input_range);
        let ranges = std::mem::take(&mut self.ranges);
        // Place the input range correctly within the existing ranges, either by
        // merging it with another range, inserting between, or appending to the
        // end.
        let mut ranges = ranges
            .into_iter()
            .flat_map(|mut existing_range| {
                println!("  existing range: {existing_range:?}");
                if let Some(mut input_range) = maybe_input_range.take() {
                    println!("    input range: {input_range:?}");
                    if let Some(overlap) = existing_range.overlap(&input_range) {
                        println!("    overlap: {overlap:?}");
                        if input_range.range().contains(existing_range.range()) {
                            // Return nothing, the existing_range dissappears as input_range would overwrite
                            // it, but carries on to possibly overlap other ranges.
                            println!("      input_range overwrites existing_range");
                            maybe_input_range = Some(input_range);
                            vec![]
                        } else if existing_range.contains(&input_range) {
                            // The existing_range consumes the input_range, overwriting the overlapping
                            // portion of its data with that of input_range.
                            println!("      existing_range contains input_range");
                            existing_range.merge_contained(input_range);
                            vec![existing_range]
                        } else
                        // If we've made it this far it means the overlap is partial
                        if input_range.range().first_index
                            < existing_range.range().first_index
                        {
                            println!("      inserting input_range before existing_range");
                            // Reduce the existing_range, as the input range overwrites it
                            existing_range.take(overlap.len());
                            // Then insert the input_range before the existing_range.
                            // They are contiguous and will be joined in the next call to `.defrag()`
                            vec![input_range, existing_range]
                        } else {
                            println!("      overwriting right portion of existing_range and carrying the remainder");
                            // Split off the overlapping portion of input_range and merge it into
                            // existing_range, then put the remainder back to check against the
                            // other ranges.
                            let overlapping = input_range.take(overlap.len());
                            existing_range.merge_contained(overlapping);
                            if !input_range.range().is_empty() {
                                maybe_input_range = Some(input_range);
                            }
                            vec![existing_range]
                        }
                    } else {
                        if input_range.range().first_index < existing_range.range().first_index {
                            println!("      non-overlapping insertion before existing");
                            vec![input_range, existing_range]
                        } else {
                            maybe_input_range = Some(input_range);
                            vec![existing_range]
                        }
                    }
                } else {
                    vec![existing_range]
                }
            })
            .collect::<Vec<_>>();
        if let Some(input_range) = maybe_input_range {
            println!("  append input range to the end");
            ranges.push(input_range);
        }

        self.ranges = ranges;
    }

    /// Defragment the ranges.
    ///
    /// This is O(2n).
    pub fn defrag(mut self) -> Self {
        let mut ranges = std::mem::take(&mut self.ranges);
        for i in 0..ranges.len() {
            let (mut left, mut right) = ranges.split_at_mut(i);
            let left = left.split_off_last_mut();
            let right = right.split_off_first_mut();
            match (left, right) {
                (None, None) => {
                    // Empty
                }
                (None, Some(_)) => {
                    // This is the first element, nothing to do
                }
                (Some(_), None) => {
                    // This is the last element, nothing to do
                }
                (Some(left), Some(right)) => {
                    // if left.overlaps(right) {
                    //     // Merge the left into the right, then zero out the left
                    //     right.eat_merge(left);
                    //     left.range_mut().nullify();
                    // }
                }
            }
        }
        self.ranges = ranges
            .into_iter()
            .filter(|r| !r.range().is_degenerate())
            .collect();
        self
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

#[cfg(test)]
mod test {
    use proptest::{
        prelude::{Just, Strategy},
        proptest,
    };

    use super::*;

    #[test]
    fn split_off_sanity() {
        let mut range = Range::from(0u32..10);
        let right = range.split_off(5);
        assert_eq!(Range::from(0u32..5), range);
        assert_eq!(Range::from(5u32..10), right);
    }

    #[test]
    fn range_take_and_drop_sanity() {
        let mut right = Range::from(0u32..=9);
        let left = right.take(4);
        assert_eq!(Range::from(0u32..=3), left);
        assert_eq!(Range::from(4u32..=9), right);

        let mut left = Range::from(0u32..=9);
        let right = left.drop(4);
        assert_eq!(Range::from(0u32..=5), left);
        assert_eq!(Range::from(6u32..=9), right);
    }

    // #[test]
    // fn merge_overlapping_updates() {
    //     let a = Update {
    //         range: (0u32..=4).into(),
    //         data: vec![0u32; 5],
    //     };
    //     let b = Update {
    //         range: (4u32..=7).into(),
    //         data: vec![1u32; 4],
    //     };
    //     assert!(a.overlaps(&b));
    //     let mut update = a.clone();
    //     update.eat_merge(&b);
    //     assert_eq!(
    //         Update {
    //             range: (0u32..=7).into(),
    //             data: vec![0, 0, 0, 0, 1, 1, 1, 1]
    //         },
    //         update
    //     );

    //     update.eat_merge(&Update {
    //         range: (3u32..=4).into(),
    //         data: vec![2, 2],
    //     });
    //     assert_eq!(
    //         Update {
    //             range: (0u32..=7).into(),
    //             data: vec![0, 0, 0, 2, 2, 1, 1, 1]
    //         },
    //         update
    //     );
    // }

    fn arb_range(slab_length: u32, max_range_length: u32) -> impl Strategy<Value = Range> {
        (1..max_range_length)
            .prop_flat_map(move |range_length| {
                ((range_length - 1)..slab_length, Just(range_length))
            })
            .prop_map(|(last_index, length)| Range {
                first_index: last_index - (length - 1),
                last_index,
            })
    }

    fn assert_ranges_are_ordered(ranges: &[Range]) {
        for i in 0..ranges.len() {
            let (mut left, mut right) = ranges.split_at(i);
            let left = left.split_off_last();
            let right = right.split_off_first();
            match (left, right) {
                (None, None) => {
                    // Empty
                }
                (None, Some(_)) => {
                    // This is the first element, nothing to do
                }
                (Some(_), None) => {
                    // This is the last element, nothing to do
                }
                (Some(left), Some(right)) => {
                    assert!(
                        left.range().first_index < right.range().first_index,
                        "ranges are not ordered, {i} {left:?} !< {right:?}\nranges: {:#?}",
                        ranges
                    );
                }
            }
        }
    }

    fn assert_ranges_are_non_overlapping(ranges: &Vec<Range>) {
        for i in 0..ranges.len() {
            let (mut left, mut right) = ranges.split_at(i);
            let left = left.split_off_last();
            let right = right.split_off_first();
            match (left, right) {
                (None, None) => {
                    // Empty
                }
                (None, Some(_)) => {
                    // This is the first element, nothing to do
                }
                (Some(_), None) => {
                    // This is the last element, nothing to do
                }
                (Some(left), Some(right)) => {
                    // assert!(!left.overlaps(right), "ranges overlap: {left:?} {right:?}");
                }
            }
        }
    }

    fn add_ranges(manager: &mut RangeManager<Range>, ranges: impl IntoIterator<Item = Range>) {
        for range in ranges {
            manager.add_range(range);
            assert_ranges_are_non_overlapping(&manager.ranges);
            assert_ranges_are_ordered(&manager.ranges);
        }
    }

    proptest! {
        #[test]
        fn proptest_range_manager_sanity(ranges in proptest::collection::vec(arb_range(1024, 128), 32)) {
            let mut manager = RangeManager::default();
            add_ranges(&mut manager, ranges);
            // let manager = manager.defrag();
        }
    }

    #[test]
    fn regression() {
        let ranges = vec![
            Range::from(0u32..=0),
            Range::from(0u32..=0),
            Range::from(0u32..=0),
            Range::from(0u32..=0),
            Range::from(0u32..=0),
            Range::from(0u32..=0),
            Range::from(0u32..=0),
            Range::from(0u32..=0),
            Range::from(0u32..=0),
            Range::from(0u32..=0),
            Range::from(0u32..=0),
            Range::from(0u32..=0),
            Range::from(0u32..=0),
            Range::from(0u32..=0),
            Range::from(0u32..=0),
            Range::from(0u32..=4),
            Range::from(623u32..=671),
            Range::from(559u32..=575),
            Range::from(125u32..=218),
            Range::from(453u32..=468),
            Range::from(664u32..=710),
            Range::from(463u32..=535),
            Range::from(435u32..=505),
            Range::from(777u32..=836),
            Range::from(240u32..=292),
            Range::from(562u32..=631),
            Range::from(63u32..=169),
            Range::from(40u32..=165),
            Range::from(895u32..=945),
            Range::from(645u32..=720),
            Range::from(225u32..=330),
            Range::from(266u32..=328),
        ];

        let mut manager = RangeManager::default();
        add_ranges(&mut manager, ranges);
    }
}
