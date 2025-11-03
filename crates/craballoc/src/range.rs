//! Managing ranges of values.

use std::ops::{Index, IndexMut};

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

    /// Returns `true` if `self` contains `other`.
    fn contains(&self, other: &Self) -> bool {
        self.first_index <= other.first_index && self.last_index >= other.last_index
    }

    /// Returns `Some` containing the overlapping range
    /// if `self` overlaps `rhs`, but returns `false` if they are contiguous or
    /// separated or either are degenerate.
    fn overlap(&self, rhs: &Self) -> Option<Range> {
        (!self.is_degenerate()
            && !rhs.is_degenerate()
            && self.first_index <= rhs.last_index
            && self.last_index >= rhs.first_index)
            .then_some(())?;
        Some(if self.first_index < rhs.first_index {
            Range {
                first_index: rhs.first_index,
                last_index: self.last_index,
            }
        } else {
            Range {
                first_index: self.first_index,
                last_index: rhs.last_index,
            }
        })
    }
}

/// Represents a block of contiguous numbers.
pub trait IsRange: Sized {
    /// Returns the inner `Range`.
    fn range(&self) -> &Range;

    /// Returns the inner `Range`, mutably.
    fn range_mut(&mut self) -> &mut Range;

    /// Returns the union of two contiguous ranges.
    fn merge_with_right_neighbor(&mut self, rhs: Self);

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
        debug_assert!(self.range().contains(contained.range()));
    }
}

impl IsRange for Update {
    fn range(&self) -> &Range {
        &self.range
    }

    fn range_mut(&mut self) -> &mut Range {
        &mut self.range
    }

    fn merge_with_right_neighbor(&mut self, rhs: Self) {
        self.range.merge_with_right_neighbor(rhs.range);
        self.data.extend(rhs.data);
    }

    fn merge_contained(&mut self, contained: Self) {
        debug_assert!(self.range().contains(contained.range()));
        let contained_local_to_self = contained.range().offset(self.range());
        self.data[contained_local_to_self].copy_from_slice(&contained.data);
    }

    fn split_off(&mut self, n: u32) -> Self {
        let range = self.range.split_off(n);
        let data = self.data.split_off(n as usize);
        Self { range, data }
    }
}

struct RangeAccumulator<R> {
    visited_ranges: Vec<R>,
    maybe_input_range: Option<R>,
}

impl<R: IsRange + std::fmt::Debug> RangeAccumulator<R> {
    /// Merge the last gap in the ranges, if possible.
    fn merge_last_gap_if_possible(&mut self) {
        log::trace!("      attempting to merge a gap");
        if let Some(right) = self.visited_ranges.pop() {
            log::trace!("        right {right:?}");
            if let Some(mut left) = self.visited_ranges.pop() {
                log::trace!("        left {left:?}");
                if left.range().last_index + 1 == right.range().first_index {
                    log::trace!("      merging contiguous {left:?} {right:?}");
                    left.merge_with_right_neighbor(right);
                    log::trace!("        {left:?}");
                    self.visited_ranges.push(left);
                    return;
                } else {
                    log::trace!("        not contiguous");
                }
                self.visited_ranges.push(left);
            }
            self.visited_ranges.push(right);
        }
    }

    /// Step a fold over ranges.
    fn step(mut self, mut existing_range: R) -> Self {
        // Place the input range correctly within the existing ranges, either by
        // merging it with another range, inserting between, or appending to the
        // end.
        log::trace!("  existing range: {existing_range:?}");
        if let Some(mut input_range) = self.maybe_input_range.take() {
            log::trace!("    input range: {input_range:?}");
            if let Some(overlap) = existing_range.range().overlap(input_range.range()) {
                log::trace!("    overlap: {overlap:?}");
                if input_range.range().contains(existing_range.range()) {
                    // Return nothing, the existing_range dissappears as input_range would overwrite
                    // it, but carries on to possibly overlap other ranges.
                    log::trace!("      input_range overwrites existing_range");
                    self.maybe_input_range = Some(input_range);
                } else if existing_range.range().contains(input_range.range()) {
                    // The existing_range consumes the input_range, overwriting the overlapping
                    // portion of its data with that of input_range.
                    log::trace!("      existing_range contains input_range");
                    existing_range.merge_contained(input_range);
                    self.visited_ranges.push(existing_range);
                } else
                // If we've made it this far it means the overlap is partial
                if input_range.range().first_index < existing_range.range().first_index {
                    log::trace!("      inserting input_range before existing_range");
                    // Reduce the existing_range, as the input range overwrites it
                    existing_range.take(overlap.len());
                    // They are contiguous, join them and insert into the ranges
                    input_range.merge_with_right_neighbor(existing_range);
                    self.visited_ranges.push(input_range);
                    // Check the last two ranges for continuity
                    self.merge_last_gap_if_possible();
                } else {
                    log::trace!("      overwriting right portion of existing_range and carrying the remainder");
                    // Split off the overlapping portion of input_range and merge it into
                    // existing_range, then put the remainder back to check against the
                    // other ranges.
                    let overlapping = input_range.take(overlap.len());
                    existing_range.merge_contained(overlapping);
                    if !input_range.range().is_empty() {
                        self.maybe_input_range = Some(input_range);
                    }
                    self.visited_ranges.push(existing_range);
                }
            } else if input_range.range().first_index < existing_range.range().first_index {
                log::trace!("      non-overlapping insertion before existing");
                if input_range.range().last_index + 1 == existing_range.range().first_index {
                    log::trace!("      input_range is contiguous with existing_range, merging");
                    input_range.merge_with_right_neighbor(existing_range);
                    self.visited_ranges.push(input_range);
                    self.merge_last_gap_if_possible();
                } else {
                    self.visited_ranges.push(input_range);
                    self.merge_last_gap_if_possible();
                    self.visited_ranges.push(existing_range);
                }
            } else {
                self.maybe_input_range = Some(input_range);
                self.visited_ranges.push(existing_range);
            }
        } else {
            self.visited_ranges.push(existing_range);
        }
        self
    }

    /// Finish accumulation, returning the ranges.
    fn finish(mut self) -> Vec<R> {
        if let Some(input_range) = self.maybe_input_range.take() {
            log::trace!("  append input range to the end");
            self.visited_ranges.push(input_range);
            if self.visited_ranges.len() > 1 {
                self.merge_last_gap_if_possible();
            }
        }
        self.visited_ranges
    }
}

/// Manages contiguous ranges.
pub struct RangeManager<R> {
    /// The sorted, non-overlapping, disjoint ranges managed by this manager.
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

    /// Add a range to the manager.
    ///
    /// Internally the manager will merge the range and maintain a non-overlapping,
    /// disjoint and ordered list of ranges. This results in the smallest possible number of
    /// updates when used to manage buffer updates.
    ///
    /// O(n) worst case time complexity. In actuality it ends up being quite a bit better than
    /// that because the ranges are coalesced into disjoint regions. So `n` is proportional to
    /// the sparcity of the input ranges.
    pub fn insert(&mut self, input_range: R) {
        log::trace!("  inserting range: {input_range:?}");
        let acc = RangeAccumulator {
            visited_ranges: vec![],
            maybe_input_range: Some(input_range),
        };
        let ranges = std::mem::take(&mut self.ranges);
        self.ranges = ranges
            .into_iter()
            .fold(acc, RangeAccumulator::step)
            .finish();
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
                        "ranges are not ordered: {i} {left:?} !< {right:?}\nranges: {:#?}",
                        ranges
                    );
                }
            }
        }
    }

    fn assert_ranges_are_non_overlapping(ranges: &[Range]) {
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
                    let overlap = left.overlap(right);
                    assert!(
                        overlap.is_none(),
                        "ranges overlap: {i} {left:?} {right:?} on {:?}",
                        overlap.unwrap()
                    );
                }
            }
        }
    }

    /// During insertion any overlapping or contiguous ranges should have been merged,
    /// meaning we only have disjoint ranges.
    fn assert_ranges_are_disjoint(ranges: &[Range]) {
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
                    assert_ne!(
                        left.last_index + 1,
                        right.first_index,
                        "ranges are contiguous: {i} {left:?} {right:?}"
                    );
                }
            }
        }
    }

    fn add_ranges(manager: &mut RangeManager<Range>, ranges: impl IntoIterator<Item = Range>) {
        for range in ranges {
            manager.insert(range);
            assert_ranges_are_non_overlapping(&manager.ranges);
            assert_ranges_are_ordered(&manager.ranges);
            assert_ranges_are_disjoint(&manager.ranges);
        }
    }

    proptest! {
        #[test]
        fn proptest_range_manager_sanity(ranges in proptest::collection::vec(arb_range(1024, 128), 32)) {
            let mut manager = RangeManager::default();
            add_ranges(&mut manager, ranges);
        }
    }

    /// Used to run a proptest regression.
    #[allow(dead_code)]
    fn run_regression(ranges: impl IntoIterator<Item = std::ops::RangeInclusive<u32>>) {
        let mut manager = RangeManager::default();
        let ranges = ranges.into_iter().map(Range::from);
        add_ranges(&mut manager, ranges);
    }
}
