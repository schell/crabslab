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

#[test]
fn update_take_and_drop_sanity() {
    let mut right = Update {
        range: Range::from(0u32..=9),
        data: (0..=9u32).collect(),
    };
    let left = right.take(4);
    pretty_assertions::assert_eq!(
        Update {
            range: Range::from(0u32..=3),
            data: (0..=3u32).collect()
        },
        left
    );
    pretty_assertions::assert_eq!(
        Update {
            range: Range::from(4u32..=9),
            data: (4u32..=9).collect(),
        },
        right
    );

    let mut left = Update {
        range: Range::from(0u32..=9),
        data: (0..=9u32).collect(),
    };
    let right = left.drop(4);
    assert_eq!(
        Update {
            range: Range::from(0u32..=5),
            data: (0u32..=5).collect(),
        },
        left
    );
    assert_eq!(
        Update {
            range: Range::from(6u32..=9),
            data: (6u32..=9).collect(),
        },
        right
    );
}

fn arb_range(slab_length: u32, max_range_length: u32) -> impl Strategy<Value = Range> {
    (1..max_range_length)
        .prop_flat_map(move |range_length| ((range_length - 1)..slab_length, Just(range_length)))
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
