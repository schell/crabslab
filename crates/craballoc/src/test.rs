use std::{
    ops::{Deref, DerefMut},
    panic::catch_unwind,
};

use craballoc_test_wire_types::{
    AnyChangeId, ApplyDataChangeInvocation, ArrayChange, Data, DataChange, DataChangeTy,
};
use crabslab::{Array, Id};
use proptest::{
    prelude::{Just, Strategy},
    prop_compose, proptest,
};

use crate::{
    arena::{Arena, Value},
    range::Range,
    runtime::{CpuRuntime, IsRuntime},
    test::wgpu::TestBackendWgpu,
};

mod wgpu;

#[test]
fn mngr_updates_count_sanity() {
    let slab = Arena::new(&CpuRuntime, "sanity", None);
    assert!(
        slab.get_buffer().is_none(),
        "should not have a buffer until after 'commit'"
    );
    assert!(
        !slab.has_queued_updates(),
        "should not have any queued updates"
    );
    {
        let value = slab.new_value(666u32);
        assert_eq!(
            1,
            value.ref_count(),
            "slab should not retain a count on value"
        );
        assert!(
            slab.has_queued_updates(),
            "should have queued updates after new value"
        );
    }
    let buffer = slab.commit();
    assert_eq!(
        0,
        slab.get_live_source_ids().count(),
        "value should have dropped with no refs"
    );
    {
        let values = slab.new_array([666u32, 420u32]);
        assert_eq!(
            1,
            values.ref_count(),
            "slab should not retain a count on array"
        );
    }
    let new_buffer = slab.commit();
    assert!(
        new_buffer.creation_time() > buffer.creation_time(),
        "buffer capacity change should have invalidated the old buffer"
    );
    assert_eq!(
        0,
        slab.get_live_source_ids().count(),
        "array should have dropped with no refs"
    );
}

#[test]
fn range_sanity() {
    let a = Range {
        first_index: 1,
        last_index: 2,
    };
    let b = Range {
        first_index: 0,
        last_index: 0,
    };
    assert!(!a.intersects(&b));
    assert!(!b.intersects(&a));
}

#[test]
fn arena_roundtrip_sanity() {
    {
        // Do it with u32s
        let arena = Arena::new(&crate::wgpu_runtime(), "test", None);
        let values = arena.new_array(0u32..=9);
        let _ = arena.commit();
        let from_gpu = futures_lite::future::block_on(arena.read_slab(values.array())).unwrap();
        let from_cpu = values.read_range(.., |ts| ts.to_vec());
        assert_eq!(from_cpu, from_gpu);
    }
    {
        // Do it with `Data`
        let arena = Arena::new(&crate::wgpu_runtime(), "test", None);
        let values = arena.new_array([
            Data {
                i: 0,
                float: 0.0,
                ints: (0, 0),
            },
            Data {
                i: 1,
                float: 1.0,
                ints: (1, 1),
            },
            Data {
                i: 2,
                float: 2.0,
                ints: (2, 2),
            },
        ]);
        let _ = arena.commit();
        let from_gpu = futures_lite::future::block_on(arena.read_slab(values.array())).unwrap();
        let from_cpu = values.read_range(.., |ts| ts.to_vec());
        assert_eq!(from_cpu, from_gpu);
    }
}

#[test]
fn slab_manager_sanity() {
    let _ = env_logger::builder().is_test(true).try_init();

    let m = Arena::new(&CpuRuntime, "sanity", None);
    log::info!("allocating 4 unused u32 slots");
    let id0 = m.allocate::<u32>();
    let id1 = m.allocate::<u32>();
    let id2 = m.allocate::<u32>();
    let id3 = m.allocate::<u32>();
    log::info!("{:?}", [id0, id1, id2, id3].map(Range::from));

    log::info!("creating 4 update sources");
    let h4 = m.new_value(0u32);
    let h5 = m.new_value(0u32);
    let h6 = m.new_value(0u32);
    let h7 = m.new_value(0u32);
    assert_eq!(4, m.get_live_source_ids().count());

    log::info!("running commit");
    let starting_height = m.buffer_creation_time();
    let buffer = m.commit();
    assert!(
        starting_height < buffer.creation_time(),
        "buffer should be new on first commit: creation {}",
        buffer.creation_time()
    );
    assert_eq!(0, m.recycle_spaces());
    assert_eq!(4, m.get_live_source_ids().count());

    log::info!("dropping 4 update sources");
    drop(h4);
    drop(h5);
    drop(h6);
    drop(h7);
    let previous_buffer = buffer;
    let buffer = m.commit();
    assert!(
        buffer.creation_time() == previous_buffer.creation_time(),
        "buffer should still be valid"
    );
    assert_eq!(4, m.recycle_spaces(), "4 spaces should have been recycled");
    assert_eq!(
        1,
        m.contiguous_recycle_ranges(),
        "4 recycled spaces should have coalesced into one range"
    );
    assert_eq!(0, m.get_live_source_ids().count());

    log::info!("creating 4 update sources, round two");
    let h4 = m.new_value(0u32);
    let h5 = m.new_value(0u32);
    let h6 = m.new_value(0u32);
    let h7 = m.new_value(0u32);
    assert_eq!(
        0,
        m.recycle_spaces(),
        "after re-allocaction recycled items should have been reused"
    );
    assert_eq!(
        4,
        m.get_live_source_ids().count(),
        "should show 4 live sources"
    );

    log::info!("creating one more update source, immediately dropping it and two others");
    let h8 = m.new_value(0u32);
    drop(h8);
    drop(h4);
    drop(h6);
    let _ = m.commit();
    assert_eq!(3, m.contiguous_recycle_ranges());
    assert_eq!(2, m.get_live_source_ids().count());

    drop(h7);
    drop(h5);
    let _ = m.commit();
    assert_eq!(
        1,
        m.contiguous_recycle_ranges(),
        "Should only have one contiguous range recycled: ranges: [{}]",
        m.recycle_ranges()
            .ranges
            .iter()
            .map(|r| format!("{r:?}"))
            .collect::<Vec<_>>()
            .join(", ")
    );
}

#[test]
fn overwrite_sanity() {
    let m = Arena::new(&CpuRuntime, "sanity", None);
    let a = m.new_value(0u32);
    m.commit();
    a.modify(|u| *u = 1);
    a.modify(|u| *u = 2);
    let vs = m.get_buffer().unwrap().as_vec().clone();
    assert_eq!(0, vs[0]);

    m.commit();
    let vs = m.get_buffer().unwrap().as_vec().clone();
    assert_eq!(2, vs[0]);
}

/// This is a macro instead of a function so we get immediate panic info without
/// having to backtrace.
macro_rules! ensure {
    ($slab:ident, $initial_values:ident, $id:ident) => {{
        log::debug!("value update ranges: {:?}", $id.updated_ranges());
        log::debug!("ensuring CPU values (right) match expected (left)");
        // get it back from the cpu side
        let cpu_values = $id.get_vec();
        assert_eq!($initial_values, cpu_values, "cpu side wrong");
        log::debug!("  ...CPU values are all good :)");
        // check they still match
        log::debug!("ensuring GPU values (right) match expected (left)");
        let gpu_values = futures_lite::future::block_on($slab.read_slab($id.array())).unwrap();
        assert_eq!($initial_values, gpu_values, "gpu side wrong");
        {
            // After commit, update ranges should all be cleared
            let updated_ranges_in_values = $id.updated_ranges();
            assert!(
                updated_ranges_in_values.is_empty(),
                "Value still has updated ranges after commit"
            );
            assert!(
                !$slab.has_queued_updates(),
                "Arena still has updated ranges after commit"
            );
        }
        log::debug!("  ...GPU values are all good :)");
        log::trace!("  expected: {:?}", $initial_values);
        log::trace!("       cpu: {:?}", cpu_values);
        log::trace!("       gpu: {:?}", gpu_values);
    }};
}

#[test]
/// Ensures that a `HybridArray` can update a range.
fn array_subslice_sanity() {
    let _ = env_logger::builder().is_test(true).try_init();

    log::info!("creating the slab");
    let slab = Arena::new(&CpuRuntime, "test", None);
    let mut initial_values = vec![
        Data {
            i: 0,
            float: 1.0,
            ints: (1, 1),
        },
        Data {
            i: 1,
            float: 2.0,
            ints: (2, 2),
        },
        Data {
            i: 2,
            float: 3.0,
            ints: (3, 3),
        },
    ];

    log::info!("staging initial values");
    let values = slab.new_array(initial_values.clone());
    ensure!(slab, initial_values, values);

    log::info!("updating the initial values");
    // change the initial values
    initial_values[1].ints = (666, 666);
    initial_values[2].ints = (420, 420);
    // modify the slab values to match
    values.modify_range(1u32..3, |items| {
        items[0].ints = (666, 666);
        items[1].ints = (420, 420);
    });
    log::debug!("updated_ranges: {:?}", values.updated_ranges());
    ensure!(slab, initial_values, values);

    log::info!("updating the initial values with overlapping updates");
    // Now ensure that two updates within one commit apply correctly.
    // 1. update an outer range
    // 2. update an inner range
    // 3. ensure they are as expected
    initial_values[0].float = 10.0;
    initial_values[1].float = 20.0;
    initial_values[2].float = 30.0;
    values.modify_range(0u32..3, |items| {
        items[0].float = 10.0;
        items[1].float = 20.0;
        items[2].float = 30.0;
    });

    initial_values[1].float = 666.0;
    values.modify_range(1u32..2, |items| {
        items[0].float = 666.0;
    });
    ensure!(slab, initial_values, values);

    // Ensure the other setting functions work too
    initial_values[2].ints = (32, 32);
    values.modify_item(2, |data| data.ints = (32, 32));
    ensure!(slab, initial_values, values);
}

prop_compose! {
    fn arb_range(max_length: u32)
    (last_index in 1u32..max_length)(first_index in 0u32..last_index, last_index in Just(last_index)) -> Range {
        Range{ first_index, last_index }
    }
}

#[derive(Clone, Debug)]
struct OverlappingUpdate {
    range: Range,
    data: Vec<u32>,
}

const VALUES_LENGTH: usize = 16;

prop_compose! {
    fn arb_update(n: u32)(range in arb_range(VALUES_LENGTH as u32)) -> OverlappingUpdate {
        OverlappingUpdate { range, data: vec![n; range.len() as usize] }
    }
}

fn arb_updates(max_updates: usize) -> impl Strategy<Value = Vec<OverlappingUpdate>> {
    // Do at least 3 updates
    (3..=max_updates).prop_flat_map(|num_updates| {
        (1..=num_updates)
            .map(|i| arb_update(i as u32))
            .collect::<Vec<_>>()
    })
}

fn run_overlapping_test(updates: Vec<OverlappingUpdate>) {
    let ranges = updates
        .clone()
        .into_iter()
        .map(|up| up.range)
        .collect::<Vec<_>>();
    log::info!("ranges: {ranges:?}");
    let slab = Arena::new(&crate::wgpu_runtime(), "test", None);
    let mut local_values = vec![0u32; VALUES_LENGTH];
    let arena_values = slab.new_array(local_values.clone());
    ensure!(slab, local_values, arena_values);

    for (i, OverlappingUpdate { range, data }) in updates.into_iter().enumerate() {
        log::debug!("running update {i} {range:?} {data:?}");
        // Change the last elements to
        let new_values = data;
        local_values[range].copy_from_slice(&new_values);
        arena_values.modify_range(range, |data| {
            data.copy_from_slice(&new_values);
        });
    }
    ensure!(slab, local_values, arena_values);
}

proptest! {
    #[test]
    fn proptest_overlapping_updates(updates in arb_updates(16)) {
        let _ = env_logger::builder().is_test(true).try_init();
        run_overlapping_test(updates);
    }
}

fn arb_data() -> impl Strategy<Value = Data> {
    (
        proptest::num::u32::ANY,
        proptest::num::f32::NORMAL,
        proptest::num::u32::ANY,
        proptest::num::u32::ANY,
    )
        .prop_map(|(i, float, ints_i, ints_j)| Data {
            i,
            float,
            ints: (ints_i, ints_j),
        })
}

fn arb_array_data(max_length: usize) -> impl Strategy<Value = Vec<Data>> {
    proptest::collection::vec(arb_data(), 1..max_length)
}

fn arb_data_change() -> impl Strategy<Value = DataChange> {
    let i = proptest::num::u32::ANY.prop_map(DataChange::i).boxed();
    let float = proptest::num::f32::NORMAL
        .prop_map(DataChange::float)
        .boxed();
    let int = (proptest::num::u32::ANY, proptest::num::u32::ANY)
        .prop_map(|(i, j)| DataChange::ints(i, j))
        .boxed();
    i.prop_union(float).boxed().prop_union(int)
}

fn arb_array_change(array_len: usize) -> impl Strategy<Value = ArrayChange> {
    (0..array_len as u32, arb_data_change()).prop_map(|(i, change)| ArrayChange { i, change })
}

#[derive(Clone, Debug)]
enum ValueData {
    Single(Data, Vec<DataChange>),
    Array(Vec<Data>, Vec<ArrayChange>),
}

fn arb_value_data(max_array_length: usize, max_changes: usize) -> impl Strategy<Value = ValueData> {
    let singles = (
        arb_data(),
        proptest::collection::vec(arb_data_change(), 0..max_changes),
    )
        .prop_map(|(data, changes)| ValueData::Single(data, changes))
        .boxed();
    let arrays = arb_array_data(max_array_length)
        .prop_flat_map(move |array_data| {
            (
                proptest::collection::vec(arb_array_change(array_data.len()), 0..max_changes),
                Just(array_data),
            )
                .prop_map(|(changes, array)| ValueData::Array(array, changes))
        })
        .boxed();
    singles.prop_union(arrays)
}

#[derive(Clone, Debug)]
enum Values {
    Single {
        arena: Value<Data>,
        raw: Data,
        changes: Vec<DataChange>,
    },
    Array {
        arena: Value<[Data]>,
        raw: Vec<Data>,
        changes: Vec<ArrayChange>,
    },
}

impl Values {
    fn raw_data_and_next_change_as_single(&self) -> (Option<&Data>, Option<&DataChange>) {
        if let Self::Single {
            arena: _,
            raw,
            changes,
        } = self
        {
            (Some(raw), changes.last())
        } else {
            (None, None)
        }
    }

    fn raw_data_and_next_change_as_array(&self) -> (Option<&[Data]>, Option<&ArrayChange>) {
        if let Self::Array {
            arena: _,
            raw,
            changes,
        } = self
        {
            (Some(raw), changes.last())
        } else {
            (None, None)
        }
    }
}

trait BackendUpdate {
    /// Apply one step's worth changes to the backend, without the frontend's knowledge,
    /// in order to test the synchronization of the backend to the frontend.
    ///
    /// * In the case of CpuRuntime, this runs the update shader function manually for each change.
    /// * In the case of WgpuRuntime, this invokes the compute shader that performs the updates on the GPU.
    fn apply_backend_changes(&mut self);
}

impl BackendUpdate for GpuUpdateTest<CpuRuntime, ()> {
    fn apply_backend_changes(&mut self) {
        let slab_buffer = self.arena.get_buffer().unwrap();
        let mut data_slab = slab_buffer.as_mut_vec();
        let changes_buffer = self.changes_arena.get_buffer().unwrap();
        let changes_slab = changes_buffer.as_vec();

        let workgroups = self.invocation.get().workgroup_dimensions();
        log::debug!("  invocation workgroup dimensions: {workgroups:?}");
        let size = ApplyDataChangeInvocation::WORKGROUP_SIZE;
        for i in 0..workgroups.x * size.x {
            for j in 0..workgroups.y * size.y {
                for k in 0..workgroups.z * size.z {
                    ApplyDataChangeInvocation::run(
                        data_slab.deref_mut(),
                        changes_slab.deref(),
                        glam::UVec3::new(i, j, k),
                    );
                }
            }
        }
    }
}

#[derive(Debug)]
enum BackendChange {
    Single {
        id: Id<Data>,
        change: DataChange,
    },
    Array {
        array: Array<Data>,
        change: ArrayChange,
    },
}

struct GpuUpdateTest<R: IsRuntime, T> {
    arena: Arena<R>,
    current_values: Vec<Values>,

    previous_values: Vec<Values>,

    changes_arena: Arena<R>,
    changes_values: Vec<Value<ArrayChange>>,
    changes_all_change_ids: Value<[AnyChangeId]>,

    invocation: Value<ApplyDataChangeInvocation>,
    invocation_count: Value<u32>,
    invocations_skipped: Value<u32>,

    backend_updater: T,
}

impl<R: IsRuntime, T> GpuUpdateTest<R, T> {
    /// Apply one step's worth of changes to the raw values and return the changes to be
    /// made by the backend.
    fn apply_raw_changes(&mut self) -> Vec<BackendChange> {
        let mut backend_changes = vec![];
        for value in self.current_values.iter_mut() {
            match value {
                Values::Single {
                    arena,
                    raw,
                    changes,
                } => {
                    if let Some(change) = changes.pop() {
                        log::trace!("    applying change {change:?} to single value");
                        change.apply(raw);
                        backend_changes.push(BackendChange::Single {
                            id: arena.id(),
                            change,
                        });
                    }
                }
                Values::Array {
                    arena,
                    raw,
                    changes,
                } => {
                    if let Some(change) = changes.pop() {
                        log::trace!(
                            "    applying change {change:?} to array value {:?}",
                            arena.array()
                        );
                        change.apply(raw);
                        backend_changes.push(BackendChange::Array {
                            array: arena.array(),
                            change,
                        });
                    }
                }
            }
        }
        backend_changes
    }

    /// Asserts that the cached **CPU values** match the expected **raw values**.
    fn verify(&self) {
        let total = self.current_values.len();
        for (i, value) in self.current_values.iter().enumerate() {
            let previous = self.previous_values.get(i).unwrap();

            let i = i + 1;
            match value {
                Values::Single {
                    arena,
                    raw,
                    changes: _,
                } => {
                    let (previous_raw_data, change_made) =
                        previous.raw_data_and_next_change_as_single();
                    let previous_raw_data = previous_raw_data.unwrap();
                    let unchanged_since_previous = previous_raw_data == &arena.get();
                    let change_made = change_made
                        .map(|c| c.to_string())
                        .unwrap_or("no change".to_string());

                    pretty_assertions::assert_eq!(
                        raw,
                        &arena.get(),
                        "{reason} - unexpected single value for entry {i} out of {total} ({:?}) after applying {change_made} to previous value:\n {previous_raw_data:#?}",
                        arena.slab_range(),
                        reason = if unchanged_since_previous {
                            "Value is unchanged and probably should be"
                        } else {
                            "Value changed incorrectly"
                        }
                    )
                }
                Values::Array {
                    arena,
                    raw,
                    changes: _,
                } => {
                    let (previous_raw_data, change_made) =
                        previous.raw_data_and_next_change_as_array();
                    let previous_raw_data = previous_raw_data.unwrap();
                    let unchanged_since_previous = previous_raw_data == arena.get_vec();
                    let change_made = change_made
                        .map(|c| c.to_string())
                        .unwrap_or("no change".to_string());

                    pretty_assertions::assert_eq!(
                        raw,
                        &arena.get_vec(),
                        "{reason} - unexpected array value for entry {i} out of {total} ({:?}) after applying {change_made} to previous value:\n {previous_raw_data:#?}",
                        arena.slab_range(),
reason = if unchanged_since_previous {
                            "Value is unchanged and probably should be"
                        } else {
                            "Value changed incorrectly"
                        }                    );
                }
            }
        }
    }

    fn new(arena: Arena<R>, backend_updater: T, value_data: &[ValueData]) -> Self {
        let all_values = value_data
            .iter()
            .map(|vd| match vd {
                ValueData::Single(raw, changes) => Values::Single {
                    arena: arena.new_value(*raw),
                    raw: *raw,
                    changes: changes.to_vec(),
                },
                ValueData::Array(raw, changes) => Values::Array {
                    arena: arena.new_array(raw.clone()),
                    raw: raw.clone(),
                    changes: changes.to_vec(),
                },
            })
            .collect::<Vec<_>>();

        let changes_arena = Arena::new(arena.runtime(), "test-changes", None);
        let invocation_count = arena.new_value(0u32);
        let invocations_skipped = arena.new_value(0u32);
        let invocation = changes_arena.new_value(ApplyDataChangeInvocation {
            invocations_id: invocation_count.id(),
            invocations_skipped_id: invocations_skipped.id(),
            ..Default::default()
        });
        let changes_all_change_ids = changes_arena.new_array(vec![]);
        let test = GpuUpdateTest {
            arena,
            previous_values: all_values.clone(),
            current_values: all_values,
            backend_updater,

            changes_arena,
            changes_values: vec![],
            changes_all_change_ids,

            invocation,
            invocation_count,
            invocations_skipped,
        };
        test.verify();
        test
    }

    fn update_changes_on_backend(&mut self, changes: Vec<BackendChange>) {
        log::info!("  updating to changeset {changes:#?}");
        self.invocation_count.set(0);
        self.invocations_skipped.set(0);
        let _ = self.arena.commit();

        let mut change_values = vec![];
        let mut any_changes = vec![];
        for change in changes.into_iter() {
            let (array, change) = match change {
                BackendChange::Single { id, change } => {
                    let array_change = ArrayChange::from(change);
                    let array_id = Array::new(id, 1);
                    (array_id, array_change)
                }
                BackendChange::Array { array, change } => (array, change),
            };
            // Host the value of the change itself
            let change_value = self.changes_arena.new_value(change);
            // Store a pointer to use later in a contiguous array
            any_changes.push(AnyChangeId {
                change_id: change_value.id(),
                data_array: array,
            });
            change_values.push(change_value);
        }

        // Host all the change ids in one contiguous array
        let any_change_array = self.changes_arena.new_array(any_changes);

        // Update the invocation with the details it needs to invoke the "changes" compute shader.
        self.invocation.modify(|inv| {
            inv.changes_ids = any_change_array.array();
        });

        self.changes_values = change_values;
        self.changes_all_change_ids = any_change_array;

        // Commit the arena
        log::info!("  commiting changes");
        let _ = self.changes_arena.commit();
        log::info!("  done commiting changes");
    }

    /// `verify_each_step = true` will commit, synchronize and compare after each loop,
    /// otherwise all changes are applied and then one commit+synchronize+compare is done at the end
    fn run(mut self, verify_each_step: bool)
    where
        Self: BackendUpdate,
    {
        // Send the initial CPU values to the backend.
        log::info!("initial commit");
        let _ = self.arena.commit();
        log::info!("done initial commit");
        self.verify();

        let mut steps = 0;
        loop {
            if verify_each_step {
                log::debug!("");
                log::debug!("!-- step {steps}");
            }
            // Apply the changes to our raw values and return the backend changes to apply
            log::info!("  applying changes to raw data");
            let backend_changes = self.apply_raw_changes();
            log::info!("  done applying changes to raw data");
            if backend_changes.is_empty() {
                if verify_each_step {
                    log::trace!("  ({steps}) applied all changes");
                }
                break;
            }

            // Ready the changes on the changes_slab.
            self.update_changes_on_backend(backend_changes);
            let invocation = self.invocation.get();
            // Apply the changes from the changes_slab to the data_slab using a shader
            log::info!("  applying backend changes");
            self.apply_backend_changes();
            let invocations_ran =
                futures_lite::future::block_on(self.arena.read_slab(self.invocation_count.array()))
                    .unwrap()
                    .pop()
                    .unwrap();
            assert_eq!(
                invocation.total_invocations_required(),
                invocations_ran,
                "incorrect number of compute shader invocations"
            );
            log::info!("  done applying backend changes, ran {invocations_ran} invocations");
            log::info!("  synchronizing to CPU caches");
            futures_lite::future::block_on(self.arena.synchronize()).unwrap();
            log::info!("  done synchronizing to CPU caches");

            if verify_each_step {
                self.verify();
                log::debug!("!-- {steps}");
                log::debug!("");
            }
            self.previous_values = self.current_values.clone();
            steps += 1;
        }
        self.verify();
    }
}

#[test]
/// Run just one update and ensure we get results we expect.
fn gpu_update_test_sanity_on_cpu() {
    let _ = env_logger::builder().is_test(true).try_init();
    let arena = Arena::new(&CpuRuntime, "test", None);
    let all_values = vec![ValueData::Single(
        Data {
            i: 0,
            float: 0.0,
            ints: (0, 0),
        },
        vec![
            DataChange::i(1),
            DataChange::float(1.0),
            DataChange::ints(1, 1),
        ],
    )];
    let test = GpuUpdateTest::new(arena, (), &all_values);
    test.run(true);
}

#[test]
/// Run just one update and ensure we get results we expect.
fn gpu_update_test_sanity_on_gpu() {
    let _ = env_logger::builder().is_test(true).try_init();
    let runtime = crate::wgpu_runtime();
    let arena = Arena::new(&runtime, "test", None);
    let all_values = vec![ValueData::Single(
        Data {
            i: 0,
            float: 0.0,
            ints: (0, 0),
        },
        vec![
            DataChange::i(1),
            DataChange::float(1.0),
            DataChange::ints(1, 1),
        ],
    )];
    let test = GpuUpdateTest::new(arena, wgpu::TestBackendWgpu::new(runtime), &all_values);
    test.run(true);
}

#[test]
/// Run just one update and ensure we get results we expect.
fn gpu_array_update_test_sanity_on_cpu() {
    let _ = env_logger::builder().is_test(true).try_init();
    let runtime = CpuRuntime;
    let arena = Arena::new(&runtime, "test", None);
    let all_values = vec![ValueData::Array(
        vec![Data {
            i: 1683186,
            float: 2.1727349e24,
            ints: (348221601, 1304208859),
        }],
        vec![ArrayChange {
            i: 0,
            change: DataChange {
                ty: DataChangeTy::Ints,
                data: [3211909787, 1326905872, 0],
            },
        }],
    )];

    let test = GpuUpdateTest::new(arena, (), &all_values);
    test.run(true);
}

#[test]
/// Run just one update and ensure we get results we expect.
fn gpu_array_update_test_sanity_on_gpu() {
    let _ = env_logger::builder().is_test(true).try_init();
    let runtime = crate::wgpu_runtime();
    let arena = Arena::new(&runtime, "test", None);
    let all_values = vec![ValueData::Array(
        vec![Data {
            i: 1683186,
            float: 2.1727349e24,
            ints: (348221601, 1304208859),
        }],
        vec![ArrayChange {
            i: 0,
            change: DataChange {
                ty: DataChangeTy::Ints,
                data: [3211909787, 1326905872, 0],
            },
        }],
    )];

    let test = GpuUpdateTest::new(arena, TestBackendWgpu::new(runtime), &all_values);
    test.run(true);
}

fn arb_vec_of_value_data() -> impl Strategy<Value = Vec<ValueData>> {
    proptest::collection::vec(arb_value_data(32, 32), 1..32)
}

proptest! {
    #[test]
    fn proptest_gpu_updates_checked_on_cpu(value_data in arb_vec_of_value_data()) {
        let _ = env_logger::builder().is_test(true).try_init();
        {
            log::info!("running test with stepwise verification");
            let arena = Arena::new(&CpuRuntime, "test", None);
            let test = GpuUpdateTest::new(arena, (), &value_data);
            test.run(true);
        }
        {
            log::info!("running test with verification only at the end");
            let arena = Arena::new(&CpuRuntime, "test", None);
            let test = GpuUpdateTest::new(arena, (), &value_data);
            test.run(false);
        }
    }
}

proptest! {
    #[test]
    fn proptest_gpu_updates_checked_on_gpu(value_data in proptest::collection::vec(arb_value_data(32, 32), 1..32)) {
        let _ = env_logger::builder().is_test(true).try_init();
        {
            log::info!("running test with stepwise verification");
            let runtime = crate::wgpu_runtime();
            let arena = Arena::new(&runtime, "test", None);
            let test = GpuUpdateTest::new(arena, wgpu::TestBackendWgpu::new(runtime), &value_data);
            test.run(true);
        }
        {
            log::info!("running test with verification only at the end");
            let arena = Arena::new(&CpuRuntime, "test", None);
            let test = GpuUpdateTest::new(arena, (), &value_data);
            test.run(false);
        }
    }
}

fn one_datum() -> ValueData {
    ValueData::Array(
        vec![Data {
            i: 0,
            float: 0.0,
            ints: (0, 0),
        }],
        vec![ArrayChange {
            i: 0,
            change: DataChange {
                ty: DataChangeTy::I,
                data: [666, 0, 0],
            },
        }],
    )
}

fn data() -> Vec<ValueData> {
    use craballoc_test_wire_types::DataChangeTy::*;
    use ValueData::*;

    vec![
        Single(
            Data {
                i: 0,
                float: 8.996603e-27,
                ints: (0, 0),
            },
            vec![DataChange {
                ty: I,
                data: [0, 0, 0],
            }],
        ),
        Single(
            Data {
                i: 0,
                float: 1.9006078e32,
                ints: (0, 0),
            },
            vec![DataChange {
                ty: I,
                data: [0, 0, 0],
            }],
        ),
        Array(
            vec![
                Data {
                    i: 1376371250,
                    float: 1.8671383e29,
                    ints: (1273861579, 1171626079),
                },
                Data {
                    i: 2734217204,
                    float: 8818.755,
                    ints: (1551911770, 2160758874),
                },
                Data {
                    i: 4276625026,
                    float: 3.563882e-37,
                    ints: (249071469, 902284071),
                },
                Data {
                    i: 2032668320,
                    float: 3.8784836e-26,
                    ints: (1593977556, 4115511719),
                },
                Data {
                    i: 3689562531,
                    float: 2157724600000000.0,
                    ints: (301245638, 1571399662),
                },
                Data {
                    i: 2262747971,
                    float: 1.2125642e28,
                    ints: (2542081851, 995054215),
                },
                Data {
                    i: 3072759673,
                    float: 4.6186345e26,
                    ints: (4193599093, 3999717220),
                },
                Data {
                    i: 716422532,
                    float: 5.3523973e36,
                    ints: (2576529535, 249762356),
                },
            ],
            vec![ArrayChange {
                i: 0,
                change: DataChange {
                    ty: Ints,
                    data: [26158660, 1223788524, 0],
                },
            }],
        ),
        Array(
            vec![
                Data {
                    i: 1135351142,
                    float: 2.8615392e22,
                    ints: (404844871, 569410752),
                },
                Data {
                    i: 3471386689,
                    float: 7.092646e36,
                    ints: (1751317528, 11197837),
                },
            ],
            vec![
                ArrayChange {
                    i: 1,
                    change: DataChange {
                        ty: I,
                        data: [1421611497, 0, 0],
                    },
                },
                ArrayChange {
                    i: 1,
                    change: DataChange {
                        ty: Float,
                        data: [662243954, 0, 0],
                    },
                },
                ArrayChange {
                    i: 1,
                    change: DataChange {
                        ty: Float,
                        data: [1318824691, 0, 0],
                    },
                },
                ArrayChange {
                    i: 1,
                    change: DataChange {
                        ty: Float,
                        data: [588216627, 0, 0],
                    },
                },
            ],
        ),
        Array(
            vec![
                Data {
                    i: 52465682,
                    float: 2.0613145e30,
                    ints: (1379297340, 979525639),
                },
                Data {
                    i: 147996011,
                    float: 1.5938896e24,
                    ints: (506995624, 1727080909),
                },
                Data {
                    i: 398856234,
                    float: 126069.31,
                    ints: (1873826221, 2567184081),
                },
                Data {
                    i: 3832528027,
                    float: 1899446300000000.0,
                    ints: (2691094174, 291451694),
                },
            ],
            vec![
                ArrayChange {
                    i: 0,
                    change: DataChange {
                        ty: I,
                        data: [3635057218, 0, 0],
                    },
                },
                ArrayChange {
                    i: 3,
                    change: DataChange {
                        ty: Ints,
                        data: [2458508808, 3089057576, 0],
                    },
                },
                ArrayChange {
                    i: 1,
                    change: DataChange {
                        ty: Ints,
                        data: [272308574, 1646966257, 0],
                    },
                },
                ArrayChange {
                    i: 2,
                    change: DataChange {
                        ty: Ints,
                        data: [4194856277, 1451142527, 0],
                    },
                },
            ],
        ),
        Single(
            Data {
                i: 1266164226,
                float: 55969722000.0,
                ints: (1842986441, 445186815),
            },
            vec![
                DataChange {
                    ty: Ints,
                    data: [298737096, 936869214, 0],
                },
                DataChange {
                    ty: Ints,
                    data: [1873898507, 3034166444, 0],
                },
                DataChange {
                    ty: Float,
                    data: [682734885, 0, 0],
                },
                DataChange {
                    ty: Ints,
                    data: [1201568539, 1772735046, 0],
                },
                DataChange {
                    ty: Float,
                    data: [1709637560, 0, 0],
                },
                DataChange {
                    ty: I,
                    data: [2217038829, 0, 0],
                },
                DataChange {
                    ty: I,
                    data: [745166329, 0, 0],
                },
                DataChange {
                    ty: Float,
                    data: [1229678408, 0, 0],
                },
                DataChange {
                    ty: Float,
                    data: [644573992, 0, 0],
                },
                DataChange {
                    ty: Ints,
                    data: [2840857875, 3753624087, 0],
                },
                DataChange {
                    ty: Float,
                    data: [1746052010, 0, 0],
                },
                DataChange {
                    ty: Ints,
                    data: [3525096513, 1991389602, 0],
                },
                DataChange {
                    ty: Ints,
                    data: [36631112, 3743937828, 0],
                },
                DataChange {
                    ty: I,
                    data: [966014160, 0, 0],
                },
                DataChange {
                    ty: Ints,
                    data: [1144680572, 1277520602, 0],
                },
                DataChange {
                    ty: I,
                    data: [2363835183, 0, 0],
                },
                DataChange {
                    ty: Ints,
                    data: [3026623026, 1206640058, 0],
                },
                DataChange {
                    ty: Ints,
                    data: [3612678779, 2343634259, 0],
                },
                DataChange {
                    ty: Float,
                    data: [1629803937, 0, 0],
                },
                DataChange {
                    ty: Ints,
                    data: [2868667341, 1983961986, 0],
                },
                DataChange {
                    ty: Float,
                    data: [877923661, 0, 0],
                },
                DataChange {
                    ty: Ints,
                    data: [341852284, 608155222, 0],
                },
            ],
        ),
        Single(
            Data {
                i: 3266377771,
                float: 3.471552e-22,
                ints: (2288118850, 2635948781),
            },
            vec![
                DataChange {
                    ty: Ints,
                    data: [2574827423, 1230417735, 0],
                },
                DataChange {
                    ty: Float,
                    data: [1209827024, 0, 0],
                },
                DataChange {
                    ty: Ints,
                    data: [2374580632, 3374395687, 0],
                },
                DataChange {
                    ty: Ints,
                    data: [65433928, 4027210477, 0],
                },
                DataChange {
                    ty: Ints,
                    data: [1592153628, 2612225693, 0],
                },
                DataChange {
                    ty: Ints,
                    data: [2476731113, 4150797290, 0],
                },
                DataChange {
                    ty: Float,
                    data: [1792048994, 0, 0],
                },
                DataChange {
                    ty: Ints,
                    data: [3807076900, 4286528877, 0],
                },
                DataChange {
                    ty: Float,
                    data: [1251269371, 0, 0],
                },
                DataChange {
                    ty: Ints,
                    data: [4256036378, 2095392629, 0],
                },
                DataChange {
                    ty: Ints,
                    data: [1501531441, 3248484415, 0],
                },
                DataChange {
                    ty: Ints,
                    data: [3725085759, 1365190384, 0],
                },
                DataChange {
                    ty: Ints,
                    data: [244074221, 4094313440, 0],
                },
                DataChange {
                    ty: Ints,
                    data: [1716649088, 4111851080, 0],
                },
                DataChange {
                    ty: I,
                    data: [61800574, 0, 0],
                },
                DataChange {
                    ty: I,
                    data: [2331259046, 0, 0],
                },
                DataChange {
                    ty: I,
                    data: [3721288660, 0, 0],
                },
                DataChange {
                    ty: I,
                    data: [199872315, 0, 0],
                },
                DataChange {
                    ty: I,
                    data: [4134944636, 0, 0],
                },
                DataChange {
                    ty: Ints,
                    data: [2648722022, 1955884071, 0],
                },
                DataChange {
                    ty: I,
                    data: [207476549, 0, 0],
                },
                DataChange {
                    ty: Ints,
                    data: [265458101, 1469330093, 0],
                },
                DataChange {
                    ty: Ints,
                    data: [3057003214, 3862707780, 0],
                },
                DataChange {
                    ty: Ints,
                    data: [4200605488, 2690058445, 0],
                },
                DataChange {
                    ty: I,
                    data: [3207989472, 0, 0],
                },
                DataChange {
                    ty: Float,
                    data: [700666377, 0, 0],
                },
            ],
        ),
        Array(
            vec![
                Data {
                    i: 1988244455,
                    float: 2.5331946e-7,
                    ints: (3566118249, 1703133033),
                },
                Data {
                    i: 3196338618,
                    float: 5.0157976e-38,
                    ints: (2388662438, 1284598280),
                },
                Data {
                    i: 1932153588,
                    float: 0.004156772,
                    ints: (3841941202, 153582876),
                },
                Data {
                    i: 117707338,
                    float: 599477650000.0,
                    ints: (3579520472, 737319063),
                },
                Data {
                    i: 1126025170,
                    float: 4.7519186e-32,
                    ints: (2961615570, 3773809254),
                },
                Data {
                    i: 3532864862,
                    float: 1.849931e-17,
                    ints: (2499677419, 2580622199),
                },
                Data {
                    i: 2651698725,
                    float: 1.1098732,
                    ints: (525541462, 1220892991),
                },
                Data {
                    i: 1619275013,
                    float: 2527.8755,
                    ints: (570335941, 4242809503),
                },
                Data {
                    i: 2626102565,
                    float: 1.9340973e31,
                    ints: (1685978918, 1342823598),
                },
                Data {
                    i: 2154946556,
                    float: 2.2384903e-30,
                    ints: (2103108064, 1505020628),
                },
                Data {
                    i: 352617577,
                    float: 3.412318e-29,
                    ints: (2727374644, 3499961406),
                },
                Data {
                    i: 1140357599,
                    float: 1.8892947e28,
                    ints: (652640480, 3517598269),
                },
                Data {
                    i: 401125772,
                    float: 8.2772843e36,
                    ints: (1302897911, 2197665969),
                },
                Data {
                    i: 149175688,
                    float: 5.6962755e35,
                    ints: (4156596474, 828643998),
                },
                Data {
                    i: 1949007361,
                    float: 1.15383e-28,
                    ints: (2892018580, 1289212742),
                },
                Data {
                    i: 37978610,
                    float: 3.4835858e-34,
                    ints: (3803022042, 1972958976),
                },
                Data {
                    i: 3997480783,
                    float: 1.4896469e29,
                    ints: (2272545032, 343136871),
                },
                Data {
                    i: 815140099,
                    float: 1.2587422e19,
                    ints: (82280121, 1447925801),
                },
                Data {
                    i: 1145246056,
                    float: 1.2847695e26,
                    ints: (1824896305, 244754129),
                },
                Data {
                    i: 4283850502,
                    float: 3.715054,
                    ints: (1943913226, 3842133012),
                },
                Data {
                    i: 329478981,
                    float: 9.4852837e-14,
                    ints: (3224156438, 521275852),
                },
                Data {
                    i: 2621774062,
                    float: 1.6554092e-26,
                    ints: (3861094524, 1228315180),
                },
                Data {
                    i: 1609973672,
                    float: 459684600.0,
                    ints: (1482641358, 3892087524),
                },
                Data {
                    i: 1590813335,
                    float: 0.00025400313,
                    ints: (633359787, 1579193683),
                },
                Data {
                    i: 19054292,
                    float: 9.452252e-33,
                    ints: (1189344160, 2668710509),
                },
                Data {
                    i: 2424807105,
                    float: 4.5035424e-16,
                    ints: (4157876265, 540534394),
                },
                Data {
                    i: 1327444533,
                    float: 1.2939051e18,
                    ints: (1173626343, 2208141979),
                },
                Data {
                    i: 2655265549,
                    float: 1.9245987e-30,
                    ints: (1189106189, 3240796362),
                },
            ],
            vec![
                ArrayChange {
                    i: 5,
                    change: DataChange {
                        ty: Float,
                        data: [2100881734, 0, 0],
                    },
                },
                ArrayChange {
                    i: 1,
                    change: DataChange {
                        ty: I,
                        data: [137585795, 0, 0],
                    },
                },
                ArrayChange {
                    i: 20,
                    change: DataChange {
                        ty: I,
                        data: [4284241363, 0, 0],
                    },
                },
                ArrayChange {
                    i: 17,
                    change: DataChange {
                        ty: I,
                        data: [1025664878, 0, 0],
                    },
                },
                ArrayChange {
                    i: 0,
                    change: DataChange {
                        ty: Ints,
                        data: [672022944, 2600021116, 0],
                    },
                },
                ArrayChange {
                    i: 27,
                    change: DataChange {
                        ty: Ints,
                        data: [1704149924, 896220844, 0],
                    },
                },
                ArrayChange {
                    i: 3,
                    change: DataChange {
                        ty: Ints,
                        data: [923559265, 2800577694, 0],
                    },
                },
                ArrayChange {
                    i: 25,
                    change: DataChange {
                        ty: Ints,
                        data: [2950208975, 2572554458, 0],
                    },
                },
                ArrayChange {
                    i: 17,
                    change: DataChange {
                        ty: I,
                        data: [1238221203, 0, 0],
                    },
                },
                ArrayChange {
                    i: 17,
                    change: DataChange {
                        ty: I,
                        data: [2201437745, 0, 0],
                    },
                },
                ArrayChange {
                    i: 18,
                    change: DataChange {
                        ty: Ints,
                        data: [589874393, 1408874952, 0],
                    },
                },
                ArrayChange {
                    i: 7,
                    change: DataChange {
                        ty: Float,
                        data: [52809872, 0, 0],
                    },
                },
                ArrayChange {
                    i: 24,
                    change: DataChange {
                        ty: Float,
                        data: [874819748, 0, 0],
                    },
                },
                ArrayChange {
                    i: 11,
                    change: DataChange {
                        ty: Ints,
                        data: [3102235763, 4225969872, 0],
                    },
                },
                ArrayChange {
                    i: 5,
                    change: DataChange {
                        ty: Ints,
                        data: [703232318, 922809889, 0],
                    },
                },
                ArrayChange {
                    i: 18,
                    change: DataChange {
                        ty: Ints,
                        data: [146173140, 3400526883, 0],
                    },
                },
                ArrayChange {
                    i: 17,
                    change: DataChange {
                        ty: I,
                        data: [960106239, 0, 0],
                    },
                },
                ArrayChange {
                    i: 17,
                    change: DataChange {
                        ty: Ints,
                        data: [423902476, 1771431976, 0],
                    },
                },
                ArrayChange {
                    i: 20,
                    change: DataChange {
                        ty: Ints,
                        data: [3187993473, 2540095887, 0],
                    },
                },
                ArrayChange {
                    i: 26,
                    change: DataChange {
                        ty: Float,
                        data: [380615065, 0, 0],
                    },
                },
                ArrayChange {
                    i: 3,
                    change: DataChange {
                        ty: I,
                        data: [905215458, 0, 0],
                    },
                },
                ArrayChange {
                    i: 14,
                    change: DataChange {
                        ty: Ints,
                        data: [3705114122, 2414044335, 0],
                    },
                },
                ArrayChange {
                    i: 20,
                    change: DataChange {
                        ty: Float,
                        data: [1957538525, 0, 0],
                    },
                },
                ArrayChange {
                    i: 24,
                    change: DataChange {
                        ty: I,
                        data: [3780158351, 0, 0],
                    },
                },
                ArrayChange {
                    i: 20,
                    change: DataChange {
                        ty: Float,
                        data: [1241379163, 0, 0],
                    },
                },
                ArrayChange {
                    i: 19,
                    change: DataChange {
                        ty: Ints,
                        data: [1456284399, 1524459168, 0],
                    },
                },
                ArrayChange {
                    i: 22,
                    change: DataChange {
                        ty: Float,
                        data: [26944829, 0, 0],
                    },
                },
                ArrayChange {
                    i: 9,
                    change: DataChange {
                        ty: Ints,
                        data: [2476289519, 4199798049, 0],
                    },
                },
            ],
        ),
        Single(
            Data {
                i: 1658404051,
                float: 7.693471e28,
                ints: (3572724392, 3344084455),
            },
            vec![
                DataChange {
                    ty: Float,
                    data: [492381839, 0, 0],
                },
                DataChange {
                    ty: Ints,
                    data: [745180664, 2777444511, 0],
                },
                DataChange {
                    ty: Float,
                    data: [108131741, 0, 0],
                },
                DataChange {
                    ty: Ints,
                    data: [3894786092, 1718487860, 0],
                },
                DataChange {
                    ty: Ints,
                    data: [1505344345, 3744148885, 0],
                },
                DataChange {
                    ty: I,
                    data: [4104032146, 0, 0],
                },
                DataChange {
                    ty: Ints,
                    data: [493772061, 3089318762, 0],
                },
                DataChange {
                    ty: Ints,
                    data: [2451117049, 1559803835, 0],
                },
                DataChange {
                    ty: I,
                    data: [2625829620, 0, 0],
                },
                DataChange {
                    ty: Ints,
                    data: [2823787384, 1545679433, 0],
                },
                DataChange {
                    ty: I,
                    data: [1801463123, 0, 0],
                },
                DataChange {
                    ty: Ints,
                    data: [1860714901, 3812706503, 0],
                },
                DataChange {
                    ty: I,
                    data: [1332410443, 0, 0],
                },
                DataChange {
                    ty: I,
                    data: [619872008, 0, 0],
                },
                DataChange {
                    ty: I,
                    data: [2646992721, 0, 0],
                },
                DataChange {
                    ty: Float,
                    data: [97741273, 0, 0],
                },
                DataChange {
                    ty: Ints,
                    data: [513182073, 4257441003, 0],
                },
                DataChange {
                    ty: Float,
                    data: [1114987236, 0, 0],
                },
                DataChange {
                    ty: I,
                    data: [4120473200, 0, 0],
                },
                DataChange {
                    ty: Float,
                    data: [512545172, 0, 0],
                },
                DataChange {
                    ty: I,
                    data: [600391249, 0, 0],
                },
                DataChange {
                    ty: Ints,
                    data: [409591071, 1895973060, 0],
                },
                DataChange {
                    ty: Ints,
                    data: [3732010277, 935649929, 0],
                },
                DataChange {
                    ty: Float,
                    data: [1956416655, 0, 0],
                },
                DataChange {
                    ty: I,
                    data: [2601180430, 0, 0],
                },
                DataChange {
                    ty: Ints,
                    data: [1059639347, 3372275883, 0],
                },
            ],
        ),
        Array(
            vec![
                Data {
                    i: 3304226443,
                    float: 1.0033503e-6,
                    ints: (1493424198, 236605024),
                },
                Data {
                    i: 439350665,
                    float: 7.146764e-11,
                    ints: (2212277708, 376782371),
                },
                Data {
                    i: 180559379,
                    float: 7.435722e-29,
                    ints: (3157076886, 3180075169),
                },
                Data {
                    i: 1788212367,
                    float: 8548018000000.0,
                    ints: (909768672, 2407530360),
                },
                Data {
                    i: 3349700732,
                    float: 0.0056111077,
                    ints: (3961752753, 1419761496),
                },
                Data {
                    i: 3801520128,
                    float: 98785.83,
                    ints: (1932993929, 2289220397),
                },
                Data {
                    i: 3337060221,
                    float: 3535689300000000.0,
                    ints: (930157246, 1575967756),
                },
                Data {
                    i: 1468146607,
                    float: 0.0007737411,
                    ints: (1051619328, 1202274613),
                },
                Data {
                    i: 1087913324,
                    float: 1.7668136e32,
                    ints: (4721339, 2725508123),
                },
                Data {
                    i: 1465596176,
                    float: 1.8388995e-35,
                    ints: (4230359612, 582358467),
                },
                Data {
                    i: 4220578792,
                    float: 1.1517518,
                    ints: (1124726459, 2560547821),
                },
                Data {
                    i: 2026862081,
                    float: 3.4949498e-32,
                    ints: (3666872380, 3323468334),
                },
                Data {
                    i: 3074274145,
                    float: 3.2394806e-14,
                    ints: (425270349, 3373048350),
                },
                Data {
                    i: 3648912328,
                    float: 6.423049e-14,
                    ints: (480312788, 2820306273),
                },
                Data {
                    i: 1468248262,
                    float: 1235242000000000.0,
                    ints: (2717965090, 4141428586),
                },
                Data {
                    i: 4068086748,
                    float: 3.3334652e-19,
                    ints: (234690818, 1388889175),
                },
                Data {
                    i: 3009499003,
                    float: 2.544228e16,
                    ints: (2303100902, 49335513),
                },
                Data {
                    i: 3598931111,
                    float: 0.0016578501,
                    ints: (3409193162, 4213775674),
                },
            ],
            vec![
                ArrayChange {
                    i: 8,
                    change: DataChange {
                        ty: Ints,
                        data: [1968835577, 2900750072, 0],
                    },
                },
                ArrayChange {
                    i: 0,
                    change: DataChange {
                        ty: Float,
                        data: [1498661606, 0, 0],
                    },
                },
                ArrayChange {
                    i: 15,
                    change: DataChange {
                        ty: Float,
                        data: [1903241132, 0, 0],
                    },
                },
                ArrayChange {
                    i: 0,
                    change: DataChange {
                        ty: Ints,
                        data: [2859467695, 1456411874, 0],
                    },
                },
                ArrayChange {
                    i: 2,
                    change: DataChange {
                        ty: I,
                        data: [503672344, 0, 0],
                    },
                },
                ArrayChange {
                    i: 13,
                    change: DataChange {
                        ty: Ints,
                        data: [966520065, 3879411604, 0],
                    },
                },
                ArrayChange {
                    i: 14,
                    change: DataChange {
                        ty: Ints,
                        data: [21492386, 1425506613, 0],
                    },
                },
                ArrayChange {
                    i: 14,
                    change: DataChange {
                        ty: Ints,
                        data: [1820089982, 1095894710, 0],
                    },
                },
                ArrayChange {
                    i: 15,
                    change: DataChange {
                        ty: Ints,
                        data: [1199771904, 2309304543, 0],
                    },
                },
                ArrayChange {
                    i: 5,
                    change: DataChange {
                        ty: Ints,
                        data: [823903755, 425267376, 0],
                    },
                },
                ArrayChange {
                    i: 12,
                    change: DataChange {
                        ty: I,
                        data: [4216456839, 0, 0],
                    },
                },
                ArrayChange {
                    i: 7,
                    change: DataChange {
                        ty: Ints,
                        data: [2840241586, 3550013929, 0],
                    },
                },
                ArrayChange {
                    i: 2,
                    change: DataChange {
                        ty: Ints,
                        data: [2140276761, 1168666498, 0],
                    },
                },
                ArrayChange {
                    i: 8,
                    change: DataChange {
                        ty: I,
                        data: [3092883570, 0, 0],
                    },
                },
                ArrayChange {
                    i: 3,
                    change: DataChange {
                        ty: Ints,
                        data: [2326395260, 3664958639, 0],
                    },
                },
                ArrayChange {
                    i: 4,
                    change: DataChange {
                        ty: Ints,
                        data: [2932301963, 302485527, 0],
                    },
                },
                ArrayChange {
                    i: 0,
                    change: DataChange {
                        ty: Ints,
                        data: [4197463277, 2380103846, 0],
                    },
                },
                ArrayChange {
                    i: 11,
                    change: DataChange {
                        ty: Float,
                        data: [826891420, 0, 0],
                    },
                },
                ArrayChange {
                    i: 8,
                    change: DataChange {
                        ty: Float,
                        data: [304030882, 0, 0],
                    },
                },
                ArrayChange {
                    i: 14,
                    change: DataChange {
                        ty: Ints,
                        data: [1211686747, 2950484429, 0],
                    },
                },
                ArrayChange {
                    i: 4,
                    change: DataChange {
                        ty: Ints,
                        data: [2147876464, 749597549, 0],
                    },
                },
                ArrayChange {
                    i: 1,
                    change: DataChange {
                        ty: Ints,
                        data: [667946860, 1598710426, 0],
                    },
                },
                ArrayChange {
                    i: 14,
                    change: DataChange {
                        ty: Float,
                        data: [917280002, 0, 0],
                    },
                },
                ArrayChange {
                    i: 1,
                    change: DataChange {
                        ty: Ints,
                        data: [1960934252, 1394552955, 0],
                    },
                },
            ],
        ),
        Array(
            vec![
                Data {
                    i: 2850609091,
                    float: 9600226.0,
                    ints: (3308497712, 3789756841),
                },
                Data {
                    i: 3942780682,
                    float: 0.0008439499,
                    ints: (2263470273, 4037436481),
                },
                Data {
                    i: 3611835213,
                    float: 7.8243366e-16,
                    ints: (2559791737, 1473107258),
                },
                Data {
                    i: 1437663869,
                    float: 5429994.0,
                    ints: (2841400029, 2946874104),
                },
                Data {
                    i: 1254695614,
                    float: 1.4532201e-13,
                    ints: (1195645356, 2331873899),
                },
                Data {
                    i: 3361176297,
                    float: 2.916837e-22,
                    ints: (523692148, 515106305),
                },
                Data {
                    i: 1319817484,
                    float: 2.6343316e31,
                    ints: (4253490503, 2810726620),
                },
                Data {
                    i: 702724902,
                    float: 1.4715766e27,
                    ints: (3448585558, 982533235),
                },
                Data {
                    i: 1016956924,
                    float: 2.1316158e-8,
                    ints: (2365614184, 2060194212),
                },
                Data {
                    i: 558420069,
                    float: 1.811508e23,
                    ints: (3908142574, 3078653874),
                },
                Data {
                    i: 860822432,
                    float: 4.3380583e-35,
                    ints: (1854904222, 1928226020),
                },
                Data {
                    i: 2413360816,
                    float: 2.8937698e28,
                    ints: (4011153622, 3720976613),
                },
                Data {
                    i: 2050619237,
                    float: 7.1833885e35,
                    ints: (1129110551, 4169639369),
                },
                Data {
                    i: 97044957,
                    float: 1.7826627e-29,
                    ints: (3933076027, 1509088521),
                },
                Data {
                    i: 558235271,
                    float: 1.759464e-19,
                    ints: (1344136395, 593141983),
                },
                Data {
                    i: 577142441,
                    float: 5.6447323e-8,
                    ints: (71152795, 1673018549),
                },
                Data {
                    i: 2075566409,
                    float: 2.1397346e30,
                    ints: (2494517345, 1988855335),
                },
                Data {
                    i: 3383346648,
                    float: 6.6023373e34,
                    ints: (4147443238, 2203189924),
                },
                Data {
                    i: 2205563124,
                    float: 5.073123e-26,
                    ints: (1243813715, 2315139210),
                },
                Data {
                    i: 3718895381,
                    float: 4.672435e28,
                    ints: (3654689360, 686447571),
                },
                Data {
                    i: 997197664,
                    float: 9.1579374e-18,
                    ints: (2215267596, 4168349367),
                },
                Data {
                    i: 4206927252,
                    float: 1.8409946e-8,
                    ints: (152366977, 2073875828),
                },
                Data {
                    i: 428454071,
                    float: 1.05472816e27,
                    ints: (2717990916, 85691760),
                },
                Data {
                    i: 1958891020,
                    float: 1138748.9,
                    ints: (2720884709, 1564401688),
                },
                Data {
                    i: 739270771,
                    float: 14522880000.0,
                    ints: (151469083, 3589953659),
                },
                Data {
                    i: 2518206525,
                    float: 1.5879325e-26,
                    ints: (3868576640, 611894180),
                },
                Data {
                    i: 3166699656,
                    float: 462644640000.0,
                    ints: (2834461184, 2854783191),
                },
                Data {
                    i: 1097040171,
                    float: 30050344.0,
                    ints: (3849877731, 4150800508),
                },
                Data {
                    i: 2101337156,
                    float: 28212519000.0,
                    ints: (3148640553, 1172632557),
                },
            ],
            vec![
                ArrayChange {
                    i: 23,
                    change: DataChange {
                        ty: Ints,
                        data: [3948276569, 896035092, 0],
                    },
                },
                ArrayChange {
                    i: 5,
                    change: DataChange {
                        ty: I,
                        data: [3281366555, 0, 0],
                    },
                },
                ArrayChange {
                    i: 19,
                    change: DataChange {
                        ty: Ints,
                        data: [3863653157, 2040935085, 0],
                    },
                },
                ArrayChange {
                    i: 13,
                    change: DataChange {
                        ty: Ints,
                        data: [47697667, 1281074468, 0],
                    },
                },
            ],
        ),
        Array(
            vec![
                Data {
                    i: 4108700821,
                    float: 2.3532795e-23,
                    ints: (1989162274, 3201965851),
                },
                Data {
                    i: 819786153,
                    float: 1.7842965e-13,
                    ints: (682667252, 3934773945),
                },
                Data {
                    i: 1859444907,
                    float: 1.515573,
                    ints: (317317405, 421747710),
                },
                Data {
                    i: 837373864,
                    float: 9.64573e-14,
                    ints: (1144163242, 3005001286),
                },
                Data {
                    i: 1711209274,
                    float: 1.7474194e23,
                    ints: (3839431954, 4006808515),
                },
                Data {
                    i: 3781749850,
                    float: 1.4423151e21,
                    ints: (45193974, 726023514),
                },
                Data {
                    i: 1856525474,
                    float: 70202770000.0,
                    ints: (1826197542, 19190429),
                },
                Data {
                    i: 3547409817,
                    float: 56608495000000.0,
                    ints: (2033120147, 863032527),
                },
                Data {
                    i: 3296783013,
                    float: 2.8188155e-16,
                    ints: (159997448, 2886670451),
                },
                Data {
                    i: 3605150887,
                    float: 2.7599584e20,
                    ints: (508021810, 1058143563),
                },
                Data {
                    i: 1721293032,
                    float: 0.0006606878,
                    ints: (144615506, 1487777951),
                },
                Data {
                    i: 3969907373,
                    float: 7.758692e23,
                    ints: (4164460581, 4206815198),
                },
                Data {
                    i: 2764813943,
                    float: 3.90077e26,
                    ints: (4258789077, 3652240748),
                },
                Data {
                    i: 17956065,
                    float: 1.8622266e25,
                    ints: (898081428, 4284629345),
                },
                Data {
                    i: 2010289761,
                    float: 57991860.0,
                    ints: (317357075, 2285118590),
                },
                Data {
                    i: 1320780029,
                    float: 3.327365e-19,
                    ints: (4157117119, 530027988),
                },
            ],
            vec![
                ArrayChange {
                    i: 13,
                    change: DataChange {
                        ty: Float,
                        data: [307569050, 0, 0],
                    },
                },
                ArrayChange {
                    i: 15,
                    change: DataChange {
                        ty: Ints,
                        data: [2273120895, 4232503072, 0],
                    },
                },
                ArrayChange {
                    i: 12,
                    change: DataChange {
                        ty: Ints,
                        data: [817756117, 827269799, 0],
                    },
                },
                ArrayChange {
                    i: 1,
                    change: DataChange {
                        ty: Ints,
                        data: [1444728801, 4240285268, 0],
                    },
                },
                ArrayChange {
                    i: 1,
                    change: DataChange {
                        ty: I,
                        data: [642816015, 0, 0],
                    },
                },
                ArrayChange {
                    i: 13,
                    change: DataChange {
                        ty: I,
                        data: [1130952027, 0, 0],
                    },
                },
                ArrayChange {
                    i: 5,
                    change: DataChange {
                        ty: Ints,
                        data: [1707292705, 3623275308, 0],
                    },
                },
                ArrayChange {
                    i: 8,
                    change: DataChange {
                        ty: Float,
                        data: [574497434, 0, 0],
                    },
                },
                ArrayChange {
                    i: 6,
                    change: DataChange {
                        ty: Ints,
                        data: [4266403751, 3628814659, 0],
                    },
                },
                ArrayChange {
                    i: 8,
                    change: DataChange {
                        ty: I,
                        data: [2966437175, 0, 0],
                    },
                },
                ArrayChange {
                    i: 4,
                    change: DataChange {
                        ty: I,
                        data: [2092493111, 0, 0],
                    },
                },
                ArrayChange {
                    i: 2,
                    change: DataChange {
                        ty: Float,
                        data: [950114045, 0, 0],
                    },
                },
                ArrayChange {
                    i: 7,
                    change: DataChange {
                        ty: I,
                        data: [3603614393, 0, 0],
                    },
                },
                ArrayChange {
                    i: 14,
                    change: DataChange {
                        ty: I,
                        data: [3993823232, 0, 0],
                    },
                },
                ArrayChange {
                    i: 5,
                    change: DataChange {
                        ty: Ints,
                        data: [3869302782, 1210180853, 0],
                    },
                },
                ArrayChange {
                    i: 15,
                    change: DataChange {
                        ty: Ints,
                        data: [652015964, 1450905939, 0],
                    },
                },
                ArrayChange {
                    i: 10,
                    change: DataChange {
                        ty: I,
                        data: [4026587813, 0, 0],
                    },
                },
                ArrayChange {
                    i: 5,
                    change: DataChange {
                        ty: Ints,
                        data: [4289369174, 1169045052, 0],
                    },
                },
                ArrayChange {
                    i: 7,
                    change: DataChange {
                        ty: Ints,
                        data: [793573633, 3564135190, 0],
                    },
                },
                ArrayChange {
                    i: 11,
                    change: DataChange {
                        ty: I,
                        data: [1759424871, 0, 0],
                    },
                },
                ArrayChange {
                    i: 6,
                    change: DataChange {
                        ty: Ints,
                        data: [1957025788, 4130199034, 0],
                    },
                },
                ArrayChange {
                    i: 13,
                    change: DataChange {
                        ty: Ints,
                        data: [1541813436, 1911436948, 0],
                    },
                },
                ArrayChange {
                    i: 5,
                    change: DataChange {
                        ty: Float,
                        data: [840482139, 0, 0],
                    },
                },
                ArrayChange {
                    i: 14,
                    change: DataChange {
                        ty: Ints,
                        data: [3431179236, 1773335669, 0],
                    },
                },
            ],
        ),
        Single(
            Data {
                i: 3327916968,
                float: 3.1761143e-23,
                ints: (2961536465, 4294940558),
            },
            vec![
                DataChange {
                    ty: Ints,
                    data: [102664037, 692397628, 0],
                },
                DataChange {
                    ty: I,
                    data: [2041263296, 0, 0],
                },
                DataChange {
                    ty: I,
                    data: [3822420069, 0, 0],
                },
                DataChange {
                    ty: Float,
                    data: [1188894174, 0, 0],
                },
                DataChange {
                    ty: Ints,
                    data: [3360219795, 1629833370, 0],
                },
                DataChange {
                    ty: I,
                    data: [3470123572, 0, 0],
                },
                DataChange {
                    ty: Ints,
                    data: [2626156937, 812016440, 0],
                },
                DataChange {
                    ty: Ints,
                    data: [3984947695, 725750800, 0],
                },
                DataChange {
                    ty: Float,
                    data: [1319038062, 0, 0],
                },
                DataChange {
                    ty: I,
                    data: [264795148, 0, 0],
                },
                DataChange {
                    ty: Float,
                    data: [2060912298, 0, 0],
                },
            ],
        ),
        Single(
            Data {
                i: 1531136390,
                float: 1.0477427e-14,
                ints: (1786779511, 2275420017),
            },
            vec![
                DataChange {
                    ty: Float,
                    data: [223828336, 0, 0],
                },
                DataChange {
                    ty: Ints,
                    data: [2730943259, 2498707315, 0],
                },
                DataChange {
                    ty: Ints,
                    data: [3275793585, 3036608748, 0],
                },
                DataChange {
                    ty: Float,
                    data: [1306922164, 0, 0],
                },
                DataChange {
                    ty: Float,
                    data: [1355349787, 0, 0],
                },
                DataChange {
                    ty: Ints,
                    data: [3601681471, 1699718073, 0],
                },
                DataChange {
                    ty: Ints,
                    data: [1128518784, 2008977347, 0],
                },
                DataChange {
                    ty: I,
                    data: [1311400651, 0, 0],
                },
                DataChange {
                    ty: I,
                    data: [2584961649, 0, 0],
                },
                DataChange {
                    ty: Ints,
                    data: [4030176026, 2676988480, 0],
                },
                DataChange {
                    ty: I,
                    data: [99290228, 0, 0],
                },
                DataChange {
                    ty: Float,
                    data: [47282897, 0, 0],
                },
                DataChange {
                    ty: Ints,
                    data: [2766945847, 4229656173, 0],
                },
                DataChange {
                    ty: Float,
                    data: [571159051, 0, 0],
                },
                DataChange {
                    ty: Ints,
                    data: [1311210421, 1103950828, 0],
                },
                DataChange {
                    ty: Ints,
                    data: [654292856, 2604374611, 0],
                },
                DataChange {
                    ty: Float,
                    data: [424717588, 0, 0],
                },
                DataChange {
                    ty: Ints,
                    data: [3009121784, 3190772463, 0],
                },
                DataChange {
                    ty: I,
                    data: [1733158341, 0, 0],
                },
                DataChange {
                    ty: I,
                    data: [2933666233, 0, 0],
                },
                DataChange {
                    ty: I,
                    data: [3874947890, 0, 0],
                },
                DataChange {
                    ty: Float,
                    data: [433120522, 0, 0],
                },
                DataChange {
                    ty: Ints,
                    data: [2653319291, 2237007380, 0],
                },
                DataChange {
                    ty: I,
                    data: [1957827346, 0, 0],
                },
                DataChange {
                    ty: Ints,
                    data: [3096485923, 1207032526, 0],
                },
                DataChange {
                    ty: Ints,
                    data: [1980091155, 3598982332, 0],
                },
                DataChange {
                    ty: Ints,
                    data: [571343744, 2234545222, 0],
                },
                DataChange {
                    ty: Ints,
                    data: [782679408, 3180379333, 0],
                },
                DataChange {
                    ty: Ints,
                    data: [852462275, 563722289, 0],
                },
                DataChange {
                    ty: Ints,
                    data: [3547192514, 3297928195, 0],
                },
                DataChange {
                    ty: I,
                    data: [2484021413, 0, 0],
                },
            ],
        ),
        Single(
            Data {
                i: 2232699745,
                float: 3.144831e23,
                ints: (264343566, 3440978791),
            },
            vec![
                DataChange {
                    ty: I,
                    data: [45553698, 0, 0],
                },
                DataChange {
                    ty: Float,
                    data: [1776183808, 0, 0],
                },
                DataChange {
                    ty: Ints,
                    data: [1665279516, 4219759138, 0],
                },
                DataChange {
                    ty: Ints,
                    data: [3669877590, 706926532, 0],
                },
                DataChange {
                    ty: Ints,
                    data: [3082769937, 3909180387, 0],
                },
                DataChange {
                    ty: Ints,
                    data: [3170206966, 3301291954, 0],
                },
                DataChange {
                    ty: Ints,
                    data: [3481243354, 3190755354, 0],
                },
                DataChange {
                    ty: Ints,
                    data: [1857916888, 2575948146, 0],
                },
                DataChange {
                    ty: Ints,
                    data: [1935275872, 1397870338, 0],
                },
                DataChange {
                    ty: Ints,
                    data: [933453898, 2636078958, 0],
                },
                DataChange {
                    ty: Ints,
                    data: [3652534480, 1281740419, 0],
                },
                DataChange {
                    ty: Ints,
                    data: [1102843157, 3010455474, 0],
                },
                DataChange {
                    ty: Float,
                    data: [2054017825, 0, 0],
                },
                DataChange {
                    ty: Float,
                    data: [1014630347, 0, 0],
                },
                DataChange {
                    ty: Float,
                    data: [136043821, 0, 0],
                },
                DataChange {
                    ty: Float,
                    data: [462639233, 0, 0],
                },
                DataChange {
                    ty: Ints,
                    data: [2110757351, 183519806, 0],
                },
                DataChange {
                    ty: I,
                    data: [1036049705, 0, 0],
                },
                DataChange {
                    ty: Ints,
                    data: [1593898453, 2086815312, 0],
                },
                DataChange {
                    ty: Ints,
                    data: [577204420, 817715446, 0],
                },
                DataChange {
                    ty: Float,
                    data: [1863305386, 0, 0],
                },
                DataChange {
                    ty: Ints,
                    data: [2085033689, 2799173270, 0],
                },
                DataChange {
                    ty: Ints,
                    data: [4227154321, 2053459804, 0],
                },
                DataChange {
                    ty: Float,
                    data: [282588424, 0, 0],
                },
                DataChange {
                    ty: I,
                    data: [1592488329, 0, 0],
                },
                DataChange {
                    ty: Ints,
                    data: [1800987635, 3125375625, 0],
                },
            ],
        ),
        Array(
            vec![
                Data {
                    i: 1927516848,
                    float: 2.1232194e-33,
                    ints: (374226535, 1408803975),
                },
                Data {
                    i: 3033075754,
                    float: 2.1709772e21,
                    ints: (2646448684, 2634048262),
                },
                Data {
                    i: 1479906731,
                    float: 1.0666891e-37,
                    ints: (3341584256, 3324684483),
                },
                Data {
                    i: 3889331836,
                    float: 1.1067571e-7,
                    ints: (990295164, 4212703750),
                },
                Data {
                    i: 430895737,
                    float: 2.1727349e24,
                    ints: (348221601, 1304208859),
                },
            ],
            vec![
                ArrayChange {
                    i: 0,
                    change: DataChange {
                        ty: Float,
                        data: [374179447, 0, 0],
                    },
                },
                ArrayChange {
                    i: 0,
                    change: DataChange {
                        ty: Ints,
                        data: [3449435443, 2767415573, 0],
                    },
                },
                ArrayChange {
                    i: 2,
                    change: DataChange {
                        ty: Float,
                        data: [40923417, 0, 0],
                    },
                },
                ArrayChange {
                    i: 2,
                    change: DataChange {
                        ty: Float,
                        data: [459075998, 0, 0],
                    },
                },
                ArrayChange {
                    i: 1,
                    change: DataChange {
                        ty: I,
                        data: [3594247999, 0, 0],
                    },
                },
                ArrayChange {
                    i: 4,
                    change: DataChange {
                        ty: I,
                        data: [1852840333, 0, 0],
                    },
                },
                ArrayChange {
                    i: 0,
                    change: DataChange {
                        ty: Ints,
                        data: [1388893112, 1424126266, 0],
                    },
                },
                ArrayChange {
                    i: 3,
                    change: DataChange {
                        ty: Float,
                        data: [1688647724, 0, 0],
                    },
                },
                ArrayChange {
                    i: 0,
                    change: DataChange {
                        ty: Ints,
                        data: [2977964735, 3700036420, 0],
                    },
                },
                ArrayChange {
                    i: 4,
                    change: DataChange {
                        ty: Float,
                        data: [1405562018, 0, 0],
                    },
                },
                ArrayChange {
                    i: 0,
                    change: DataChange {
                        ty: Float,
                        data: [344367331, 0, 0],
                    },
                },
                ArrayChange {
                    i: 2,
                    change: DataChange {
                        ty: Float,
                        data: [876311006, 0, 0],
                    },
                },
                ArrayChange {
                    i: 2,
                    change: DataChange {
                        ty: I,
                        data: [2716295373, 0, 0],
                    },
                },
                ArrayChange {
                    i: 0,
                    change: DataChange {
                        ty: Float,
                        data: [1568409498, 0, 0],
                    },
                },
                ArrayChange {
                    i: 4,
                    change: DataChange {
                        ty: Ints,
                        data: [319187780, 2478894784, 0],
                    },
                },
                ArrayChange {
                    i: 1,
                    change: DataChange {
                        ty: Ints,
                        data: [2425686603, 1035541092, 0],
                    },
                },
                ArrayChange {
                    i: 0,
                    change: DataChange {
                        ty: I,
                        data: [232482543, 0, 0],
                    },
                },
                ArrayChange {
                    i: 2,
                    change: DataChange {
                        ty: Float,
                        data: [1738946162, 0, 0],
                    },
                },
                ArrayChange {
                    i: 2,
                    change: DataChange {
                        ty: Float,
                        data: [1361456668, 0, 0],
                    },
                },
                ArrayChange {
                    i: 0,
                    change: DataChange {
                        ty: Ints,
                        data: [2841395469, 2414228912, 0],
                    },
                },
                ArrayChange {
                    i: 4,
                    change: DataChange {
                        ty: I,
                        data: [2454305667, 0, 0],
                    },
                },
                ArrayChange {
                    i: 0,
                    change: DataChange {
                        ty: Ints,
                        data: [3306884445, 2510102584, 0],
                    },
                },
                ArrayChange {
                    i: 0,
                    change: DataChange {
                        ty: I,
                        data: [416331468, 0, 0],
                    },
                },
                ArrayChange {
                    i: 0,
                    change: DataChange {
                        ty: Ints,
                        data: [887761361, 1131566633, 0],
                    },
                },
                ArrayChange {
                    i: 2,
                    change: DataChange {
                        ty: I,
                        data: [3867337541, 0, 0],
                    },
                },
                ArrayChange {
                    i: 4,
                    change: DataChange {
                        ty: Ints,
                        data: [2996890142, 292271156, 0],
                    },
                },
            ],
        ),
        Array(
            vec![
                Data {
                    i: 965743161,
                    float: 3.6483124e-28,
                    ints: (1977386712, 3121834325),
                },
                Data {
                    i: 2538938120,
                    float: 1.4983756e37,
                    ints: (1308659737, 3794926245),
                },
                Data {
                    i: 2574183155,
                    float: 2.2444947e-25,
                    ints: (1562594067, 2372920284),
                },
                Data {
                    i: 1603412132,
                    float: 5.6606537e-9,
                    ints: (350831498, 3810736981),
                },
                Data {
                    i: 2271848450,
                    float: 7.420961e22,
                    ints: (88531621, 3785262504),
                },
            ],
            vec![
                ArrayChange {
                    i: 4,
                    change: DataChange {
                        ty: I,
                        data: [2669980058, 0, 0],
                    },
                },
                ArrayChange {
                    i: 0,
                    change: DataChange {
                        ty: Float,
                        data: [1101793094, 0, 0],
                    },
                },
                ArrayChange {
                    i: 3,
                    change: DataChange {
                        ty: Ints,
                        data: [3999758903, 36725604, 0],
                    },
                },
                ArrayChange {
                    i: 4,
                    change: DataChange {
                        ty: Ints,
                        data: [1339848159, 3496571880, 0],
                    },
                },
                ArrayChange {
                    i: 4,
                    change: DataChange {
                        ty: I,
                        data: [666, 0, 0],
                    },
                },
            ],
        ),
    ]
}

use glam::UVec3;

#[test]
fn workgroup_dimensions_to_id_sanity() {
    let mut indices = vec![];
    let size = ApplyDataChangeInvocation::WORKGROUP_SIZE;
    for z in 0..size.z {
        for y in 0..size.y {
            for x in 0..size.x {
                let global_id = UVec3::new(x, y, z);
                let index = ApplyDataChangeInvocation::index(global_id);
                indices.push(index);
            }
        }
    }
    pretty_assertions::assert_eq!((0..(size.x * size.y * size.z)).collect::<Vec<_>>(), indices);
}

#[test]
fn invocations_sanity() {
    let changes = vec![one_datum()];
    let runtime = crate::wgpu_runtime();
    let arena = Arena::new(&runtime, "test", None);
    let mut test = GpuUpdateTest::new(arena, TestBackendWgpu::new(runtime), &changes);
    test.arena.commit();
    let changes = test.apply_raw_changes();
    test.update_changes_on_backend(changes);
    test.apply_backend_changes();
    let apply_invocation = test.invocation.get();
    let workgroups = apply_invocation.workgroup_dimensions();
    assert_eq!(UVec3::ONE, workgroups);

    let invocations_ran =
        futures_lite::future::block_on(test.arena.read_slab(test.invocation_count.array()))
            .unwrap()
            .pop()
            .unwrap();
    let invocations_skipped =
        futures_lite::future::block_on(test.arena.read_slab(test.invocations_skipped.array()))
            .unwrap()
            .pop()
            .unwrap();
    let total_invocations = invocations_ran + invocations_skipped;

    assert_eq!(
        (16, 1, 15),
        (total_invocations, invocations_ran, invocations_skipped),
    );
}

#[test]
/// Here to aid in addressing regressions caught by proptest
fn regression() {
    let data = data();

    {
        log::info!("START CPU TEST");
        let arena = Arena::new(&CpuRuntime, "test", None);
        let test = GpuUpdateTest::new(arena, (), &data);
        test.run(true);
    }

    {
        log::info!("START GPU TEST");
        let runtime = crate::wgpu_runtime();
        let arena = Arena::new(&runtime, "test", None);
        let test = GpuUpdateTest::new(arena, TestBackendWgpu::new(runtime), &data);
        test.run(true);
    }
}
