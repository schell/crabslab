use std::ops::{Deref, DerefMut};

use craballoc_test_wire_types::{
    AnyChangeId, ApplyDataChangeInvocation, ArrayChange, Data, DataChange, DataChangeTy,
};
use crabslab::{Array, Id};
use glam::UVec3;
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
    let data = vec![one_datum()];

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
