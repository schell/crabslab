use craballoc_test_wire_types::{ArrayChange, Data, DataChange};
use crabslab::{Array, Id, Slab};
use proptest::{
    prelude::{Just, Strategy},
    prop_compose, proptest,
};

use crate::{
    arena::{Arena, Value},
    range::Range,
    runtime::{CpuRuntime, IsRuntime},
};

#[test]
fn mngr_updates_count_sanity() {
    let slab = Arena::new(CpuRuntime, "sanity", ());
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
        let arena = Arena::new(crate::wgpu_runtime(), "test", wgpu::BufferUsages::empty());
        let values = arena.new_array(0u32..=9);
        let _ = arena.commit();
        let from_gpu = futures_lite::future::block_on(arena.read_slab(values.array())).unwrap();
        let from_cpu = values.read_range(.., |ts| ts.to_vec());
        assert_eq!(from_cpu, from_gpu);
    }
    {
        // Do it with `Data`
        let arena = Arena::new(crate::wgpu_runtime(), "test", wgpu::BufferUsages::empty());
        let values = arena.new_array([
            Data {
                i: 0,
                float: 0.0,
                int: 0,
            },
            Data {
                i: 1,
                float: 1.0,
                int: 1,
            },
            Data {
                i: 2,
                float: 2.0,
                int: 2,
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

    let m = Arena::new(CpuRuntime, "sanity", ());
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
    let m = Arena::new(CpuRuntime, "sanity", ());
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
    let slab = Arena::new(CpuRuntime, "test", ());
    let mut initial_values = vec![
        Data {
            i: 0,
            float: 1.0,
            int: 1,
        },
        Data {
            i: 1,
            float: 2.0,
            int: 2,
        },
        Data {
            i: 2,
            float: 3.0,
            int: 3,
        },
    ];

    log::info!("staging initial values");
    let values = slab.new_array(initial_values.clone());
    ensure!(slab, initial_values, values);

    log::info!("updating the initial values");
    // change the initial values
    initial_values[1].int = 666;
    initial_values[2].int = 420;
    // modify the slab values to match
    values.modify_range(1u32..3, |items| {
        items[0].int = 666;
        items[1].int = 420;
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
    initial_values[2].int = -32;
    values.modify_item(2, |data| data.int = -32);
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
    let slab = Arena::new(crate::wgpu_runtime(), "test", wgpu::BufferUsages::empty());
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
        proptest::num::i64::ANY,
    )
        .prop_map(|(i, float, int)| Data { i, float, int })
}

fn arb_array_data(max_length: usize) -> impl Strategy<Value = Vec<Data>> {
    proptest::collection::vec(arb_data(), 1..max_length)
}

fn arb_data_change() -> impl Strategy<Value = DataChange> {
    let i = proptest::num::u32::ANY.prop_map(DataChange::I).boxed();
    let float = proptest::num::f32::NORMAL
        .prop_map(DataChange::Float)
        .boxed();
    let int = proptest::num::i64::ANY.prop_map(DataChange::Int).boxed();
    i.prop_union(float).boxed().prop_union(int)
}

fn arb_array_change(array_len: usize) -> impl Strategy<Value = ArrayChange> {
    (0..array_len, arb_data_change()).prop_map(|(i, change)| ArrayChange { i, change })
}

#[derive(Debug)]
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

#[derive(Debug)]
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

trait BackendUpdate {
    /// Apply one step's worth changes to the backend, without the frontend's knowledge,
    /// in order to test the synchronization of the backend to the frontend.
    ///
    /// * In the case of CpuRuntime, this should alter the underlying Vector slab.
    /// * In the case of WgpuRuntime, this should pack a "change buffer" and invoke
    ///   a compute shader that performs the updates on the GPU.
    ///
    /// Returns `true` if any changes were made or are queued.
    ///
    /// Returns `false` if no changes were made and no changes are queued.
    fn apply_backend_changes(&mut self, changes: Vec<BackendChange>);
}

impl BackendUpdate for GpuUpdateTest<CpuRuntime, ()> {
    fn apply_backend_changes(&mut self, changes: Vec<BackendChange>) {
        log::trace!("  applying CPU backend changes");
        let buffer = self.arena.get_buffer().unwrap();
        let mut slab = buffer.as_mut_vec();
        for change in changes.into_iter() {
            match change {
                BackendChange::Single { id, change } => {
                    log::trace!("    applying {change:?} to {id:?}");
                    let mut data = slab.read_unchecked(id);
                    change.apply(&mut data);
                    log::trace!("      data: {data:?}");
                    slab.write(id, &data);
                }
                BackendChange::Array { array, change } => {
                    log::trace!("    applying {change:?} to {array:?}");
                    let id = array.at(change.i);
                    let mut data = slab.read_unchecked(id);
                    change.change.apply(&mut data);
                    log::trace!("      data: {data:?}");
                    slab.write(id, &data);
                }
            }
        }
        log::trace!("  done!");
    }
}

struct GpuUpdateTest<R: IsRuntime, T> {
    arena: Arena<R>,
    all_values: Vec<Values>,
    backend_updater: T,
}

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

impl<R: IsRuntime, T> GpuUpdateTest<R, T> {
    /// Apply one step's worth of changes to the raw values and return the changes to be
    /// made by the backend.
    fn apply_raw_changes(&mut self) -> Vec<BackendChange> {
        let mut backend_changes = vec![];
        for value in self.all_values.iter_mut() {
            match value {
                Values::Single {
                    arena,
                    raw,
                    changes,
                } => {
                    if let Some(change) = changes.pop() {
                        log::trace!("  applying change {change:?} to single value");
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
                        log::trace!("  applying change {change:?} to array value");
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

    /// Asserts that the **CPU values** have the changes made by the backend.
    ///
    /// Essentially, ensures that `Arena::synchronize` was implemented correctly.
    fn verify(&self) {
        for value in self.all_values.iter() {
            match value {
                Values::Single {
                    arena,
                    raw,
                    changes: _,
                } => {
                    pretty_assertions::assert_eq!(raw, &arena.get())
                }
                Values::Array {
                    arena,
                    raw,
                    changes: _,
                } => {
                    pretty_assertions::assert_eq!(raw, &arena.get_vec());
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

        let test = GpuUpdateTest {
            arena,
            all_values,
            backend_updater,
        };
        test.verify();
        test
    }

    /// `verify_each_step = true` will commit, synchronize and compare after each loop,
    /// otherwise all changes are applied and then one commit+synchronize+compare is done at the end
    fn run(mut self, verify_each_step: bool)
    where
        Self: BackendUpdate,
    {
        // Send the initial CPU values to the backend.
        log::trace!("initial commit");
        self.arena.commit();
        self.verify();

        let mut steps = 0;
        loop {
            log::trace!("step {steps}");
            let backend_changes = self.apply_raw_changes();
            if backend_changes.is_empty() {
                log::trace!("  ({steps}) applied all changes");
                break;
            }

            // Apply the changes **on the backend** and synchronize to bring
            // them back to the CPU
            self.apply_backend_changes(backend_changes);
            futures_lite::future::block_on(self.arena.synchronize()).unwrap();

            if verify_each_step {
                log::trace!("  {steps}");
                self.verify();
            }
            steps += 1;
        }
        self.verify();
    }
}

#[test]
/// Run just one update and ensure we get results we expect.
fn gpu_update_test_sanity() {
    let _ = env_logger::builder().is_test(true).try_init();
    let arena = Arena::new(CpuRuntime, "test", ());
    let all_values = vec![ValueData::Single(
        Data {
            i: 0,
            float: 0.0,
            int: 0,
        },
        vec![DataChange::I(1), DataChange::Float(1.0), DataChange::Int(1)],
    )];
    let test = GpuUpdateTest::new(arena, (), &all_values);
    test.run(true);
}

proptest! {
    #[test]
    fn proptest_gpu_updates_checked_on_cpu(value_data in proptest::collection::vec(arb_value_data(32, 32), 1..32)) {
        let _ = env_logger::builder().is_test(true).try_init();
        {
            let arena = Arena::new(CpuRuntime, "test", ());
            let test = GpuUpdateTest::new(arena, (), &value_data);
            test.run(true);
        }
        {
            let arena = Arena::new(CpuRuntime, "test", ());
            let test = GpuUpdateTest::new(arena, (), &value_data);
            test.run(false);
        }
    }
}
