use std::ops::DerefMut;

use proptest::{
    prelude::{Just, Strategy},
    prop_compose, proptest,
};
use wgsl_rs::std::{atomic_load, Atomic, RuntimeArray};

use crate::{
    arena::{Arena, Value},
    range::Range,
    runtime::{CpuRuntime, IsRuntime},
};

// ---------------------------------------------------------------------------
// Wire types + compute shader — single #[slab_module(wgsl(...))] module.
// ---------------------------------------------------------------------------

#[allow(clippy::needless_late_init)]
#[crabslab::slab_module(wgsl())]
pub mod apply_data_changes {
    use wgsl_rs::std::*;

    // -- Wire types --

    #[slab_item]
    #[derive(Clone, Copy, Debug, Default, PartialEq)]
    pub struct Data {
        pub i: u32,
        pub float_val: f32,
        pub ints_0: u32,
        pub ints_1: u32,
    }

    #[slab_item]
    #[derive(Clone, Copy, Debug, Default)]
    #[repr(u32)]
    pub enum DataChangeTy {
        #[default]
        I = 0,
        Float = 1,
        Ints = 2,
    }

    #[slab_item]
    #[derive(Clone, Copy, Debug, Default)]
    pub struct DataChange {
        pub ty: DataChangeTy,
        pub data_0: u32,
        pub data_1: u32,
        pub data_2: u32,
    }

    impl DataChange {
        pub fn apply(change: DataChange, data: Data) -> Data {
            let mut result: Data;
            #[wgsl_allow(non_literal_match_statement_patterns)]
            match change.ty {
                DataChangeTy::I => {
                    result = Data {
                        i: change.data_0,
                        float_val: data.float_val,
                        ints_0: data.ints_0,
                        ints_1: data.ints_1,
                    };
                }
                DataChangeTy::Float => {
                    result = Data {
                        i: data.i,
                        float_val: bitcast_f32(change.data_0),
                        ints_0: data.ints_0,
                        ints_1: data.ints_1,
                    };
                }
                _ => {
                    result = Data {
                        i: data.i,
                        float_val: data.float_val,
                        ints_0: change.data_0,
                        ints_1: change.data_1,
                    };
                }
            }
            result
        }
    }

    #[slab_item]
    #[derive(Clone, Copy, Debug, Default)]
    pub struct ArrayChange {
        pub i: u32,
        pub change: DataChange,
    }

    #[slab_item]
    #[derive(Clone, Copy, Debug, Default)]
    pub struct AnyChangeId {
        pub change_id: ArrayChangeId,
        pub data_array: DataArray,
    }

    #[slab_item]
    #[derive(Clone, Copy, Debug, Default)]
    pub struct ApplyDataChangeInvocation {
        pub changes_ids: AnyChangeIdArray,
    }

    // -- Storage bindings --

    storage!(group(0), binding(0), read_write, DATA_SLAB: RuntimeArray<u32>);
    storage!(group(0), binding(1), CHANGES_SLAB: RuntimeArray<u32>);
    storage!(group(0), binding(2), read_write, INVOCATIONS_RAN: Atomic<u32>);
    storage!(group(0), binding(3), read_write, INVOCATIONS_SKIPPED: Atomic<u32>);

    // -- Compute entry point --

    #[compute]
    #[workgroup_size(16, 1, 1)]
    pub fn main(#[builtin(global_invocation_id)] global_id: Vec3u) {
        let invocation = slab_read!(ApplyDataChangeInvocation, get!(CHANGES_SLAB), 0u32);

        let index = global_id.x;
        if index >= invocation.changes_ids.len {
            atomic_add(&get!(INVOCATIONS_SKIPPED), 1u32);
            return;
        }
        atomic_add(&get!(INVOCATIONS_RAN), 1u32);

        let change_id_id = AnyChangeIdArray::at(invocation.changes_ids, index);
        let any_change = slab_read!(AnyChangeId, get!(CHANGES_SLAB), change_id_id.inner);

        let array_change = slab_read!(ArrayChange, get!(CHANGES_SLAB), any_change.change_id.inner);

        let data_id = DataArray::at(any_change.data_array, array_change.i);
        let data = slab_read!(Data, get!(DATA_SLAB), data_id.inner);

        let result = DataChange::apply(array_change.change, data);
        slab_write!(Data, get_mut!(DATA_SLAB), data_id.inner, result);
    }
}

// ---------------------------------------------------------------------------
// CPU-only helpers — outside the #[wgsl] module.
// ---------------------------------------------------------------------------

use apply_data_changes::*;

const WORKGROUP_SIZE: u32 = 16;

impl DataChange {
    fn i(val: u32) -> Self {
        Self {
            ty: DataChangeTy::I,
            data_0: val,
            data_1: 0,
            data_2: 0,
        }
    }

    fn float(val: f32) -> Self {
        Self {
            ty: DataChangeTy::Float,
            data_0: val.to_bits(),
            data_1: 0,
            data_2: 0,
        }
    }

    fn ints(a: u32, b: u32) -> Self {
        Self {
            ty: DataChangeTy::Ints,
            data_0: a,
            data_1: b,
            data_2: 0,
        }
    }

    fn apply_mut(&self, data: &mut Data) {
        *data = Self::apply(*self, *data);
    }
}

impl core::fmt::Display for DataChange {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.ty {
            DataChangeTy::I => write!(f, "change i to {}", self.data_0),
            DataChangeTy::Float => {
                write!(f, "change float_val to {}", f32::from_bits(self.data_0))
            }
            DataChangeTy::Ints => {
                write!(f, "change ints to ({}, {})", self.data_0, self.data_1)
            }
        }
    }
}

impl From<DataChange> for ArrayChange {
    fn from(change: DataChange) -> Self {
        ArrayChange { i: 0, change }
    }
}

impl ArrayChange {
    fn apply(&self, data: &mut [Data]) {
        self.change.apply_mut(&mut data[self.i as usize]);
    }
}

impl core::fmt::Display for ArrayChange {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{} of data at index {}", self.change, self.i)
    }
}

impl ApplyDataChangeInvocation {
    fn total_invocations_required(&self) -> u32 {
        self.changes_ids.len
    }

    fn workgroup_dimensions(&self) -> (u32, u32, u32) {
        let count = self.total_invocations_required();
        let volume = WORKGROUP_SIZE;
        let groups = count.div_ceil(volume);
        (groups, 1, 1)
    }
}

// ---------------------------------------------------------------------------
// Tests that don't depend on wire types.
// ---------------------------------------------------------------------------

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
        // u32 roundtrip
        let arena = Arena::new(&crate::wgpu_runtime(), "test", None);
        let values = arena.new_array(0u32..=9);
        let _ = arena.commit();
        let from_gpu = futures_lite::future::block_on(arena.read_slab(values.array())).unwrap();
        let from_cpu = values.read_range(.., |ts| ts.to_vec());
        assert_eq!(from_cpu, from_gpu);
    }
    {
        // Data roundtrip
        let arena = Arena::new(&crate::wgpu_runtime(), "test", None);
        let values = arena.new_array([
            Data {
                i: 0,
                float_val: 0.0,
                ints_0: 0,
                ints_1: 0,
            },
            Data {
                i: 1,
                float_val: 1.0,
                ints_0: 1,
                ints_1: 1,
            },
            Data {
                i: 2,
                float_val: 2.0,
                ints_0: 2,
                ints_1: 2,
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
    log::info!("{:?}", [id0, id1, id2, id3]);

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
        let cpu_values = $id.get_vec();
        assert_eq!($initial_values, cpu_values, "cpu side wrong");
        log::debug!("  ...CPU values are all good :)");
        log::debug!("ensuring GPU values (right) match expected (left)");
        let gpu_values = futures_lite::future::block_on($slab.read_slab($id.array())).unwrap();
        assert_eq!($initial_values, gpu_values, "gpu side wrong");
        {
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
fn array_subslice_sanity() {
    let _ = env_logger::builder().is_test(true).try_init();

    log::info!("creating the slab");
    let slab = Arena::new(&CpuRuntime, "test", None);
    let mut initial_values = vec![
        Data {
            i: 0,
            float_val: 1.0,
            ints_0: 1,
            ints_1: 1,
        },
        Data {
            i: 1,
            float_val: 2.0,
            ints_0: 2,
            ints_1: 2,
        },
        Data {
            i: 2,
            float_val: 3.0,
            ints_0: 3,
            ints_1: 3,
        },
    ];

    log::info!("staging initial values");
    let values = slab.new_array(initial_values.clone());
    ensure!(slab, initial_values, values);

    log::info!("updating the initial values");
    initial_values[1].ints_0 = 666;
    initial_values[1].ints_1 = 666;
    initial_values[2].ints_0 = 420;
    initial_values[2].ints_1 = 420;
    values.modify_range(1u32..3, |items| {
        items[0].ints_0 = 666;
        items[0].ints_1 = 666;
        items[1].ints_0 = 420;
        items[1].ints_1 = 420;
    });
    log::debug!("updated_ranges: {:?}", values.updated_ranges());
    ensure!(slab, initial_values, values);

    log::info!("updating the initial values with overlapping updates");
    initial_values[0].float_val = 10.0;
    initial_values[1].float_val = 20.0;
    initial_values[2].float_val = 30.0;
    values.modify_range(0u32..3, |items| {
        items[0].float_val = 10.0;
        items[1].float_val = 20.0;
        items[2].float_val = 30.0;
    });

    initial_values[1].float_val = 666.0;
    values.modify_range(1u32..2, |items| {
        items[0].float_val = 666.0;
    });
    ensure!(slab, initial_values, values);

    initial_values[2].ints_0 = 32;
    initial_values[2].ints_1 = 32;
    values.modify_item(2, |data| {
        data.ints_0 = 32;
        data.ints_1 = 32;
    });
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

// ---------------------------------------------------------------------------
// GPU update test infrastructure.
// ---------------------------------------------------------------------------

fn arb_data() -> impl Strategy<Value = Data> {
    (
        proptest::num::u32::ANY,
        proptest::num::f32::NORMAL,
        proptest::num::u32::ANY,
        proptest::num::u32::ANY,
    )
        .prop_map(|(i, float_val, ints_0, ints_1)| Data {
            i,
            float_val,
            ints_0,
            ints_1,
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
    /// Apply one step's worth of changes to the backend.
    ///
    /// Returns (invocations_ran, invocations_skipped).
    fn apply_backend_changes(&mut self) -> (u32, u32);
}

impl BackendUpdate for GpuUpdateTest<CpuRuntime, ()> {
    fn apply_backend_changes(&mut self) -> (u32, u32) {
        let slab_buffer = self.arena.get_buffer().unwrap();
        let data_slab_vec = slab_buffer.as_vec().clone();
        let changes_buffer = self.changes_arena.get_buffer().unwrap();
        let changes_slab_vec = changes_buffer.as_vec().clone();

        // Set up Storage statics for CPU dispatch.
        apply_data_changes::DATA_SLAB.set(RuntimeArray {
            data: data_slab_vec,
        });
        apply_data_changes::CHANGES_SLAB.set(RuntimeArray {
            data: changes_slab_vec,
        });
        apply_data_changes::INVOCATIONS_RAN.set(Atomic::default());
        apply_data_changes::INVOCATIONS_SKIPPED.set(Atomic::default());

        // Dispatch compute shader on CPU.
        let invocation = self.invocation.get();
        let (wg_x, wg_y, wg_z) = invocation.workgroup_dimensions();
        log::debug!("  invocation workgroup dimensions: ({wg_x}, {wg_y}, {wg_z})");
        for i in 0..wg_x * WORKGROUP_SIZE {
            for j in 0..wg_y {
                for k in 0..wg_z {
                    apply_data_changes::main(wgsl_rs::std::vec3u(i, j, k));
                }
            }
        }

        // Read counters.
        let ran = atomic_load(&apply_data_changes::INVOCATIONS_RAN.get());
        let skipped = atomic_load(&apply_data_changes::INVOCATIONS_SKIPPED.get());

        // Copy results back to the buffer.
        let result_data = apply_data_changes::DATA_SLAB.get();
        let mut data_slab_mut = slab_buffer.as_mut_vec();
        data_slab_mut.deref_mut()[..result_data.data.len()].copy_from_slice(&result_data.data);

        (ran, skipped)
    }
}

#[derive(Debug)]
enum BackendChange {
    Single {
        id: u32,
        change: DataChange,
    },
    Array {
        array: (u32, u32),
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

    #[allow(dead_code)]
    backend_updater: T,
}

impl<R: IsRuntime, T> GpuUpdateTest<R, T> {
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
                        change.apply_mut(raw);
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
                        "{reason} - unexpected single value for entry {i} out of {total} ({:?}) \
                         after applying {change_made} to previous value:\n{previous_raw_data:#?}",
                        arena.slab_range(),
                        reason = if unchanged_since_previous {
                            "Value is unchanged and probably should be"
                        } else {
                            "Value changed incorrectly"
                        }
                    );
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
                        "{reason} - unexpected array value for entry {i} out of {total} ({:?}) \
                         after applying {change_made} to previous value:\n{previous_raw_data:#?}",
                        arena.slab_range(),
                        reason = if unchanged_since_previous {
                            "Value is unchanged and probably should be"
                        } else {
                            "Value changed incorrectly"
                        }
                    );
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
        let invocation = changes_arena.new_value(ApplyDataChangeInvocation {
            changes_ids: AnyChangeIdArray {
                id: AnyChangeIdId::NONE,
                len: 0,
            },
        });
        let changes_all_change_ids = changes_arena.new_array(Vec::<AnyChangeId>::new());
        let test = GpuUpdateTest {
            arena,
            previous_values: all_values.clone(),
            current_values: all_values,
            backend_updater,

            changes_arena,
            changes_values: vec![],
            changes_all_change_ids,

            invocation,
        };
        test.verify();
        test
    }

    fn update_changes_on_backend(&mut self, changes: Vec<BackendChange>) {
        log::info!("  updating to changeset {changes:#?}");
        let _ = self.arena.commit();

        let mut change_values = vec![];
        let mut any_changes = vec![];
        for change in changes.into_iter() {
            let (data_array, change) = match change {
                BackendChange::Single { id, change } => {
                    let array_change = ArrayChange::from(change);
                    let data_array = DataArray {
                        id: DataId::new(id),
                        len: 1,
                    };
                    (data_array, array_change)
                }
                BackendChange::Array { array, change } => {
                    let data_array = DataArray {
                        id: DataId::new(array.0),
                        len: array.1,
                    };
                    (data_array, change)
                }
            };
            let change_value = self.changes_arena.new_value(change);
            any_changes.push(AnyChangeId {
                change_id: ArrayChangeId::new(change_value.id()),
                data_array,
            });
            change_values.push(change_value);
        }

        let any_change_array = self.changes_arena.new_array(any_changes);

        self.invocation.modify(|inv| {
            let (idx, len) = any_change_array.array();
            inv.changes_ids = AnyChangeIdArray {
                id: AnyChangeIdId::new(idx),
                len,
            };
        });

        self.changes_values = change_values;
        self.changes_all_change_ids = any_change_array;

        log::info!("  commiting changes");
        let _ = self.changes_arena.commit();
        log::info!("  done commiting changes");
    }

    fn run(mut self, verify_each_step: bool)
    where
        Self: BackendUpdate,
    {
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
            log::info!("  applying changes to raw data");
            let backend_changes = self.apply_raw_changes();
            log::info!("  done applying changes to raw data");
            if backend_changes.is_empty() {
                if verify_each_step {
                    log::trace!("  ({steps}) applied all changes");
                }
                break;
            }

            self.update_changes_on_backend(backend_changes);
            let invocation = self.invocation.get();

            log::info!("  applying backend changes");
            let (invocations_ran, _invocations_skipped) = self.apply_backend_changes();
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

// ---------------------------------------------------------------------------
// GPU update tests.
// ---------------------------------------------------------------------------

#[test]
fn gpu_update_test_sanity_on_cpu() {
    let _ = env_logger::builder().is_test(true).try_init();
    let arena = Arena::new(&CpuRuntime, "test", None);
    let all_values = vec![ValueData::Single(
        Data {
            i: 0,
            float_val: 0.0,
            ints_0: 0,
            ints_1: 0,
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
fn gpu_array_update_test_sanity_on_cpu() {
    let _ = env_logger::builder().is_test(true).try_init();
    let arena = Arena::new(&CpuRuntime, "test", None);
    let all_values = vec![ValueData::Array(
        vec![Data {
            i: 1683186,
            float_val: 2.1727349e24,
            ints_0: 348221601,
            ints_1: 1304208859,
        }],
        vec![ArrayChange {
            i: 0,
            change: DataChange {
                ty: DataChangeTy::Ints,
                data_0: 3211909787,
                data_1: 1326905872,
                data_2: 0,
            },
        }],
    )];
    let test = GpuUpdateTest::new(arena, (), &all_values);
    test.run(true);
}

fn one_datum() -> ValueData {
    ValueData::Array(
        vec![Data {
            i: 0,
            float_val: 0.0,
            ints_0: 0,
            ints_1: 0,
        }],
        vec![ArrayChange {
            i: 0,
            change: DataChange {
                ty: DataChangeTy::I,
                data_0: 666,
                data_1: 0,
                data_2: 0,
            },
        }],
    )
}

#[test]
fn workgroup_dimensions_to_id_sanity() {
    let mut indices = vec![];
    for x in 0..WORKGROUP_SIZE {
        indices.push(x);
    }
    pretty_assertions::assert_eq!((0..WORKGROUP_SIZE).collect::<Vec<_>>(), indices);
}

#[test]
fn invocations_sanity() {
    let changes = vec![one_datum()];
    let arena = Arena::new(&CpuRuntime, "test", None);
    let mut test = GpuUpdateTest::new(arena, (), &changes);
    test.arena.commit();
    let changes = test.apply_raw_changes();
    test.update_changes_on_backend(changes);
    let (invocations_ran, invocations_skipped) = test.apply_backend_changes();
    let total_invocations = invocations_ran + invocations_skipped;

    assert_eq!(
        (WORKGROUP_SIZE, 1, WORKGROUP_SIZE - 1),
        (total_invocations, invocations_ran, invocations_skipped),
    );
}

#[test]
fn regression() {
    let data = vec![one_datum()];

    log::info!("START CPU TEST");
    let arena = Arena::new(&CpuRuntime, "test", None);
    let test = GpuUpdateTest::new(arena, (), &data);
    test.run(true);
}

proptest! {
    #[test]
    fn proptest_gpu_updates_checked_on_cpu(value_data in proptest::collection::vec(arb_value_data(32, 32), 1..32)) {
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

// ---------------------------------------------------------------------------
// WGSL inspection
// ---------------------------------------------------------------------------

#[test]
fn wgsl_source_contains_companion_types() {
    let source = apply_data_changes::WGSL_MODULE.wgsl_source();
    let wgsl = source.join("\n");

    // Verify companion types are present in the WGSL output.
    assert!(
        wgsl.contains("struct DataId"),
        "WGSL should contain 'struct DataId'"
    );
    assert!(
        wgsl.contains("struct DataArray"),
        "WGSL should contain 'struct DataArray'"
    );
    assert!(
        wgsl.contains("struct DataChangeTyId"),
        "WGSL should contain 'struct DataChangeTyId'"
    );
    assert!(
        wgsl.contains("struct DataChange"),
        "WGSL should contain 'struct DataChange'"
    );
    assert!(
        wgsl.contains("@compute"),
        "WGSL should contain a compute entry point"
    );
}

#[test]
fn wgsl_source_validates_with_naga() {
    apply_data_changes::WGSL_MODULE
        .validate()
        .expect("WGSL should validate with naga");
}

// ---------------------------------------------------------------------------
// GPU (wgpu) backend — TestBackendWgpu + BackendUpdate impl
// ---------------------------------------------------------------------------

struct TestBackendWgpu {
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    invocations_ran_buffer: wgpu::Buffer,
    invocations_skipped_buffer: wgpu::Buffer,
}

impl TestBackendWgpu {
    fn new(runtime: &crate::runtime::WgpuRuntime) -> Self {
        use apply_data_changes::linkage;

        let device = &runtime.device;
        let module = linkage::shader_module(device);
        let bind_group_layout = linkage::bind_group_0::layout(device);
        let pipeline_layout = linkage::main::pipeline_layout(device, &[&bind_group_layout]);
        let pipeline = linkage::main::compute_pipeline(device, Some(&pipeline_layout), &module);

        let counter_desc = |label| wgpu::BufferDescriptor {
            label: Some(label),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        };
        let invocations_ran_buffer = device.create_buffer(&counter_desc("invocations_ran"));
        let invocations_skipped_buffer = device.create_buffer(&counter_desc("invocations_skipped"));

        Self {
            bind_group_layout,
            pipeline,
            invocations_ran_buffer,
            invocations_skipped_buffer,
        }
    }
}

impl BackendUpdate for GpuUpdateTest<crate::runtime::WgpuRuntime, TestBackendWgpu> {
    fn apply_backend_changes(&mut self) -> (u32, u32) {
        use apply_data_changes::linkage;

        let runtime = self.arena.runtime();
        let device = &runtime.device;
        let queue = &runtime.queue;

        // Zero the counter buffers.
        queue.write_buffer(&self.backend_updater.invocations_ran_buffer, 0, &[0; 4]);
        queue.write_buffer(&self.backend_updater.invocations_skipped_buffer, 0, &[0; 4]);

        // Get the arena buffers (commit has already been called).
        let data_buffer = self.arena.get_buffer().expect("data arena has no buffer");
        let changes_buffer = self
            .changes_arena
            .get_buffer()
            .expect("changes arena has no buffer");

        // Create bind group.
        let bind_group = linkage::bind_group_0::create(
            device,
            &self.backend_updater.bind_group_layout,
            data_buffer.as_entire_binding(),
            changes_buffer.as_entire_binding(),
            self.backend_updater
                .invocations_ran_buffer
                .as_entire_binding(),
            self.backend_updater
                .invocations_skipped_buffer
                .as_entire_binding(),
        );

        // Dispatch compute shader.
        let invocation = self.invocation.get();
        let (wg_x, wg_y, wg_z) = invocation.workgroup_dimensions();
        log::debug!("  GPU dispatch workgroup dimensions: ({wg_x}, {wg_y}, {wg_z})");

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("apply_data_changes"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.backend_updater.pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            pass.dispatch_workgroups(wg_x, wg_y, wg_z);
        }

        // Create staging buffers to read back counter values.
        let staging_desc = |label| wgpu::BufferDescriptor {
            label: Some(label),
            size: 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        };
        let ran_staging = device.create_buffer(&staging_desc("ran_staging"));
        let skipped_staging = device.create_buffer(&staging_desc("skipped_staging"));

        encoder.copy_buffer_to_buffer(
            &self.backend_updater.invocations_ran_buffer,
            0,
            &ran_staging,
            0,
            4,
        );
        encoder.copy_buffer_to_buffer(
            &self.backend_updater.invocations_skipped_buffer,
            0,
            &skipped_staging,
            0,
            4,
        );

        let submission_index = queue.submit(std::iter::once(encoder.finish()));

        // Map and read back the counters.
        let read_u32 = |buffer: &wgpu::Buffer| -> u32 {
            let slice = buffer.slice(..);
            let (tx, rx) = async_channel::bounded(1);
            slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send_blocking(result);
            });
            device
                .poll(wgpu::PollType::Wait {
                    submission_index: Some(submission_index.clone()),
                    timeout: None,
                })
                .expect("device poll");
            futures_lite::future::block_on(rx.recv())
                .expect("channel recv")
                .expect("map_async");
            let data = slice.get_mapped_range();
            let value = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
            drop(data);
            buffer.unmap();
            value
        };

        let ran = read_u32(&ran_staging);
        let skipped = read_u32(&skipped_staging);

        (ran, skipped)
    }
}

// ---------------------------------------------------------------------------
// GPU (wgpu) tests
// ---------------------------------------------------------------------------

#[test]
fn gpu_update_test_sanity_on_gpu() {
    let _ = env_logger::builder().is_test(true).try_init();
    let runtime = crate::wgpu_runtime();
    let backend = TestBackendWgpu::new(&runtime);
    let arena = Arena::new(&runtime, "test-gpu", None);
    let all_values = vec![ValueData::Single(
        Data {
            i: 0,
            float_val: 0.0,
            ints_0: 0,
            ints_1: 0,
        },
        vec![
            DataChange::i(1),
            DataChange::float(1.0),
            DataChange::ints(1, 1),
        ],
    )];
    let test = GpuUpdateTest::new(arena, backend, &all_values);
    test.run(true);
}

#[test]
fn gpu_array_update_test_sanity_on_gpu() {
    let _ = env_logger::builder().is_test(true).try_init();
    let runtime = crate::wgpu_runtime();
    let backend = TestBackendWgpu::new(&runtime);
    let arena = Arena::new(&runtime, "test-gpu", None);
    let all_values = vec![ValueData::Array(
        vec![Data {
            i: 1683186,
            float_val: 2.1727349e24,
            ints_0: 348221601,
            ints_1: 1304208859,
        }],
        vec![ArrayChange {
            i: 0,
            change: DataChange {
                ty: DataChangeTy::Ints,
                data_0: 3211909787,
                data_1: 1326905872,
                data_2: 0,
            },
        }],
    )];
    let test = GpuUpdateTest::new(arena, backend, &all_values);
    test.run(true);
}

proptest! {
    #[test]
    fn proptest_gpu_updates_checked_on_gpu(value_data in proptest::collection::vec(arb_value_data(8, 8), 1..8)) {
        let _ = env_logger::builder().is_test(true).try_init();
        let runtime = crate::wgpu_runtime();
        let backend = TestBackendWgpu::new(&runtime);
        let arena = Arena::new(&runtime, "test-gpu", None);
        let test = GpuUpdateTest::new(arena, backend, &value_data);
        test.run(true);
    }
}
