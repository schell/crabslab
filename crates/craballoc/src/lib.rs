//! Crafting a tasty slab.
//!
//! This crate provides [`Arena`] allocation backed by a `u32` slab.
#![doc = include_str!("../README.md")]

use snafu::prelude::*;

pub mod arena;
mod buffer;
pub mod range;
pub mod runtime;
// pub mod slab;
mod update;
// pub mod value;

pub mod prelude {
    //! Easy-include prelude module.
    pub extern crate crabslab;
    pub use super::arena::*;
    pub use super::runtime::*;
    // pub use super::slab::*;
    // pub use super::value::*;
}

#[cfg(doc)]
use prelude::crabslab::SlabItem;
#[cfg(doc)]
use prelude::*;

#[derive(Debug, Snafu)]
#[snafu(visibility(pub(crate)))]
pub enum Error {
    #[snafu(display(
        "Slab has no internal buffer. Please call SlabAllocator::commit or \
         SlabAllocator::get_updated_buffer first."
    ))]
    NoInternalBuffer,

    #[snafu(display("Async recv error: {source}"))]
    AsyncRecv { source: async_channel::RecvError },

    #[cfg(feature = "wgpu")]
    #[snafu(display("Async error: {source}"))]
    Async { source: wgpu::BufferAsyncError },

    #[cfg(feature = "wgpu")]
    #[snafu(display("Poll error: {source}"))]
    Poll { source: wgpu::PollError },

    #[snafu(display("{source}"))]
    Other { source: Box<dyn std::error::Error> },
}

#[cfg(all(test, feature = "wgpu"))]
fn wgpu_runtime() -> crate::runtime::WgpuRuntime {
    let backends = wgpu::Backends::PRIMARY;
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends,
        ..Default::default()
    });
    let adapter =
        futures_lite::future::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .unwrap();
    let (device, queue) =
        futures_lite::future::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
            .unwrap();
    crate::runtime::WgpuRuntime {
        device: device.into(),
        queue: queue.into(),
    }
}

#[cfg(test)]
mod test {
    use crabslab::SlabItem;
    use proptest::{
        prelude::{Just, Strategy},
        prop_compose, proptest,
    };

    use crate::{arena::Arena, range::Range, runtime::CpuRuntime};

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

        #[derive(Clone, Copy, Debug, Default, PartialEq, SlabItem)]
        struct Data {
            i: u32,
            float: f32,
            int: i64,
        }

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
}
