//! Crafting a tasty slab.
#![doc = include_str!("../README.md")]

pub mod range;
pub mod runtime;
pub mod slab;
pub mod value;

pub mod prelude {
    //! Easy-include prelude module.
    pub extern crate crabslab;
    pub use super::runtime::*;
    pub use super::slab::*;
    pub use super::value::*;
}

#[cfg(doc)]
use prelude::crabslab::SlabItem;
#[cfg(doc)]
use prelude::*;

#[cfg(test)]
mod test {
    use std::sync::atomic::Ordering;

    pub use crabslab::Slab;

    use crate::{range::Range, runtime::CpuRuntime, slab::SlabAllocator};

    #[test]
    fn mngr_updates_count_sanity() {
        let slab = SlabAllocator::new(CpuRuntime, ());
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
            slab.update_sources.read().unwrap().len(),
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
            buffer.is_invalid(),
            "buffer capacity change should have invalidated the old buffer"
        );
        assert!(new_buffer.is_valid());
        assert_eq!(
            0,
            slab.update_sources.read().unwrap().len(),
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
        let m = SlabAllocator::new(CpuRuntime, ());
        log::info!("allocating 4 unused u32 slots");
        let _ = m.allocate::<u32>();
        let _ = m.allocate::<u32>();
        let _ = m.allocate::<u32>();
        let _ = m.allocate::<u32>();

        log::info!("creating 4 update sources");
        let h4 = m.new_value(0u32);
        let h5 = m.new_value(0u32);
        let h6 = m.new_value(0u32);
        let h7 = m.new_value(0u32);
        log::info!("running commit");
        let buffer = m.commit();
        assert!(
            buffer.is_new_this_commit(),
            "invocation {} != invalidation {} != creation {}",
            buffer.invocation_k(),
            buffer.invalidation_k(),
            buffer.creation_time()
        );
        assert!(m.recycles.read().unwrap().ranges.is_empty());
        assert_eq!(4, m.update_sources.read().unwrap().len());
        let k = m.update_k.load(Ordering::Relaxed);
        assert_eq!(4, k);

        log::info!("dropping 4 update sources");
        drop(h4);
        drop(h5);
        drop(h6);
        drop(h7);
        let _ = m.commit();
        assert!(
            !buffer.is_new_this_commit(),
            "buffer was created last commit"
        );
        assert!(buffer.is_valid(), "decreasing capacity never happens");
        assert_eq!(1, m.recycles.read().unwrap().ranges.len());
        assert!(m.update_sources.read().unwrap().is_empty());

        log::info!("creating 4 update sources, round two");
        let h4 = m.new_value(0u32);
        let h5 = m.new_value(0u32);
        let h6 = m.new_value(0u32);
        let h7 = m.new_value(0u32);
        assert!(m.recycles.read().unwrap().ranges.is_empty());
        assert_eq!(4, m.update_sources.read().unwrap().len());
        let k = m.update_k.load(Ordering::Relaxed);
        // MAYBE_TODO: recycle "update_k"s instead of incrementing for each new source
        assert_eq!(8, k);

        log::info!("creating one more update source, immediately dropping it and two others");
        let h8 = m.new_value(0u32);
        drop(h8);
        drop(h4);
        drop(h6);
        let _ = m.commit();
        assert_eq!(3, m.recycles.read().unwrap().ranges.len());
        assert_eq!(2, m.update_sources.read().unwrap().len());
        assert_eq!(9, m.update_k.load(Ordering::Relaxed));

        drop(h7);
        drop(h5);
        let _ = m.commit();
        m.defrag();
        assert_eq!(
            1,
            m.recycles.read().unwrap().ranges.len(),
            "ranges: {:#?}",
            m.recycles.read().unwrap().ranges
        );
    }

    #[test]
    fn hybrid_write_guard() {
        let m = SlabAllocator::new(CpuRuntime, ());
        let h4 = m.new_value(0u32);
        assert!(
            m.get_buffer().is_none(),
            "buffer should not be created until after first `commit`"
        );
        m.commit();

        {
            let mut guard = h4.lock();
            assert_eq!(0, *guard);
            assert!(!guard.mutated, "guard should not mutate with deref");

            *guard += 10;
            assert_eq!(10, *guard);
            assert!(guard.mutated, "guard should mutate with deref mut");
            let buffer = m.get_buffer().unwrap();
            let v = buffer.as_vec();
            assert_eq!(
                &[0],
                v.as_slice(),
                "underlying buffer should not be written until after `commit`",
            );
        }
        m.commit();

        let buffer = m.get_buffer().unwrap();
        assert_eq!(&[10], buffer.as_vec().as_slice());
    }
}
