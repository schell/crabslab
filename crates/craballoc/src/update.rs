//! An abstraction over a lazy updating mechanism.
//!
//! This abstraction allows us to send updates to the GPU lazily,
//! by updating a slab's subslice range that corresponds to an
//! allocated `T` _locally_, and then reading those values
//! once during a commit phase, guided by update notifications.

use std::{
    hash::Hash,
    sync::{Arc, RwLock, Weak},
};

use rustc_hash::{FxHashMap, FxHashSet};

use crate::range::{Range, RangeManager};

/// The unique source range of a slab update.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct SourceId {
    pub range: Range,
    /// This field is just for debugging.
    pub type_is: &'static str,
}

impl Hash for SourceId {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.range.hash(state);
    }
}

impl core::fmt::Display for SourceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("{}({:?})", self.type_is, self.range))
    }
}

/// Held internally by the `CpuUpdateManager` and checked for updated data
/// when it receives a `SourceId` during a commit phase.
#[derive(Clone)]
#[repr(transparent)]
pub struct CpuData {
    pub data: Weak<RwLock<Vec<u32>>>,
}

/// Held externally by allocated values.
///
/// This struct is used to communicate updates from userland values that
/// have been allocated, to the inner update machinery that ultimately sends
/// this data to the GPU.
#[derive(Clone)]
pub struct CpuUpdateSource {
    source: SourceId,
    sender: async_channel::Sender<SourceId>,
    data: Arc<RwLock<Vec<u32>>>,
}

impl CpuUpdateSource {
    pub fn cpu_data(&self) -> CpuData {
        CpuData {
            data: Arc::downgrade(&self.data),
        }
    }

    pub fn read<T>(&self, f: impl FnOnce(&[u32]) -> T) -> T {
        let guard = self.data.read().unwrap();
        f(&guard)
    }

    pub fn modify<T>(&self, f: impl FnOnce(&mut [u32]) -> T) -> T {
        let mut guard = self.data.write().unwrap();
        let t = f(&mut guard);
        // UNWRAP: safe because this channel is unbounded
        self.sender.try_send(self.source).unwrap();
        t
    }
}

#[derive(Clone)]
pub struct CpuUpdate {
    pub range: Range,
    pub data: Vec<u32>,
}

pub struct CpuUpdateSummary {
    pub update_ranges: RangeManager<CpuUpdate>,
    pub recycle_ranges: RangeManager<SourceId>,
}

/// An abstraction over updates made from the CPU.
#[derive(Clone)]
pub struct CpuUpdateManager {
    /// Sends notification of an update at the time of update.
    notifier_sender: async_channel::Sender<SourceId>,
    /// Receives update notifications during a commit.
    notifier_receiver: async_channel::Receiver<SourceId>,

    // Weak references to all values that can write updates into a slab
    update_sources: Arc<RwLock<FxHashMap<SourceId, CpuData>>>,
    // Set of ids of the update sources that have updates queued
    update_queue: Arc<RwLock<FxHashSet<SourceId>>>,
}

impl Default for CpuUpdateManager {
    fn default() -> Self {
        let (tx, rx) = async_channel::unbounded();
        Self {
            notifier_sender: tx,
            notifier_receiver: rx,
            update_sources: Default::default(),
            update_queue: Default::default(),
        }
    }
}

impl CpuUpdateManager {
    /// Create a new update source.
    pub fn new_update_source(&self, source: SourceId) -> CpuUpdateSource {
        let update_source = CpuUpdateSource {
            source,
            sender: self.notifier_sender.clone(),
            data: Arc::new(RwLock::new(vec![0u32; source.range.len() as usize])),
        };
        let cpu_data = update_source.cpu_data();
        self.update_sources
            .write()
            .unwrap()
            .insert(source, cpu_data);
        update_source
    }

    /// Return the ids of all sources that require updating.
    pub fn clear_updated_source_ids(&self) -> FxHashSet<SourceId> {
        // UNWRAP: panic on purpose
        let mut update_set = self.update_queue.write().unwrap();
        while let Ok(source_id) = self.notifier_receiver.try_recv() {
            update_set.insert(source_id);
        }
        update_set.clone()
    }

    /// Clear the updates in the queue and convert them into a managed set of ranges.
    ///
    /// This ensures that an entire frame of updates is coalesced into the smallest
    /// number of buffer writes as is possible.
    pub fn clear_updated_sources(&self) -> CpuUpdateSummary {
        let mut update_ranges = RangeManager::<CpuUpdate>::default();
        let mut recycle_ranges = RangeManager::<SourceId>::default();

        let update_source_ids = self.clear_updated_source_ids();
        // UNWRAP: panic on purpose
        *self.update_queue.write().unwrap() = Default::default();
        // Prepare all of our GPU buffer writes
        {
            // Recycle any update sources that are no longer needed, and collect the active
            // sources' updates into `writes`.
            let mut updates_guard = self.update_sources.write().unwrap();
            for id in update_source_ids {
                if let Some(cpu_data) = updates_guard.get_mut(&id) {
                    if let Some(data) = cpu_data.data.upgrade() {
                        // Userland still holds a reference to this, schedule the update
                        let update_data = CpuUpdate {
                            data: data.read().unwrap().clone(),
                            range: id.range,
                        };
                        log::trace!("updating {id}");
                        update_ranges.add_range(update_data);
                    } else {
                        // Recycle this allocation, it has been dropped from userland
                        log::debug!("recycling {id}");
                        recycle_ranges.add_range(id);
                        updates_guard.remove(&id);
                    }
                } else {
                    log::error!("Could not find {id}");
                }
            }
        }

        CpuUpdateSummary {
            update_ranges: update_ranges.defrag(),
            recycle_ranges: recycle_ranges.defrag(),
        }
    }
}
