//! An abstraction over a lazy updating mechanism.
//!
//! This abstraction allows us to send updates to the GPU lazily,
//! by updating a slab's subslice range that corresponds to an
//! allocated `T` _locally_, and then reading those values
//! once during a commit phase, guided by update notifications.

use std::{
    collections::HashSet,
    hash::Hash,
    ops::Deref,
    sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard, Weak},
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

#[derive(Default)]
pub struct SynchronizationData {
    /// The ranges of updates in `data`.
    ///
    /// These ranges point to `data`, not the slab.
    cpu_updated: RangeManager<Range>,
    gpu_update_requested: bool,
    data: Vec<u32>,
}

impl Deref for SynchronizationData {
    type Target = [u32];

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl SynchronizationData {
    pub fn new(len: usize) -> Self {
        Self {
            cpu_updated: Default::default(),
            gpu_update_requested: false,
            data: vec![0u32; len],
        }
    }

    /// Replace the inner data, triggering an update if the data has changed.
    ///
    /// ## Panics
    /// Panics on debug if `data` is a different length than the existing internal
    /// data.
    pub fn replace(&mut self, data: Vec<u32>) {
        debug_assert_eq!(
            self.data.len(),
            data.len(),
            "Existing data length doesn't match replaced data length: {} != {}",
            self.data.len(),
            data.len(),
        );

        if data != self.data {
            self.data = data;
            self.cpu_updated.insert(Range {
                first_index: 0,
                last_index: self.data.len() as u32 - 1,
            });
        }
    }
}

/// Data shared between `UpdateManager` and userland code.
///
/// Held internally by the `UpdateManager` and checked for updated data
/// when `UpdateManager` receives a `SourceId` during a commit phase.
#[derive(Clone)]
#[repr(transparent)]
pub struct CpuData {
    pub synchronization: Weak<RwLock<SynchronizationData>>,
}

/// Held externally by allocated values.
///
/// This struct is used to communicate updates from userland values that
/// have been allocated, to the inner update machinery that ultimately sends
/// this data to the GPU.
///
/// It also provides a mechanism to request that an update from the GPU to this
/// value be performed during commit.
#[derive(Clone)]
pub struct CpuUpdateSource {
    source: SourceId,
    sender: async_channel::Sender<SourceId>,
    data: Arc<RwLock<SynchronizationData>>,
}

impl Drop for CpuUpdateSource {
    fn drop(&mut self) {
        self.mark_updated();
    }
}

impl CpuUpdateSource {
    fn mark_updated(&self) {
        // UNWRAP: safe because this channel is unbounded
        self.sender.try_send(self.source).unwrap();
    }

    pub(crate) fn ref_count(&self) -> usize {
        Arc::strong_count(&self.data)
    }

    pub fn read_lock(&self) -> RwLockReadGuard<'_, SynchronizationData> {
        // UNWRAP: panic on purpoose
        self.data.read().unwrap()
    }

    pub fn write_lock(&self) -> RwLockWriteGuard<'_, SynchronizationData> {
        // UNWRAP: panic on purpoose
        self.data.write().unwrap()
    }

    pub fn cpu_data(&self) -> CpuData {
        CpuData {
            synchronization: Arc::downgrade(&self.data),
        }
    }

    pub fn source_id(&self) -> SourceId {
        self.source
    }

    /// Read the inner data, if any, returning `T`.
    ///
    /// In the case this `CpuUpdateSource` represents a [`OneWayFromGpu`](crate::arena::OneWayFromGpu)
    /// value, the slice provided to the given closure will be empty.
    pub fn read<T>(&self, f: impl FnOnce(&[u32]) -> T) -> T {
        let guard = self.data.read().unwrap();
        f(&guard.data)
    }

    /// Modify the inner data, if any, returning `T`.
    ///
    /// In the case this `CpuUpdateSource` represents a [`OneWayFromGpu`](crate::arena::OneWayFromGpu)
    /// value, the slice provided to the given closure will be empty.
    ///
    /// This marks the data as updated, which will cause the data to be sent to the GPU
    /// on the next commit.
    pub fn modify<T>(&self, f: impl FnOnce(&mut [u32]) -> T) -> T {
        let mut guard = self.data.write().unwrap();
        let t = f(&mut guard.data);
        guard.cpu_updated.insert(Range {
            first_index: 0,
            last_index: self.source.range.len() - 1,
        });
        self.mark_updated();
        t
    }

    /// Modify the inner data, if any, returning `T`.
    ///
    /// In the case this `CpuUpdateSource` represents a [`OneWayFromGpu`](crate::arena::OneWayFromGpu)
    /// value, the slice provided to the given closure will be empty.
    ///
    /// This marks the data as updated, which will cause the data to be sent to the GPU
    /// on the next commit.
    pub fn modify_range<T>(&self, range: Range, f: impl FnOnce(&mut [u32]) -> T) -> T {
        let mut guard = self.data.write().unwrap();

        let t = f(&mut guard.data[range.first_index as usize..=range.last_index as usize]);
        guard.cpu_updated.insert(Range {
            first_index: self.source.range.first_index + range.first_index,
            last_index: self.source.range.first_index + range.last_index,
        });
        self.mark_updated();
        t
    }

    /// Request that the next commit synchronize this source with the data from the GPU.
    pub fn request_gpu_sync(&self) {
        self.data.write().unwrap().gpu_update_requested = true;
        self.mark_updated();
    }

    /// A null update source that performs no updates.
    ///
    /// Updates made to this source will fall into the void.
    pub fn null() -> Self {
        CpuUpdateSource {
            source: SourceId {
                range: Range {
                    first_index: u32::MAX,
                    last_index: u32::MAX,
                },
                type_is: "_",
            },
            sender: async_channel::bounded(1).0,
            data: Default::default(),
        }
    }

    pub fn updated_ranges(&self) -> Vec<Range> {
        self.data.read().unwrap().cpu_updated.ranges.clone()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Update {
    pub range: Range,
    pub data: Vec<u32>,
}

pub struct UpdateSummary {
    pub cpu_update_ranges: RangeManager<Update>,
    pub gpu_update_ranges: RangeManager<Range>,
    pub recycle_ranges: RangeManager<SourceId>,
}

/// Manages updates to and from the CPU.
#[derive(Clone)]
pub struct UpdateManager {
    /// Sends notification of an update at the time of update.
    notifier_sender: async_channel::Sender<SourceId>,
    /// Receives update notifications during a commit.
    notifier_receiver: async_channel::Receiver<SourceId>,

    // Weak references to all values that can write updates into a slab
    update_sources: Arc<RwLock<FxHashMap<SourceId, CpuData>>>,
    // Set of ids of the update sources that have updates queued
    update_queue: Arc<RwLock<FxHashSet<SourceId>>>,
}

impl std::fmt::Debug for UpdateManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CpuUpdateManager")
            .field(
                "update_sources_count",
                &self.update_sources.read().unwrap().len(),
            )
            .field(
                "update_queue_count",
                &self.update_queue.read().unwrap().len(),
            )
            .finish()
    }
}

impl Default for UpdateManager {
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

impl UpdateManager {
    /// Create a new update source.
    pub fn new_update_source(&self, source: SourceId) -> CpuUpdateSource {
        let update_source = CpuUpdateSource {
            source,
            sender: self.notifier_sender.clone(),
            data: Arc::new(RwLock::new(SynchronizationData::new(
                source.range.len() as usize
            ))),
        };
        let cpu_data = update_source.cpu_data();
        self.update_sources
            .write()
            .unwrap()
            .insert(source, cpu_data);
        update_source
    }

    /// Returns all the `SourceId`s managed by this `UpdateManager`.
    pub fn get_managed_source_ids(&self) -> FxHashSet<SourceId> {
        FxHashSet::from_iter(self.update_sources.read().unwrap().keys().copied())
    }

    /// Returns whether any update sources have queued updates waiting to be committed.
    pub fn has_queued_updates(&self) -> bool {
        !self.notifier_receiver.is_empty() || !self.update_queue.read().unwrap().is_empty()
    }

    /// Return the ids of all sources that require updating.
    ///
    /// This clears the update `SourceId`s from their sources and stores
    /// them for use during the next commit, returning a clone of all updated
    /// sources since  
    pub fn get_updated_source_ids(&self) -> FxHashSet<SourceId> {
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
    pub fn clear_updated_sources(&self) -> UpdateSummary {
        let mut cpu_update_ranges = RangeManager::<Update>::default();
        let mut gpu_update_ranges = RangeManager::<Range>::default();
        let mut recycle_ranges = RangeManager::<SourceId>::default();

        let update_source_ids = self.get_updated_source_ids();
        // UNWRAP: panic on purpose
        *self.update_queue.write().unwrap() = Default::default();
        // Prepare all of our GPU buffer writes
        {
            // Recycle any update sources that are no longer needed, and collect the active
            // sources' updates into `writes`.
            let mut updates_guard = self.update_sources.write().unwrap();
            for id in update_source_ids {
                if let Some(cpu_data) = updates_guard.get_mut(&id) {
                    if let Some(sync_data) = cpu_data.synchronization.upgrade() {
                        // Userland still holds a reference to this, schedule the update
                        let mut sync_data_guard = sync_data.write().unwrap();
                        let updated_ranges = std::mem::take(&mut sync_data_guard.cpu_updated);
                        if !updated_ranges.is_empty() {
                            for range in updated_ranges.ranges.into_iter() {
                                let local_range = range;
                                let slab_range = Range {
                                    first_index: id.range.first_index + range.first_index,
                                    last_index: id.range.first_index + range.last_index,
                                };
                                let update_data = Update {
                                    data: sync_data_guard.data[local_range].to_vec(),
                                    range: slab_range,
                                };
                                log::trace!("updating {id} from CPU, local range: {local_range:?}, slab range: {slab_range:?}");
                                cpu_update_ranges.insert(update_data);
                            }
                        }
                        if sync_data_guard.gpu_update_requested {
                            sync_data_guard.gpu_update_requested = false;
                            gpu_update_ranges.insert(id.range);
                        }
                    } else {
                        // Recycle this allocation, it has been dropped from userland
                        log::debug!("recycling {id}");
                        recycle_ranges.insert(id);
                        updates_guard.remove(&id);
                    }
                } else {
                    log::error!("Could not find {id}");
                }
            }
        }

        UpdateSummary {
            cpu_update_ranges, //: cpu_update_ranges.defrag(),
            gpu_update_ranges, //: gpu_update_ranges.defrag(),
            recycle_ranges,    //: recycle_ranges.defrag(),
        }
    }
}
