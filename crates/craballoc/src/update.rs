//! An abstraction over a lazy updating mechanism.
//!
//! This abstraction allows us to send updates to the GPU lazily,
//! by updating a slab's subslice range that corresponds to an
//! allocated `T` _locally_, and then reading those values
//! once during a commit phase, guided by update notifications.

use std::{
    collections::BTreeMap,
    hash::Hash,
    ops::{Deref, DerefMut},
    sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard, Weak},
};

use rustc_hash::{FxHashMap, FxHashSet};

use crate::range::{IsRange, Range, RangeManager};

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
            data: vec![0u32; len],
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

#[derive(Clone, Copy, Debug)]
pub enum SourceMessage {
    CpuCacheUpdated { id: SourceId },
    GpuSync { id: SourceId, should_sync: bool },
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
    sender: async_channel::Sender<SourceMessage>,
    data: Arc<RwLock<SynchronizationData>>,
}

impl Drop for CpuUpdateSource {
    fn drop(&mut self) {
        self.mark_updated();
    }
}

impl CpuUpdateSource {
    fn mark_updated(&self) {
        let _ = self.sender.try_send(SourceMessage::CpuCacheUpdated {
            id: self.source_id(),
        });
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

    /// Set whether or not this source's CPU cache is synchronized with the
    /// backend every frame.
    pub fn set_gpu_sync(&self, should_sync: bool) {
        let _ = self.sender.try_send(SourceMessage::GpuSync {
            id: self.source_id(),
            should_sync,
        });
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

#[derive(Default)]
pub struct GpuUpdates {
    /// CPU data requesting a GPU update, keyed by the range it occupies on the slab.
    ///
    /// Source ranges may be contiguous with each other, but must _not_ (and _will not_)
    /// be overlapping.
    sources: BTreeMap<Range, Arc<RwLock<SynchronizationData>>>,
}

impl std::fmt::Debug for GpuUpdates {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuUpdates")
            .field("sources", &self.sources.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl GpuUpdates {
    /// Insert a source.
    pub fn insert(&mut self, source: SourceId, data: Arc<RwLock<SynchronizationData>>) {
        self.sources.insert(source.range, data);
    }

    /// Return an iterator over the ranges to be updated.
    pub fn ranges(&self) -> Vec<Range> {
        self.sources.keys().copied().collect()
    }

    /// Apply updates received from the GPU to their request sources.
    pub fn apply(&mut self, updates: Vec<Update>) {
        let Self { sources } = self;
        let mut updates = updates.into_iter();
        let mut sources = sources.into_iter();
        // One update may apply to more than one source, as the requests have been coalesced
        // into disjoint ranges.
        //
        // Both updates and sources are ordered.
        let mut next_update = updates.next();
        let mut next_source = sources.next();
        loop {
            match (next_update.take(), next_source.take()) {
                (None, None) => {
                    log::trace!("  updates are done!");
                    break;
                }
                (None, Some(_)) => unreachable!("  ran out of sources"),
                (Some(_), None) => unreachable!("  ran out of updates"),
                (Some(mut update), Some((source_range, source_data))) => {
                    // Updates and sources should move in lockstep.
                    debug_assert_eq!(
                        update.range.first_index, source_range.first_index,
                        "mismatched first index"
                    );

                    log::trace!("  update: {update:?} source_range: {source_range:?}");

                    {
                        // Get the chunk of data that corresponds to this source and updated it
                        let update_chunk = update.take(source_range.len());
                        debug_assert_eq!(
                            source_range.len(),
                            update_chunk.data.len() as u32,
                            "chunk len {} != source range len {}",
                            update_chunk.data.len(),
                            source_range.len()
                        );
                        log::trace!("    took chunk {update_chunk:?}");
                        let mut guard = source_data.write().unwrap();
                        guard.data.copy_from_slice(&update_chunk.data);
                    }

                    // Put the remainder of the update, if any
                    if !update.range.is_empty() {
                        next_update = Some(update);
                    }
                }
            }
        }
    }
}

pub struct UpdateSummary {
    pub cpu_update_ranges: RangeManager<Update>,
    pub gpu_updates: GpuUpdates,
    pub recycle_ranges: RangeManager<SourceId>,
}

/// Manages updates to and from the CPU.
#[derive(Clone)]
pub struct UpdateManager {
    /// Sends notification of an update at the time of update.
    notifier_sender: async_channel::Sender<SourceMessage>,
    /// Receives update notifications during a commit.
    notifier_receiver: async_channel::Receiver<SourceMessage>,

    /// Weak references to all values that can write updates into a slab
    cpu_cache: Arc<RwLock<FxHashMap<SourceId, CpuData>>>,

    /// Set of ids of the update sources that have updates queued
    update_queue: Arc<RwLock<FxHashSet<SourceId>>>,
    /// Backend update sources.
    ///
    /// These sources have requested synchronization from the backend.
    backend_sync_sources: Arc<RwLock<FxHashSet<SourceId>>>,
}

impl std::fmt::Debug for UpdateManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CpuUpdateManager")
            .field(
                "update_sources_count",
                &self.cpu_cache.read().unwrap().len(),
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
            cpu_cache: Default::default(),
            update_queue: Default::default(),
            backend_sync_sources: Default::default(),
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
        update_source.set_gpu_sync(true);
        let cpu_data = update_source.cpu_data();
        self.cpu_cache.write().unwrap().insert(source, cpu_data);
        update_source
    }

    /// Returns all the `SourceId`s managed by this `UpdateManager`.
    pub fn get_managed_source_ids(&self) -> FxHashSet<SourceId> {
        FxHashSet::from_iter(self.cpu_cache.read().unwrap().keys().copied())
    }

    /// Returns whether any update sources have queued updates waiting to be committed.
    pub fn has_queued_updates(&self) -> bool {
        !self.notifier_receiver.is_empty() || !self.update_queue.read().unwrap().is_empty()
    }

    /// Return the ids of all sources that require updating as a result of their CPU caches
    /// being invalidated.
    ///
    /// This clears the update `SourceId`s from their sources and stores
    /// them for use during the next commit, returning a clone of all updated
    /// sources since  
    pub fn get_updated_source_ids(&self) -> FxHashSet<SourceId> {
        // UNWRAP: panic on purpose
        let mut cpu_cache_set = self.update_queue.write().unwrap();
        let mut backend_set = self.backend_sync_sources.write().unwrap();
        while let Ok(msg) = self.notifier_receiver.try_recv() {
            match msg {
                SourceMessage::CpuCacheUpdated { id } => {
                    cpu_cache_set.insert(id);
                }
                SourceMessage::GpuSync { id, should_sync } => {
                    if should_sync {
                        backend_set.insert(id);
                    } else {
                        backend_set.remove(&id);
                    }
                }
            }
        }
        cpu_cache_set.clone()
    }

    /// Clear the updates in the queue and convert them into a managed set of ranges.
    ///
    /// This ensures that an entire frame of updates is coalesced into the smallest
    /// number of buffer writes as is possible.
    pub fn clear_updated_sources(&self) -> UpdateSummary {
        log::trace!("clearing updated sources and generating the update summary");
        // Ranges of the slab that will be updated from the CPU
        let mut cpu_update_ranges = RangeManager::<Update>::default();
        // Ranges of the slab that will be updated from the GPU,
        // as well as the sources that requested those updates.
        let mut gpu_updates = GpuUpdates::default();
        // Ranges of the slab that have been dropped and should be recycled
        let mut recycle_ranges = RangeManager::<SourceId>::default();

        // Clear the update channel, populating the queue.
        let _ = self.get_updated_source_ids();
        // Then, to avoid losing updates in a race condition, rotate the CPU sources
        //
        // UNWRAP: panic on purpose
        let update_source_ids = std::mem::take(self.update_queue.write().unwrap().deref_mut());

        // UNWRAP: panic on purpose
        let mut backend_sync_source_ids = self.backend_sync_sources.write().unwrap();

        // Prepare all of our GPU buffer writes and gather the set of sources that want
        // backend synchronization
        {
            // Recycle any update sources that are no longer needed, and collect the active
            // sources' updates into `writes`.
            let mut cpu_cache_guard = self.cpu_cache.write().unwrap();
            for id in update_source_ids {
                if let Some(cpu_data) = cpu_cache_guard.get_mut(&id) {
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
                    } else {
                        // Recycle this allocation, it has been dropped from userland
                        log::debug!("recycling {id}");
                        recycle_ranges.insert(id);
                        cpu_cache_guard.remove(&id);
                        backend_sync_source_ids.remove(&id);
                    }
                } else {
                    log::error!("Could not find {id}");
                }
            }
            backend_sync_source_ids.retain(|id| {
                if let Some(cpu_data) = cpu_cache_guard.get_mut(id) {
                    if let Some(sync_data) = cpu_data.synchronization.upgrade() {
                        gpu_updates.insert(*id, sync_data.clone());
                        true
                    } else {
                        // This source has been dropped and will be recycled
                        recycle_ranges.insert(*id);
                        false
                    }
                } else {
                    // This source is not being tracked by the CPU cache, don't track it for backend updates,
                    // as there is no cache value to synchronize.
                    false
                }
            });
        }

        UpdateSummary {
            cpu_update_ranges,
            gpu_updates,
            recycle_ranges,
        }
    }
}
