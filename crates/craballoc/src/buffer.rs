//! Manages the length and capacity of a buffer.

use std::{
    ops::Deref,
    sync::{atomic::AtomicUsize, Arc, RwLock},
};

pub mod manager;

/// A buffer of `u32`s.
///
/// A thin wrapper around a buffer `T` that provides the ability to tell
/// if the buffer has been invalidated by the [`SlabAllocator`] that it
/// originated from.
///
/// Invalidation happens when the slab resizes. For this reason it is
/// important to create as many values as necessary _before_ calling
/// [`SlabAllocator::commit`] to avoid unnecessary invalidation.
pub struct SlabBuffer<T> {
    // Id of the slab's last `commit` invocation.
    slab_commit_invocation_k: Arc<AtomicUsize>,
    // Id of the slab's last buffer invalidation.
    slab_invalidation_k: Arc<AtomicUsize>,
    // The slab's `slab_update_k` at the time of this buffer's creation.
    buffer_creation_k: usize,
    // The buffer created at `buffer_creation_k`
    buffer: Arc<T>,
    // The buffer the source slab is currently working with
    source_slab_buffer: Arc<RwLock<Option<SlabBuffer<T>>>>,
}

impl<T> Clone for SlabBuffer<T> {
    fn clone(&self) -> Self {
        Self {
            slab_commit_invocation_k: self.slab_commit_invocation_k.clone(),
            slab_invalidation_k: self.slab_invalidation_k.clone(),
            buffer_creation_k: self.buffer_creation_k,
            buffer: self.buffer.clone(),
            source_slab_buffer: self.source_slab_buffer.clone(),
        }
    }
}

impl<T> Deref for SlabBuffer<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.buffer
    }
}

impl<T> SlabBuffer<T> {
    /// Create a new `SlabBuffer`.
    pub(crate) fn new(
        invalidation_k: Arc<AtomicUsize>,
        invocation_k: Arc<AtomicUsize>,
        buffer: T,
        source_slab_buffer: Arc<RwLock<Option<SlabBuffer<T>>>>,
    ) -> Self {
        SlabBuffer {
            buffer: buffer.into(),
            buffer_creation_k: invalidation_k.load(std::sync::atomic::Ordering::Relaxed),
            slab_invalidation_k: invalidation_k,
            slab_commit_invocation_k: invocation_k,
            source_slab_buffer,
        }
    }

    /// Returns the timestamp at which the internal buffer was created.
    ///
    /// The returned timestamp is not a unix timestamp. It is a
    /// monotonically increasing count of buffer invalidations.
    pub fn creation_time(&self) -> usize {
        self.buffer_creation_k
    }
}
