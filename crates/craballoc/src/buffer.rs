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

    /// Returns the current invalidation key of the slab.
    ///
    /// This key is a monotonically increasing value that indicates the number
    /// of times the slab's buffer has been invalidated. It is used to determine
    /// if the current buffer is still valid or if it has been replaced by a new
    /// buffer due to resizing or other operations.
    pub(crate) fn invalidation_k(&self) -> usize {
        self.slab_invalidation_k
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Returns the current invocation key of the slab.
    ///
    /// This key is a monotonically increasing value that indicates the number
    /// of times the slab's buffer has been committed. It helps track the
    /// version of the buffer in use and is useful for ensuring that operations
    /// are performed on the most recent version of the buffer.
    pub(crate) fn invocation_k(&self) -> usize {
        self.slab_commit_invocation_k
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Returns the timestamp at which the internal buffer was created.
    ///
    /// The returned timestamp is not a unix timestamp. It is a
    /// monotonically increasing count of buffer invalidations.
    pub fn creation_time(&self) -> usize {
        self.buffer_creation_k
    }

    #[deprecated(
        since = "0.4.0",
        note = "Manually track SlabBuffer::creation_time() instead"
    )]
    /// Determines whether this buffer has been invalidated by the slab
    /// it originated from.
    pub fn is_invalid(&self) -> bool {
        self.creation_time() < self.invalidation_k()
    }

    #[deprecated(
        since = "0.4.0",
        note = "Manually track SlabBuffer::creation_time() instead"
    )]
    /// Determines whether this buffer has been invalidated by the slab
    /// it originated from.
    pub fn is_valid(&self) -> bool {
        #[allow(deprecated)]
        !self.is_invalid()
    }

    #[deprecated(
        since = "0.4.0",
        note = "Manually track SlabBuffer::creation_time() instead"
    )]
    /// Returns `true` when the slab's internal buffer has been recreated, and this is that
    /// newly created buffer.
    ///
    /// This will return false if [`SlabAllocator::commit`] has been called since the creation
    /// of this buffer.
    ///
    /// Typically this function is used by structs that own the [`SlabAllocator`]. These owning
    /// structs will call [`SlabAllocator::commit`] which returns a [`SlabBuffer`]. The callsite
    /// can then call [`SlabBuffer::is_new_this_commit`] to determine if any
    /// downstream resources (like bindgroups) need to be recreated.
    ///
    /// This pattern keeps the owning struct from having to also store the `SlabBuffer`.
    pub fn is_new_this_commit(&self) -> bool {
        self.invocation_k() == self.buffer_creation_k
    }

    #[deprecated(
        since = "0.4.0",
        note = "Manually track SlabBuffer::creation_time() instead"
    )]
    /// Syncronize the buffer with the slab's internal buffer.
    ///
    /// This checks to ensure that the internal buffer is the one the slab is working with,
    /// and updates it if the slab is working with a newer buffer.
    ///
    /// Returns `true` if the buffer was updated.
    /// Returns `false` if the buffer remains the same.
    ///
    /// Use the result of this function to invalidate any bind groups or other downstream
    /// resources.
    ///
    /// ## Note
    /// Be cautious when using this function with multiple buffers to invalidate downstream
    /// resources. Keep in mind that using the
    /// [lazy boolean operators](https://doc.rust-lang.org/reference/expressions/operator-expr.html#lazy-boolean-operators)
    /// might not have the effect you are expecting!
    ///
    /// For example:
    ///
    /// ```rust,no_run
    /// use craballoc::prelude::*;
    ///
    /// let buffer_a: SlabBuffer<wgpu::Buffer> = todo!();
    /// let buffer_b: SlabBuffer<wgpu::Buffer> = todo!();
    /// let buffer_c: SlabBuffer<wgpu::Buffer> = todo!();
    ///
    /// let should_invalidate = buffer_a.update_if_invalid()
    ///     || buffer_b.update_if_invalid()
    ///     || buffer_c.update_if_invalid();
    /// ```
    ///
    /// If `buffer_a` is invalid, neither `buffer_b` nor `buffer_c` will be synchronized, because
    /// `||` is lazy in its parameter evaluation.
    ///
    /// Instead, we should write the following:
    ///
    /// ```rust,no_run
    /// use craballoc::prelude::*;
    ///
    /// let buffer_a: SlabBuffer<wgpu::Buffer> = todo!();
    /// let buffer_b: SlabBuffer<wgpu::Buffer> = todo!();
    /// let buffer_c: SlabBuffer<wgpu::Buffer> = todo!();
    ///
    /// let buffer_a_invalid = buffer_a.update_if_invalid();
    /// let buffer_b_invalid = buffer_b.update_if_invalid();
    /// let buffer_c_invalid = buffer_c.update_if_invalid();
    ///
    /// let should_invalidate = buffer_a_invalid || buffer_b_invalid || buffer_c_invalid;
    /// ```
    pub fn update_if_invalid(&mut self) -> bool {
        #[allow(deprecated)]
        if self.is_invalid() {
            // UNWRAP: Safe because it is an invariant of the system. Once the `SlabBuffer`
            // is created, source_slab_buffer will always be Some.
            let updated_buffer = {
                let guard = self.source_slab_buffer.read().unwrap();
                guard.as_ref().unwrap().clone()
            };
            debug_assert!(updated_buffer.is_valid());
            *self = updated_buffer;
            true
        } else {
            false
        }
    }
}
