//! Manages the length and capacity of a buffer.

use std::{
    borrow::Cow,
    num::NonZeroU32,
    sync::{
        atomic::{AtomicBool, AtomicU32, AtomicUsize},
        Arc, RwLock,
    },
};

use crabslab::SlabItem;

use crate::{buffer::SlabBuffer, range::Range, runtime::IsRuntime};

const ATOMIC_ORDERING: std::sync::atomic::Ordering = std::sync::atomic::Ordering::Relaxed;

/// Manages a runtime buffer.
///
/// The `BumpAllocator` is responsible for allocating contiguous u32 ranges on the
/// slab. It maintains a count of the number of u32 slots allocated (the length)
/// and the number of u32 slots available to be allocated (the capacity).
/// It resizes the buffer during commit when allocations would overrun the capacity.
pub struct BumpAllocator<R: IsRuntime> {
    label: Arc<Cow<'static, str>>,

    runtime: R,

    buffer_length: Arc<AtomicU32>,
    buffer_capacity: Arc<RwLock<u32>>,
    buffer_usages: R::BufferUsages,
    buffer_needs_expansion: Arc<AtomicBool>,
    buffer: Arc<RwLock<Option<SlabBuffer<R::Buffer>>>>,

    /// The `commit_height` when the last buffer invalidation happened
    buffer_creation_height: Arc<AtomicUsize>,
    /// The next monotonically increasing commit invocation "height"
    commit_height: Arc<AtomicUsize>,
}

impl<Runtime: IsRuntime + std::fmt::Debug> std::fmt::Debug for BumpAllocator<Runtime> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Self {
            runtime: _,
            label,
            buffer_length,
            buffer_capacity,
            buffer_usages: _,
            buffer: _,
            buffer_needs_expansion,

            buffer_creation_height: _,
            commit_height: commit_count,
        } = self;
        f.debug_struct("BufferManager")
            .field("label", label)
            .field("length", &buffer_length.load(ATOMIC_ORDERING))
            .field("capacity", &buffer_capacity.read().unwrap())
            .field(
                "needs_expansion",
                &buffer_needs_expansion.load(ATOMIC_ORDERING),
            )
            .field("commit_count", &commit_count)
            .finish()
    }
}

impl<R: IsRuntime> Clone for BumpAllocator<R> {
    fn clone(&self) -> Self {
        Self {
            runtime: self.runtime.clone(),
            label: self.label.clone(),
            buffer_length: self.buffer_length.clone(),
            buffer_capacity: self.buffer_capacity.clone(),
            buffer_usages: self.buffer_usages.clone(),
            buffer: self.buffer.clone(),
            buffer_needs_expansion: self.buffer_needs_expansion.clone(),
            buffer_creation_height: self.buffer_creation_height.clone(),
            commit_height: self.commit_height.clone(),
        }
    }
}

impl<R: IsRuntime> BumpAllocator<R> {
    pub fn new(
        runtime: impl AsRef<R>,
        label: impl Into<Cow<'static, str>>,
        buffer_usages: R::BufferUsages,
    ) -> Self {
        Self {
            label: Arc::new(label.into()),
            runtime: runtime.as_ref().clone(),
            buffer_length: Arc::new(0.into()),
            // Start with size 1, because some of `wgpu`'s validation depends on it.
            // See <https://github.com/gfx-rs/wgpu/issues/6414> for more info.
            buffer_capacity: Arc::new(RwLock::new(1)),
            buffer_usages,
            buffer: Arc::new(RwLock::new(None)),
            buffer_needs_expansion: Arc::new(true.into()),
            buffer_creation_height: Default::default(),
            commit_height: Default::default(),
        }
    }

    /// Returns the runtime of the buffer being managed.
    pub fn runtime(&self) -> &R {
        &self.runtime
    }

    /// Returns the label of the buffer being managed.
    pub fn label(&self) -> &str {
        &self.label
    }

    /// The length of the underlying buffer, in u32 slots.
    pub fn len(&self) -> u32 {
        self.buffer_length.load(ATOMIC_ORDERING)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn capacity(&self) -> u32 {
        *self.buffer_capacity.read().unwrap()
    }

    /// Returns whether the buffer needs to be expanded.
    ///
    /// Sets `buffer_needs_expansion` to `false`.
    fn clear_buffer_needs_expansion(&self) -> bool {
        self.buffer_needs_expansion.swap(false, ATOMIC_ORDERING)
    }

    pub fn create_buffer(&self, runtime: &R, label: &str) -> R::Buffer {
        log::trace!("creating a new buffer for {label}");
        let capacity = self.buffer_capacity.read().unwrap();
        let buffer =
            runtime.buffer_create(*capacity as usize, Some(label), self.buffer_usages.clone());
        log::trace!("  ...done!");
        buffer
    }

    #[deprecated(since = "0.4.0", note = "Use BufferManager::alloc instead")]
    pub fn maybe_expand_to_fit<T: SlabItem>(&self, len: usize) {
        if let Some(spaces) = NonZeroU32::new(T::SLAB_SIZE as u32 * len as u32) {
            let _ = self.alloc(spaces);
        }
    }

    pub fn create_slab_buffer(
        &self,
        new_buffer: R::Buffer,
        invalidation_k: Arc<AtomicUsize>,
        invocation_k: Arc<AtomicUsize>,
    ) -> SlabBuffer<R::Buffer> {
        SlabBuffer::new(
            invalidation_k,
            invocation_k,
            new_buffer,
            self.buffer.clone(),
        )
    }

    /// Recreate the internal buffer, writing the contents of the previous buffer (if it
    /// exists) to the new one, then return the new buffer.
    fn recreate_buffer(&self) -> SlabBuffer<R::Buffer> {
        // Create the new buffer
        log::trace!("recreating buffer '{}'", self.label);
        let new_buffer = self.create_buffer(&self.runtime, self.label.as_ref());
        // Replace the old buffer
        let mut guard = self.buffer.write().unwrap();
        if let Some(old_buffer) = guard.take() {
            // If the old buffer exists we need to copy the data from one to the other.
            log::trace!("copying previous data to the new buffer");
            self.runtime
                .buffer_copy(&old_buffer, &new_buffer, Some(self.label.as_ref()));
        } else {
            log::trace!("this is a new buffer's first creation");
        }
        let slab_buffer = self.create_slab_buffer(
            new_buffer,
            self.buffer_creation_height.clone(),
            self.commit_height.clone(),
        );
        *guard = Some(slab_buffer.clone());
        slab_buffer
    }

    /// Allocate `spaces` u32 slots in the buffer, resizing the buffer if needed.
    ///
    /// Returns the range of u32 slots allocated.
    ///
    /// ## Note
    /// Keep in mind that the `Range` returned is not [`craballoc::range::Range`](crate::range::Range),
    /// **not** the `Range` in `std`.
    pub fn alloc(&self, spaces: NonZeroU32) -> Range {
        let spaces = u32::from(spaces);
        let mut capacity = self.buffer_capacity.write().unwrap();
        let start = self.buffer_length.fetch_add(spaces, ATOMIC_ORDERING);
        let capacity_needed = start + spaces;
        if capacity_needed > *capacity {
            let mut new_capacity = *capacity * 2;
            while new_capacity < capacity_needed {
                new_capacity = (new_capacity * 2).max(2);
            }
            *capacity = new_capacity;
            self.buffer_needs_expansion.store(true, ATOMIC_ORDERING);
        }
        Range {
            first_index: start,
            last_index: start + (spaces - 1),
        }
    }

    /// Returns the internal [`SlabBuffer`], if any.
    ///
    /// Returns `None` before [`BufferManager::commit`] is called at least once.
    ///
    /// ## Note
    /// This returns the current internal buffer and does not commit any data
    /// awaiting synchronization. To ensure you have a buffer that contains
    /// the latest data, use [`BufferManager::commit`].
    pub fn get_buffer(&self) -> Option<SlabBuffer<R::Buffer>> {
        let guard = self.buffer.read().unwrap();
        guard.as_ref().cloned()
    }

    /// Perform upkeep on the slab, synchronizing changes to the internal buffer.
    ///
    /// Changes made to allocated values stored on this slab are not committed
    /// until this function has been called, and sometimes not until the runtime
    /// has finished synchronizing the CPU to the GPU (eg with `wgpu::Device::poll`).
    ///
    /// The internal buffer is not created until the after the first time this
    /// function is called.
    ///
    /// Returns a [`SlabBuffer`] wrapping the internal buffer that is currently
    /// in use by the allocator.
    pub fn commit(&self) -> SlabBuffer<R::Buffer> {
        let invocation_k = self
            .commit_height
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            + 1;
        let buffer = if self.clear_buffer_needs_expansion() {
            log::trace!("buffer '{}' needs expansion", self.label);
            // The previous buffer is invalidated
            self.buffer_creation_height
                .store(invocation_k, std::sync::atomic::Ordering::Relaxed);
            self.recreate_buffer()
        } else {
            // UNWRAP: Safe because we know it exists or else it would need expansion
            let guard = self.buffer.read().unwrap();
            let slab_buffer = guard.as_ref().unwrap();
            slab_buffer.clone()
        };

        buffer
    }

    /// Returns the current buffer's invalidation height.
    pub fn buffer_creation_time(&self) -> usize {
        self.buffer_creation_height.load(ATOMIC_ORDERING)
    }
}
