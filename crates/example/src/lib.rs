#![cfg_attr(target_arch = "spirv", no_std)]

#[cfg(feature = "slab")]
mod slab;
#[cfg(feature = "slab")]
pub use slab::*;

#[cfg(feature = "non_slab")]
mod non_slab;
#[cfg(feature = "non_slab")]
pub use non_slab::*;

#[cfg(feature = "new_slab")]
mod new_slab;
#[cfg(feature = "new_slab")]
pub use new_slab::*;
