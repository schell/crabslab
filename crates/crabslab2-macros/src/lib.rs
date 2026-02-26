//! Proc-macros for `crabslab2`.
//!
//! Provides `#[slab_module]` and `#[slab_item]` attribute macros for
//! generating slab serialization code that works on both CPU (Rust) and
//! GPU (WGSL via `wgsl-rs`).

use proc_macro::TokenStream;

/// Attribute macro that processes a module containing `#[slab_item]`-annotated
/// types.
///
/// For each `#[slab_item]` type, generates:
/// - A concrete ID type (`{TypeName}Id`)
/// - A concrete array type (`{TypeName}Array`)
/// - A slab size constant (`{TYPE_NAME}_SLAB_SIZE`)
/// - `{type_name}_from_array` / `{type_name}_to_array` functions
/// - A CPU-side `impl crabslab2::SlabItem` (emitted outside the module)
///
/// After processing, replaces `#[slab_module]` with `#[wgsl]` so that
/// `wgsl-rs` transpiles the module to WGSL.
#[proc_macro_attribute]
pub fn slab_module(_attr: TokenStream, item: TokenStream) -> TokenStream {
    // TODO: Phase 1, Steps 4-8 — implement code generation
    item
}

/// Marker attribute for types inside a `#[slab_module]`.
///
/// This attribute is a no-op on its own. It is processed by
/// `#[slab_module]` to identify types that need slab serialization
/// code generation.
#[proc_macro_attribute]
pub fn slab_item(_attr: TokenStream, item: TokenStream) -> TokenStream {
    item
}
