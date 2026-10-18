# changelog 

## Oct 19, 2026

* `crabslab-derive` 0.7.0: rename-aware `SlabItem` derive
  * resolves the runtime crate at expansion time via `proc-macro-crate`, so renamed dependencies work with no configuration
  * new `#[slab(crate = "...")]` override, for graphs carrying two crabslab versions under different names (serde's `#[serde(crate = "...")]` precedent)
  * generated code targets the runtime the consuming crate actually sees; un-renamed, single-version usage is unchanged
* `crabslab` 0.7.1: re-exports `crabslab-derive` 0.7.0 (published 0.7.0 pins derive `^0.4.7`)
* two crabslab versions can now coexist in one dependency graph, one derive node per runtime; see the README recipe, which also documents the `crabslab-derive = "=0.4.6"` pin required by `crabslab` 0.6.6 (0.4.7 removed `impl_slabitem_tuples!`, breaking fresh resolves of 0.6.6)

## Feb 26, 2024 

* changed the trait function `SlabItem::slab_size` to associated constant `SlabItem::SLAB_SIZE`
