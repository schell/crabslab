# Phase 1: `crabslab2` and `#[slab_module]` Proc-Macro

**Status:** Complete
**Estimated effort:** 1-1.5 weeks
**Prerequisites:** Phase 0

## Overview

Remove old crates from the workspace, create `crates/crabslab2` (core library)
and `crates/crabslab2-macros` (proc-macro crate), define the new `SlabItem`
trait, implement `#[slab_module]` and `#[slab_item]` attribute macros, and
provide primitive `SlabItem` impls.

---

## 1.1 Workspace changes

Remove the following from workspace members (they remain on disk for reference):
- `crates/crabslab`
- `crates/crabslab-derive`
- `crates/craballoc-test-shaders`
- `crates/craballoc-test-wire-types`

Add new workspace members:
- `crates/crabslab2`
- `crates/crabslab2-macros`

`crates/craballoc` stays as a workspace member but will not compile until
Phase 2.

---

## 1.2 The `SlabItem` trait (`crabslab2`)

```rust
/// CPU-side trait for types that can be stored in a `u32` slab.
///
/// Auto-implemented by the `#[slab_module]` macro for `#[slab_item]` types.
/// Can also be implemented manually for primitive types.
pub trait SlabItem: Sized + 'static {
    /// The number of `u32` slots this type occupies in a slab.
    const SLAB_SIZE: usize;

    /// The fixed-size `[u32; Self::SLAB_SIZE]` array type for this slab item.
    type Array: AsRef<[u32]> + AsMut<[u32]> + Default;

    /// Serialize this value into a `[u32; Self::SLAB_SIZE]` array.
    fn to_array(&self) -> Self::Array;

    /// Deserialize a value from a `[u32; Self::SLAB_SIZE]` array.
    fn from_array(arr: Self::Array) -> Self;
}
```

Helper functions (free functions in `crabslab2`):

```rust
/// Read a `SlabItem` from a `u32` slice at the given index.
pub fn slab_read<T: SlabItem>(slab: &[u32], index: usize) -> T {
    let mut arr = T::Array::default();
    arr.as_mut().copy_from_slice(&slab[index..index + T::SLAB_SIZE]);
    T::from_array(arr)
}

/// Write a `SlabItem` into a `u32` slice at the given index.
pub fn slab_write<T: SlabItem>(slab: &mut [u32], index: usize, val: &T) {
    let arr = val.to_array();
    slab[index..index + T::SLAB_SIZE].copy_from_slice(arr.as_ref());
}
```

---

## 1.3 Primitive `SlabItem` impls

Manual impls for `u32`, `i32`, `f32`, and `bool`. No tuples, no glam types
(wgsl-rs provides its own vector/matrix types).

| Type | `SLAB_SIZE` | `type Array` | Notes |
|---|---|---|---|
| `u32` | 1 | `[u32; 1]` | Identity |
| `i32` | 1 | `[u32; 1]` | Bitcast |
| `f32` | 1 | `[u32; 1]` | `to_bits` / `from_bits` |
| `bool` | 1 | `[u32; 1]` | `1` = true, `0` = false |

---

## 1.4 `#[slab_module]` macro behavior

### Design decisions

- **`SLAB_SIZE` is an associated constant** on the type (`Data::SLAB_SIZE`),
  not a module-level constant. wgsl-rs transpiles associated constants.
- **`from_array` / `to_array` are `impl` methods** on the type
  (`Data::from_array`, `Data::to_array`), not freestanding functions.
- **All `#[slab_item]` types are `Copy`.** This allows the `to_array` method
  to take `self` by value (WGSL semantics) and the CPU-side `SlabItem` proxy
  to dereference `*self`.
- **`wgsl-rs` is a regular dependency** of `crabslab2`. The bitcast functions
  (`bitcast_f32`, `bitcast_u32`, `bitcast_i32`) come from `wgsl_rs::std::*`
  and work on CPU.
- **`#[slab_module]` auto-injects `use wgsl_rs::std::*;`** into the module
  if the import is not already present.
- **CPU-side `impl SlabItem` is a thin proxy** that delegates to the in-module
  `Type::SLAB_SIZE`, `Type::from_array`, and `Type::to_array`. The proxy is
  identical for every type — only the type name changes.
- **Trait impls are emitted inside the module** so that `#[wgsl]` can pass
  them through to Rust without generating WGSL. Uses unqualified type names
  (no `mod::Type` path prefix).

### Input

```rust
#[slab_module]
pub mod data_types {
    use wgsl_rs::std::*;

    #[slab_item]
    pub struct Data {
        pub i: u32,
        pub float_val: f32,
        pub ints_0: u32,
        pub ints_1: u32,
    }
}
```

### Output

```rust
pub mod data_types {
    use wgsl_rs::std::*;

    pub struct Data {
        pub i: u32,
        pub float_val: f32,
        pub ints_0: u32,
        pub ints_1: u32,
    }

    // --- Generated impl block with SLAB_SIZE, from_array, to_array ---

    impl Data {
        pub const SLAB_SIZE: usize =
            u32::SLAB_SIZE + f32::SLAB_SIZE + u32::SLAB_SIZE + u32::SLAB_SIZE;

        pub fn from_array(u32s: [u32; Data::SLAB_SIZE]) -> Data {
            Data {
                i: u32s[0usize],
                float_val: bitcast_f32(u32s[0usize + u32::SLAB_SIZE]),
                ints_0: u32s[0usize + u32::SLAB_SIZE + f32::SLAB_SIZE],
                ints_1: u32s[0usize + u32::SLAB_SIZE + f32::SLAB_SIZE + u32::SLAB_SIZE],
            }
        }

        pub fn to_array(d: Data) -> [u32; Data::SLAB_SIZE] {
            let mut arr = [0u32; Data::SLAB_SIZE];
            arr[0usize] = d.i;
            arr[0usize + u32::SLAB_SIZE] = bitcast_u32(d.float_val);
            arr[0usize + u32::SLAB_SIZE + f32::SLAB_SIZE] = d.ints_0;
            arr[0usize + u32::SLAB_SIZE + f32::SLAB_SIZE + u32::SLAB_SIZE] = d.ints_1;
            arr
        }
    }

    // --- Generated ID type ---

    pub struct DataId {
        pub inner: u32,
    }

    impl DataId {
        pub const NONE: DataId = DataId { inner: 4294967295u32 };
        pub const ZERO: DataId = DataId { inner: 0u32 };

        pub const SLAB_SIZE: usize = u32::SLAB_SIZE;

        pub fn new(index: u32) -> DataId {
            DataId { inner: index }
        }

        pub fn is_none(id: DataId) -> bool {
            id.inner == 4294967295u32
        }

        pub fn from_array(u32s: [u32; DataId::SLAB_SIZE]) -> DataId {
            DataId { inner: u32s[0usize] }
        }

        pub fn to_array(d: DataId) -> [u32; DataId::SLAB_SIZE] {
            let mut arr = [0u32; DataId::SLAB_SIZE];
            arr[0usize] = d.inner;
            arr
        }
    }

    // --- Generated array type ---

    pub struct DataArray {
        pub id: DataId,
        pub len: u32,
    }

    impl DataArray {
        pub const NONE: DataArray = DataArray {
            id: DataId::NONE,
            len: 0u32,
        };

        pub const SLAB_SIZE: usize = DataId::SLAB_SIZE + u32::SLAB_SIZE;

        pub fn at(arr: DataArray, index: u32) -> DataId {
            if index >= arr.len {
                DataId::NONE
            } else {
                DataId::new(arr.id.inner + Data::SLAB_SIZE * index)
            }
        }

        pub fn from_array(u32s: [u32; DataArray::SLAB_SIZE]) -> DataArray {
            DataArray {
                id: DataId { inner: u32s[0usize] },
                len: u32s[0usize + DataId::SLAB_SIZE],
            }
        }

        pub fn to_array(d: DataArray) -> [u32; DataArray::SLAB_SIZE] {
            let mut arr = [0u32; DataArray::SLAB_SIZE];
            arr[0usize] = d.id.inner;
            arr[0usize + DataId::SLAB_SIZE] = d.len;
            arr
        }
    }

    // --- Generated trait impls (inside module so #[wgsl] passes through) ---

    impl crabslab2::SlabItem for Data {
        const SLAB_SIZE: usize = Data::SLAB_SIZE;
        type Array = [u32; Data::SLAB_SIZE];

        fn to_array(&self) -> Self::Array {
            Data::to_array(*self)
        }

        fn from_array(arr: Self::Array) -> Self {
            Data::from_array(arr)
        }
    }

    impl crabslab2::SlabItem for DataId {
        const SLAB_SIZE: usize = DataId::SLAB_SIZE;
        type Array = [u32; DataId::SLAB_SIZE];

        fn to_array(&self) -> Self::Array {
            DataId::to_array(*self)
        }

        fn from_array(arr: Self::Array) -> Self {
            DataId::from_array(arr)
        }
    }

    impl crabslab2::SlabItem for DataArray {
        const SLAB_SIZE: usize = DataArray::SLAB_SIZE;
        type Array = [u32; DataArray::SLAB_SIZE];

        fn to_array(&self) -> Self::Array {
            DataArray::to_array(*self)
        }

        fn from_array(arr: Self::Array) -> Self {
            DataArray::from_array(arr)
        }
    }
}
```

---

## 1.5 Type mapping rules for `from_array` / `to_array`

Primitive types are special-cased in the macro. All other types are assumed to
be `#[slab_item]` types with `SLAB_SIZE`, `from_array`, and `to_array` methods.

### Primitives (special-cased)

| Rust Type | Slab Size | `from_array` Expression | `to_array` Expression |
|---|---|---|---|
| `u32` | 1 | `u32s[{offset}]` | `arr[{offset}] = d.{field}` |
| `i32` | 1 | `bitcast_i32(u32s[{offset}])` | `arr[{offset}] = bitcast_u32(d.{field})` |
| `f32` | 1 | `bitcast_f32(u32s[{offset}])` | `arr[{offset}] = bitcast_u32(d.{field})` |
| `bool` | 1 | `u32s[{offset}] != 0u32` | `arr[{offset}] = if d.{field} { 1u32 } else { 0u32 }` |

### Nested types (default)

For any non-primitive field of type `Foo`, the macro generates a while-loop
sub-array copy that delegates to `Foo::from_array` / `Foo::to_array`. This
approach does not require the macro to know `Foo`'s slab size at expansion
time — the compiler resolves `Foo::SLAB_SIZE` statically.

**`from_array` pattern for nested `Foo` at offset `{offset}`:**

```rust
{
    let mut sub = [0u32; Foo::SLAB_SIZE];
    let mut j = 0usize;
    while j < Foo::SLAB_SIZE {
        sub[j] = u32s[{offset} + j];
        j += 1;
    }
    Foo::from_array(sub)
}
```

**`to_array` pattern for nested `Foo` at offset `{offset}`:**

```rust
{
    let inner = Foo::to_array(d.{field});
    let mut j = 0usize;
    while j < Foo::SLAB_SIZE {
        arr[{offset} + j] = inner[j];
        j += 1;
    }
}
```

### Offset tracking

Each field's offset is a `TokenStream` expression representing the cumulative
sum of preceding fields' slab sizes. For the first field, the offset is
`0usize`. For subsequent fields, the offset is
`0usize + Field1Type::SLAB_SIZE + Field2Type::SLAB_SIZE + ...`. These are
const expressions that the Rust compiler evaluates statically.

---

## 1.6 Nested struct handling

Nested `#[slab_item]` types are handled uniformly by the while-loop copy
pattern described in §1.5. The macro does not need to resolve nested type
sizes — it references `Foo::SLAB_SIZE` and lets the compiler resolve it.

Example for `pub struct ArrayChange { pub i: u32, pub change: DataChange }`:

```rust
impl ArrayChange {
    pub const SLAB_SIZE: usize = u32::SLAB_SIZE + DataChange::SLAB_SIZE;

    pub fn from_array(u32s: [u32; ArrayChange::SLAB_SIZE]) -> ArrayChange {
        ArrayChange {
            i: u32s[0usize],
            change: {
                let mut sub = [0u32; DataChange::SLAB_SIZE];
                let mut j = 0usize;
                while j < DataChange::SLAB_SIZE {
                    sub[j] = u32s[0usize + u32::SLAB_SIZE + j];
                    j += 1;
                }
                DataChange::from_array(sub)
            },
        }
    }

    pub fn to_array(d: ArrayChange) -> [u32; ArrayChange::SLAB_SIZE] {
        let mut arr = [0u32; ArrayChange::SLAB_SIZE];
        arr[0usize] = d.i;
        {
            let inner = DataChange::to_array(d.change);
            let mut j = 0usize;
            while j < DataChange::SLAB_SIZE {
                arr[0usize + u32::SLAB_SIZE + j] = inner[j];
                j += 1;
            }
        }
        arr
    }
}
```

---

## 1.7 Enum support

For `#[repr(u32)]` enums annotated with `#[slab_item]`:

```rust
#[slab_item]
#[repr(u32)]
pub enum DataChangeTy {
    I = 0,
    Float = 1,
    Ints = 2,
}
```

Generates:
- `impl DataChangeTy { SLAB_SIZE = 1, from_array, to_array }`
- `DataChangeTyId { inner: u32 }` with NONE, ZERO, new, is_none, SLAB_SIZE,
  from_array, to_array
- `DataChangeTyArray` with NONE, at, SLAB_SIZE, from_array, to_array
- CPU-side `impl SlabItem` proxies for all three types

---

## 1.8 Tuple struct support

**Not supported.** `#[wgsl]` does not support tuple structs, so `#[slab_item]`
rejects them with a compile error. Use named structs instead:

```rust
#[slab_item]
pub struct InvocationCount {
    pub inner: u32,
}
```

---

## 1.9 Array type generation

Every `#[slab_item]` type `T` gets a corresponding `TArray` type with two
fields: `id: TId` and `len: u32`. The array type is itself a slab item with
`SLAB_SIZE = TId::SLAB_SIZE + u32::SLAB_SIZE`, and gets `from_array` /
`to_array` methods plus a CPU-side `SlabItem` impl.

The `at` method computes element positions using `T::SLAB_SIZE` as the stride.

---

## 1.10 Naming conventions

| Generated Item | Naming Pattern | Example |
|---|---|---|
| ID type | `{TypeName}Id` | `DataId` |
| Array type | `{TypeName}Array` | `DataArray` |
| Slab size constant | `{TypeName}::SLAB_SIZE` | `Data::SLAB_SIZE` |
| From-array method | `{TypeName}::from_array` | `Data::from_array` |
| To-array method | `{TypeName}::to_array` | `Data::to_array` |

---

## 1.11 Implementation steps

| Step | Description | Status |
|---|---|---|
| 1 | Remove old crates from workspace members | Done |
| 2 | Create `crates/crabslab2-macros` skeleton | Done |
| 3 | Create `crates/crabslab2` with `SlabItem` trait, primitive impls, helpers | Done |
| 4-5 | Implement `#[slab_module]` codegen for structs + CPU trait impls | Done |
| 6 | Add enum support (`#[repr(u32)]` enums) | Done |
| 7 | ~~Tuple struct support~~ | N/A (`#[wgsl]` does not support tuple structs) |
| 8 | Add nested struct integration tests | Done |
| 9 | Integration tests (CPU round-trip, WGSL validation) | Done |

### Step 4-5 implementation details

**`crates/crabslab2/Cargo.toml`:** Add `wgsl-rs` as a regular dependency.

**`crates/crabslab2-macros/src/lib.rs`:** Rewrite with:

- `slab_module` entry point: parse `ItemMod`, call `process_module()`, output
  modified module with all generated items inside.
- `process_module`: walk items, find `#[slab_item]` types, strip markers,
  generate code, auto-inject `use wgsl_rs::std::*;` if missing.
- `process_struct`: for each `#[slab_item]` struct, generate (all inside the
  module):
  - `impl Type` block with `SLAB_SIZE`, `from_array`, `to_array`
  - ID struct + impl
  - Array struct + impl
  - `impl crabslab2::SlabItem` proxies for all three types (inside module
    so `#[wgsl]` passes them through without WGSL generation)
- Field codegen: primitives (`u32`, `i32`, `f32`, `bool`) are special-cased
  with direct index / bitcast. Everything else uses the while-loop sub-array
  copy + `Type::from_array` / `Type::to_array` pattern.
- Offset tracking: cumulative `TokenStream` const expressions.
- Naming helpers: PascalCase type name derives ID name, array name.

**`crates/crabslab2/src/lib.rs`:** Add integration tests using `#[slab_module]`
with a struct containing `u32` and `f32` fields. Test `SlabItem` round-trip
via `slab_read`/`slab_write`, test ID type (NONE, ZERO, new, is_none), and
test array type (NONE, at).
