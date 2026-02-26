# Phase 1: `crabslab2` and `#[slab_module]` Proc-Macro

**Status:** In Progress
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

    /// The fixed-size `[u32; N]` array type for this slab item.
    type Array: AsRef<[u32]> + AsMut<[u32]> + Default;

    /// Serialize this value into a `[u32; N]` array.
    fn to_array(&self) -> Self::Array;

    /// Deserialize a value from a `[u32; N]` array.
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

**Input:** A Rust module annotated with `#[slab_module]`:

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

**Output:** The macro rewrites the module to:

```rust
#[wgsl]
pub mod data_types {
    use wgsl_rs::std::*;

    pub struct Data {
        pub i: u32,
        pub float_val: f32,
        pub ints_0: u32,
        pub ints_1: u32,
    }

    // --- Generated ID type ---

    pub struct DataId {
        pub inner: u32,
    }

    impl DataId {
        pub const NONE: DataId = DataId { inner: 4294967295u32 };
        pub const ZERO: DataId = DataId { inner: 0u32 };

        pub fn new(index: u32) -> DataId {
            DataId { inner: index }
        }

        pub fn is_none(id: DataId) -> bool {
            id.inner == 4294967295u32
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

        pub fn at(arr: DataArray, index: u32) -> DataId {
            if index >= arr.len {
                DataId::NONE
            } else {
                DataId::new(arr.id.inner + DATA_SLAB_SIZE * index)
            }
        }
    }

    // --- Generated constant ---

    pub const DATA_SLAB_SIZE: usize = 4;

    // --- Generated from_array function ---

    pub fn data_from_array(u32s: [u32; DATA_SLAB_SIZE]) -> Data {
        Data {
            i: u32s[0usize],
            float_val: bitcast_f32(u32s[1usize]),
            ints_0: u32s[2usize],
            ints_1: u32s[3usize],
        }
    }

    // --- Generated to_array function ---

    pub fn data_to_array(d: Data) -> [u32; DATA_SLAB_SIZE] {
        let mut arr = [0u32; DATA_SLAB_SIZE];
        arr[0usize] = d.i;
        arr[1usize] = bitcast_u32(d.float_val);
        arr[2usize] = d.ints_0;
        arr[3usize] = d.ints_1;
        arr
    }
}

// --- Generated OUTSIDE the #[wgsl] module (CPU-only trait impl) ---

impl crabslab2::SlabItem for data_types::Data {
    const SLAB_SIZE: usize = 4;
    type Array = [u32; 4];

    fn to_array(&self) -> [u32; 4] {
        [
            self.i,
            self.float_val.to_bits(),
            self.ints_0,
            self.ints_1,
        ]
    }

    fn from_array(arr: [u32; 4]) -> Self {
        data_types::Data {
            i: arr[0],
            float_val: f32::from_bits(arr[1]),
            ints_0: arr[2],
            ints_1: arr[3],
        }
    }
}
```

---

## 1.5 Type mapping rules for `from_array` / `to_array`

| Rust Field Type | Slab Size | `from_array` Expression | `to_array` Expression |
|---|---|---|---|
| `u32` | 1 | `u32s[idx]` | `arr[idx] = val` |
| `i32` | 1 | `bitcast_i32(u32s[idx])` | `arr[idx] = bitcast_u32(val)` |
| `f32` | 1 | `bitcast_f32(u32s[idx])` | `arr[idx] = bitcast_u32(val)` |
| `bool` | 1 | `u32s[idx] != 0u32` | `arr[idx] = if val { 1u32 } else { 0u32 }` |
| `Vec2f` | 2 | `vec2f(bitcast_f32(u32s[idx]), ...)` | 2 component writes |
| `Vec3f` | 3 | `vec3f(bitcast_f32(u32s[idx]), ...)` | 3 component writes |
| `Vec4f` | 4 | `vec4f(bitcast_f32(u32s[idx]), ...)` | 4 component writes |
| `Vec2u` / `Vec3u` / `Vec4u` | 2/3/4 | `vec2u(u32s[idx], ...)` | component writes |
| `Vec2i` / `Vec3i` / `Vec4i` | 2/3/4 | `vec2i(bitcast_i32(u32s[idx]), ...)` | component writes |
| Nested `#[slab_item]` struct | N | `inner_from_array([u32s[idx], ..., u32s[idx+N-1]])` | element-by-element copy from `inner_to_array(val)` |
| `#[repr(u32)]` `#[slab_item]` enum | 1 | `u32s[idx]` (discriminant) | `arr[idx] = val` (discriminant) |
| Tuple struct `Foo(pub u32)` | inner type's size | Same as inner type | Same as inner type |

---

## 1.6 Nested struct handling

When a struct contains a nested `#[slab_item]` struct, the generated
`from_array` function builds an inner array literal element by element:

```rust
// For: pub struct ArrayChange { pub i: u32, pub change: DataChange }
// Where DataChange has SLAB_SIZE = 4, so ArrayChange has SLAB_SIZE = 5

pub fn array_change_from_array(u32s: [u32; 5]) -> ArrayChange {
    ArrayChange {
        i: u32s[0usize],
        change: data_change_from_array([
            u32s[1usize],
            u32s[2usize],
            u32s[3usize],
            u32s[4usize],
        ]),
    }
}

pub fn array_change_to_array(d: ArrayChange) -> [u32; 5] {
    let mut arr = [0u32; 5];
    arr[0usize] = d.i;
    let inner = data_change_to_array(d.change);
    arr[1usize] = inner[0usize];
    arr[2usize] = inner[1usize];
    arr[3usize] = inner[2usize];
    arr[4usize] = inner[3usize];
    arr
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
- `DataChangeTyId { inner: u32 }`
- `DATA_CHANGE_TY_SLAB_SIZE: usize = 1`
- `data_change_ty_from_array([u32; 1]) -> DataChangeTy`
- `data_change_ty_to_array(DataChangeTy) -> [u32; 1]`

---

## 1.8 Tuple struct / primitive wrapper support

```rust
#[slab_item]
pub struct InvocationCount(pub u32);
```

Generates:
- `InvocationCountId { inner: u32 }`
- `INVOCATION_COUNT_SLAB_SIZE: usize = 1`
- `invocation_count_from_array` / `invocation_count_to_array`

---

## 1.9 Array type generation

Every `#[slab_item]` type gets a corresponding array type:

```rust
pub struct DataArray {
    pub id: DataId,
    pub len: u32,
}

impl DataArray {
    pub const NONE: DataArray = DataArray {
        id: DataId::NONE,
        len: 0u32,
    };

    pub fn at(arr: DataArray, index: u32) -> DataId {
        if index >= arr.len {
            DataId::NONE
        } else {
            DataId::new(arr.id.inner + DATA_SLAB_SIZE * index)
        }
    }
}
```

The array type also gets its own `from_array`/`to_array` functions since
`DataArray` is itself a 2-element slab item (`id.inner: u32` + `len: u32`).

---

## 1.10 Naming conventions

| Generated Item | Naming Pattern | Example |
|---|---|---|
| ID type | `{TypeName}Id` | `DataId` |
| Array type | `{TypeName}Array` | `DataArray` |
| Slab size constant | `{TYPE_NAME}_SLAB_SIZE: usize` | `DATA_SLAB_SIZE` |
| From-array function | `{type_name}_from_array` | `data_from_array` |
| To-array function | `{type_name}_to_array` | `data_to_array` |

---

## 1.11 Implementation steps

| Step | Description |
|---|---|
| 1 | Remove `crabslab`, `crabslab-derive`, `craballoc-test-shaders`, `craballoc-test-wire-types` from workspace members |
| 2 | Create `crates/crabslab2-macros` with skeleton `#[slab_module]` / `#[slab_item]` |
| 3 | Create `crates/crabslab2` with `SlabItem` trait, primitive impls, `slab_read`/`slab_write` helpers |
| 4 | Implement `#[slab_item]` code generation for structs (ID, Array, slab size, from/to array) |
| 5 | Implement `#[slab_module]` outer wrapper (strip `#[slab_item]`, emit `#[wgsl]`, emit CPU trait impls) |
| 6 | Add enum support |
| 7 | Add tuple struct support |
| 8 | Add nested struct support |
| 9 | Integration tests (CPU round-trip, WGSL validation) |
