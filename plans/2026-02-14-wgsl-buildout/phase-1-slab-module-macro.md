# Phase 1: New `#[slab_module]` Proc-Macro (in `crabslab-derive`)

**Status:** In Progress
**Estimated effort:** 1-1.5 weeks
**Prerequisites:** Phase 0

## Overview

Repurpose `crates/crabslab-derive` to contain the `#[slab_module]` attribute
macro alongside (initially) the existing `#[derive(SlabItem)]`.

---

## 1.1 `#[slab_module]` macro behavior

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

impl crabslab::SlabItem for data_types::Data {
    type Id = data_types::DataId;
    const SLAB_SIZE: usize = 4;

    fn slab_read(slab: &[u32], index: u32) -> Self {
        data_types::Data {
            i: slab[index as usize],
            float_val: f32::from_bits(slab[(index + 1) as usize]),
            ints_0: slab[(index + 2) as usize],
            ints_1: slab[(index + 3) as usize],
        }
    }

    fn slab_write(data: &Self, slab: &mut [u32], index: u32) {
        slab[index as usize] = data.i;
        slab[(index + 1) as usize] = data.float_val.to_bits();
        slab[(index + 2) as usize] = data.ints_0;
        slab[(index + 3) as usize] = data.ints_1;
    }
}
```

---

## 1.2 Type mapping rules for `from_array` / `to_array`

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

## 1.3 Nested struct handling

When a struct contains a nested `#[slab_item]` struct, the generated
`from_array` function builds an inner array literal element by element:

```rust
// For: pub struct ArrayChange { pub i: u32, pub change: DataChange }
// Where DataChange has SLAB_SIZE = 4, so ArrayChange has SLAB_SIZE = 5
// (ARRAY_CHANGE_SLAB_SIZE = 5, DATA_CHANGE_SLAB_SIZE = 4)

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

This is verbose but correct, and the inner function calls cross module
boundaries normally.

---

## 1.4 Enum support

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
- `data_change_ty_from_array([u32; 1]) -> DataChangeTy` -- reads the
  discriminant as `u32`. In WGSL, the enum is represented as integer constants.
- `data_change_ty_to_array(DataChangeTy) -> [u32; 1]` -- writes the
  discriminant.

When a struct field has a `#[repr(u32)]` enum type, `from_array` reads a `u32`
from the array and the Rust side reconstructs the enum. In WGSL, the enum is
integer constants, so the field is effectively `u32`.

---

## 1.5 Tuple struct / primitive wrapper support

`#[slab_item]` supports tuple structs wrapping primitives:

```rust
#[slab_item]
pub struct InvocationCount(pub u32);
```

Generates:
- `InvocationCountId { inner: u32 }`
- `INVOCATION_COUNT_SLAB_SIZE: usize = 1`
- `invocation_count_from_array` / `invocation_count_to_array`

This allows typed IDs for primitive slab entries.

---

## 1.6 Array type generation

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

## 1.7 Naming conventions

| Generated Item | Naming Pattern | Example |
|---|---|---|
| ID type | `{TypeName}Id` | `DataId` |
| Array type | `{TypeName}Array` | `DataArray` |
| Slab size constant | `{TYPE_NAME}_SLAB_SIZE: usize` | `DATA_SLAB_SIZE` |
| From-array function | `{type_name}_from_array` | `data_from_array` |
| To-array function | `{type_name}_to_array` | `data_to_array` |

All generated function names use snake_case to ensure uniqueness when imported
via glob across modules.
