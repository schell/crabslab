//! Proc-macros for `crabslab`.
//!
//! Provides `#[slab_module]` and `#[slab_item]` attribute macros for
//! generating slab serialization code that works on both CPU (Rust) and
//! GPU (WGSL via `wgsl-rs`).

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::{parse_macro_input, spanned::Spanned};

// ---------------------------------------------------------------------------
// Attribute parsing for #[slab_module(...)]
// ---------------------------------------------------------------------------

/// Parsed attributes from `#[slab_module(...)]`.
///
/// Supported parameters (comma-separated):
///
/// - `wgsl_crate = <path>` — Override the path to the `wgsl_rs` crate. Default:
///   `wgsl_rs`. Used to emit `#[<path>::wgsl(...)]`.
///
/// - `wgsl(...)` — Forward parameters to `#[wgsl_rs::wgsl(...)]`. When present
///   (even with no inner args), the macro emits a `#[<wgsl_crate>::wgsl(...)]`
///   attribute on the output module. When absent, no `#[wgsl]` attribute is
///   emitted.
///
/// # Examples
///
/// ```ignore
/// #[slab_module]                                  // no #[wgsl] emitted
/// #[slab_module(wgsl(skip_validation))]           // emits #[wgsl_rs::wgsl(skip_validation)]
/// #[slab_module(wgsl())]                          // emits #[wgsl_rs::wgsl]
/// #[slab_module(wgsl_crate = my_crate, wgsl())]  // emits #[my_crate::wgsl]
/// ```
struct SlabModuleAttrs {
    /// Path to the `wgsl_rs` crate. Default: `wgsl_rs`.
    wgsl_crate: syn::Path,
    /// If `wgsl(...)` was present, the inner tokens to forward.
    /// `Some(empty)` means `wgsl()` — emit `#[wgsl]` with no args.
    /// `None` means no `wgsl` group — don't emit `#[wgsl]`.
    wgsl_args: Option<TokenStream2>,
}

impl Default for SlabModuleAttrs {
    fn default() -> Self {
        Self {
            wgsl_crate: syn::parse_quote!(wgsl_rs),
            wgsl_args: None,
        }
    }
}

impl syn::parse::Parse for SlabModuleAttrs {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let mut attrs = SlabModuleAttrs::default();

        while !input.is_empty() {
            let ident: syn::Ident = input.parse()?;

            if ident == "wgsl_crate" {
                input.parse::<syn::Token![=]>()?;
                attrs.wgsl_crate = input.parse()?;
            } else if ident == "wgsl" {
                let content;
                syn::parenthesized!(content in input);
                let inner: TokenStream2 = content.parse()?;
                attrs.wgsl_args = Some(inner);
            } else {
                return Err(syn::Error::new(
                    ident.span(),
                    format!(
                        "unknown #[slab_module] parameter '{ident}', expected 'wgsl_crate' or \
                         'wgsl'"
                    ),
                ));
            }

            // Consume optional trailing comma.
            if !input.is_empty() {
                input.parse::<syn::Token![,]>()?;
            }
        }

        Ok(attrs)
    }
}

// ---------------------------------------------------------------------------
// Public proc-macro entry points
// ---------------------------------------------------------------------------

/// Attribute macro that processes a module containing `#[slab_item]`-annotated
/// types.
///
/// For each `#[slab_item]` type, generates (all inside the module):
/// - An `impl` block with `SLAB_SIZE`, `from_array`, `to_array`
/// - A concrete ID type (`{TypeName}Id`)
/// - A concrete array type (`{TypeName}Array`)
/// - `impl crabslab::SlabItem` proxies for all generated types
///
/// Trait impls are emitted inside the module so that `#[wgsl]` can pass
/// them through to Rust without generating WGSL.
///
/// Auto-injects `use wgsl_rs::std::*;` if not already present.
///
/// When a `wgsl(...)` parameter group is present, the macro replaces
/// itself with `#[wgsl_rs::wgsl(...)]` on the output module, ensuring
/// that `#[wgsl]` always runs *after* companion types have been
/// generated. This eliminates the need to manually stack both
/// attributes and removes the macro ordering footgun.
///
/// See [`SlabModuleAttrs`] for the full list of parameters.
#[proc_macro_attribute]
pub fn slab_module(attr: TokenStream, item: TokenStream) -> TokenStream {
    let attrs = parse_macro_input!(attr as SlabModuleAttrs);
    let module = parse_macro_input!(item as syn::ItemMod);
    match process_module(module, &attrs) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
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

// ---------------------------------------------------------------------------
// Module processing
// ---------------------------------------------------------------------------

fn process_module(
    mut module: syn::ItemMod,
    macro_attrs: &SlabModuleAttrs,
) -> syn::Result<TokenStream2> {
    let mod_name = &module.ident;

    let Some((brace, ref items)) = module.content else {
        return Err(syn::Error::new(
            module.span(),
            "#[slab_module] requires an inline module (not `mod foo;`)",
        ));
    };

    let wgsl_crate = &macro_attrs.wgsl_crate;

    // Collect generated items (all inside the module).
    let mut new_items: Vec<syn::Item> = Vec::new();
    let mut has_wgsl_std_import = false;

    // Process each item in the module.
    let mut processed_items: Vec<syn::Item> = Vec::new();
    for item in items {
        // Check for `use <wgsl_crate>::std::*;`
        if is_wgsl_std_import(item, wgsl_crate) {
            has_wgsl_std_import = true;
        }

        if has_slab_item_attr(item) {
            match item {
                syn::Item::Struct(s) => {
                    let mut stripped = s.clone();
                    strip_slab_item_attr(&mut stripped.attrs);
                    let info = StructInfo::from_item_struct(&stripped)?;

                    // Keep the original struct (with #[slab_item] stripped).
                    processed_items.push(syn::Item::Struct(stripped));

                    // Generate in-module items (including trait impls).
                    new_items.extend(info.generate_in_module_items());
                }
                syn::Item::Enum(e) => {
                    let mut stripped = e.clone();
                    strip_slab_item_attr(&mut stripped.attrs);
                    let info = EnumInfo::from_item_enum(&stripped)?;

                    // Keep the original enum (with #[slab_item] stripped).
                    processed_items.push(syn::Item::Enum(stripped));

                    // Generate in-module items (including trait impls).
                    new_items.extend(info.generate_in_module_items());
                }
                other => {
                    return Err(syn::Error::new(
                        other.span(),
                        "#[slab_item] only supports structs and #[repr(u32)] enums",
                    ));
                }
            }
        } else {
            processed_items.push(item.clone());
        }
    }

    // Auto-inject `use <wgsl_crate>::std::*;` if missing.
    if !has_wgsl_std_import {
        let import: syn::Item = syn::parse_quote! {
            use #wgsl_crate::std::*;
        };
        processed_items.insert(0, import);
    }

    // Append generated items to the module body.
    processed_items.extend(new_items);

    // Expand slab_read! / slab_write! macro calls in all items.
    expand_slab_macros(&mut processed_items);

    // Rebuild the module with processed items.
    module.content = Some((brace, processed_items));

    // Emit the module (trait impls are now inside).
    // If `wgsl(...)` was specified, prepend `#[<wgsl_crate>::wgsl(...)]` so
    // that the compiler expands `#[wgsl]` in a subsequent pass — after all
    // companion types have been generated.
    let vis = &module.vis;
    let module_attrs = &module.attrs;
    let content = &module.content.as_ref().unwrap().1;

    let wgsl_attr = match &macro_attrs.wgsl_args {
        Some(args) if !args.is_empty() => {
            quote! { #[#wgsl_crate::wgsl(#args)] }
        }
        Some(_) => {
            quote! { #[#wgsl_crate::wgsl] }
        }
        None => {
            quote! {}
        }
    };

    Ok(quote! {
        #wgsl_attr
        #(#module_attrs)*
        #vis mod #mod_name {
            #(#content)*
        }
    })
}

// ---------------------------------------------------------------------------
// Struct analysis
// ---------------------------------------------------------------------------

struct StructInfo {
    /// The struct name (e.g., `Data`).
    name: syn::Ident,
    /// The ID type name (e.g., `DataId`).
    id_name: syn::Ident,
    /// The array type name (e.g., `DataArray`).
    array_name: syn::Ident,
    /// Field names and types (named structs only).
    fields: Vec<FieldInfo>,
}

struct FieldInfo {
    name: syn::Ident,
    ty: syn::Type,
}

impl StructInfo {
    fn from_item_struct(s: &syn::ItemStruct) -> syn::Result<Self> {
        let name = s.ident.clone();
        let id_name = format_ident!("{}Id", name);
        let array_name = format_ident!("{}Array", name);

        let fields = match &s.fields {
            syn::Fields::Named(named) => named
                .named
                .iter()
                .map(|f| {
                    Ok(FieldInfo {
                        name: f.ident.clone().unwrap(),
                        ty: f.ty.clone(),
                    })
                })
                .collect::<syn::Result<Vec<_>>>()?,
            _ => {
                return Err(syn::Error::new(
                    s.span(),
                    "#[slab_item] only supports named structs",
                ));
            }
        };

        Ok(StructInfo {
            name,
            id_name,
            array_name,
            fields,
        })
    }

    /// Generate all in-module items for this struct.
    fn generate_in_module_items(&self) -> Vec<syn::Item> {
        let mut items = Vec::new();
        items.push(self.generate_struct_impl());
        items.extend(generate_id_type(&self.id_name));
        items.extend(generate_array_type(
            &self.name,
            &self.id_name,
            &self.array_name,
        ));
        items.extend(generate_trait_impls(
            &self.name,
            &self.id_name,
            &self.array_name,
        ));
        items
    }

    /// Generate `impl Data { SLAB_SIZE, from_array, to_array }`.
    fn generate_struct_impl(&self) -> syn::Item {
        let name = &self.name;

        // SLAB_SIZE = sum of field slab sizes.
        let slab_size_terms: Vec<TokenStream2> = self
            .fields
            .iter()
            .map(|f| field_slab_size_expr(&f.ty))
            .collect();

        // Build cumulative offset expressions for each field.
        let offsets = cumulative_offsets(&self.fields);

        // from_array: read each field from the u32 array.
        // For nested types, we generate pre-computation statements
        // (sub-array reads) before the struct literal, since WGSL does
        // not support block expressions as struct field initializers.
        let mut from_array_preamble: Vec<TokenStream2> = Vec::new();
        let mut from_array_fields: Vec<TokenStream2> = Vec::new();
        for (f, offset) in self.fields.iter().zip(offsets.iter()) {
            let fname = &f.name;
            let (pre_stmts, read_expr) = field_from_array_parts(&f.ty, offset, fname);
            from_array_preamble.extend(pre_stmts);
            from_array_fields.push(quote! { #fname: #read_expr });
        }

        // to_array: write each field into the u32 array.
        let to_array_stmts: Vec<TokenStream2> = self
            .fields
            .iter()
            .zip(offsets.iter())
            .map(|(f, offset)| field_to_array_stmt(&f.name, &f.ty, offset))
            .collect();

        syn::parse_quote! {
            impl #name {
                pub const SLAB_SIZE: usize = 0usize #(+ #slab_size_terms)*;

                pub fn from_array(u32s: [u32; #name::SLAB_SIZE]) -> #name {
                    #(#from_array_preamble)*
                    #name {
                        #(#from_array_fields,)*
                    }
                }

                pub fn to_array(d: #name) -> [u32; #name::SLAB_SIZE] {
                    let mut arr = [0u32; #name::SLAB_SIZE];
                    #(#to_array_stmts)*
                    arr
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Enum analysis
// ---------------------------------------------------------------------------

struct EnumInfo {
    /// The enum name (e.g., `DataChangeTy`).
    name: syn::Ident,
    /// The ID type name (e.g., `DataChangeTyId`).
    id_name: syn::Ident,
    /// The array type name (e.g., `DataChangeTyArray`).
    array_name: syn::Ident,
    /// Variant names and their explicit discriminant values.
    variants: Vec<VariantInfo>,
}

struct VariantInfo {
    name: syn::Ident,
    discriminant: u32,
}

impl EnumInfo {
    fn from_item_enum(e: &syn::ItemEnum) -> syn::Result<Self> {
        // Require #[repr(u32)].
        let has_repr_u32 = e.attrs.iter().any(|a| {
            if !a.path().is_ident("repr") {
                return false;
            }
            let Ok(inner) = a.parse_args::<syn::Ident>() else {
                return false;
            };
            inner == "u32"
        });
        if !has_repr_u32 {
            return Err(syn::Error::new(
                e.span(),
                "#[slab_item] enums must have #[repr(u32)]",
            ));
        }

        let name = e.ident.clone();
        let id_name = format_ident!("{}Id", name);
        let array_name = format_ident!("{}Array", name);

        let mut variants = Vec::new();
        let mut next_discriminant = 0u32;

        for v in &e.variants {
            // Only unit variants are supported.
            if !matches!(v.fields, syn::Fields::Unit) {
                return Err(syn::Error::new(
                    v.span(),
                    "#[slab_item] enums only support unit variants (no data)",
                ));
            }

            let disc = if let Some((_, expr)) = &v.discriminant {
                // Parse the explicit discriminant.
                parse_u32_expr(expr)?
            } else {
                next_discriminant
            };

            variants.push(VariantInfo {
                name: v.ident.clone(),
                discriminant: disc,
            });
            next_discriminant = disc + 1;
        }

        if variants.is_empty() {
            return Err(syn::Error::new(
                e.span(),
                "#[slab_item] enums must have at least one variant",
            ));
        }

        Ok(EnumInfo {
            name,
            id_name,
            array_name,
            variants,
        })
    }

    /// Generate all in-module items for this enum.
    fn generate_in_module_items(&self) -> Vec<syn::Item> {
        let mut items = Vec::new();
        items.push(self.generate_enum_impl());
        items.extend(generate_id_type(&self.id_name));
        items.extend(generate_array_type(
            &self.name,
            &self.id_name,
            &self.array_name,
        ));
        items.extend(generate_trait_impls(
            &self.name,
            &self.id_name,
            &self.array_name,
        ));
        items
    }

    /// Generate `impl EnumName { SLAB_SIZE, from_array, to_array }`.
    fn generate_enum_impl(&self) -> syn::Item {
        let name = &self.name;

        // Build match arms for from_array (u32 -> enum).
        // Use block-form bodies with assignment because WGSL `switch` is a
        // statement, not an expression.
        let from_arms: Vec<TokenStream2> = self
            .variants
            .iter()
            .map(|v| {
                let vname = &v.name;
                let disc = v.discriminant;
                quote! { #disc => { result = #name::#vname; } }
            })
            .collect();

        // Use the first variant as the default (fallback for unknown values).
        let default_variant = &self.variants[0].name;

        // Build match arms for to_array (enum -> u32).
        // Assign inside each arm body because WGSL `switch` is a statement,
        // not an expression — it cannot appear in `let` bindings.
        let to_arms: Vec<TokenStream2> = self
            .variants
            .iter()
            .map(|v| {
                let vname = &v.name;
                let disc = v.discriminant;
                quote! { #name::#vname => { v = #disc; } }
            })
            .collect();

        syn::parse_quote! {
            impl #name {
                pub const SLAB_SIZE: usize = 1usize;

                pub fn from_array(u32s: [u32; 1usize]) -> #name {
                    let mut result: #name;
                    match u32s[0usize] {
                        #(#from_arms,)*
                        _ => { result = #name::#default_variant; },
                    }
                    result
                }

                pub fn to_array(d: #name) -> [u32; 1usize] {
                    let mut v: u32 = 0u32;
                    match d {
                        #(#to_arms,)*
                        _ => {},
                    }
                    [v]
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Shared codegen (used by both StructInfo and EnumInfo)
// ---------------------------------------------------------------------------

/// Generate the ID struct and its impl block for a `#[slab_item]` type.
fn generate_id_type(id_name: &syn::Ident) -> Vec<syn::Item> {
    let id_struct: syn::Item = syn::parse_quote! {
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
        pub struct #id_name {
            pub inner: u32,
        }
    };

    let id_impl: syn::Item = syn::parse_quote! {
        impl #id_name {
            pub const NONE: #id_name = #id_name { inner: 4294967295u32 };
            pub const ZERO: #id_name = #id_name { inner: 0u32 };

            pub const SLAB_SIZE: usize = 1usize;

            pub fn new(index: u32) -> #id_name {
                #id_name { inner: index }
            }

            pub fn is_none(id: #id_name) -> bool {
                id.inner == 4294967295u32
            }

            pub fn from_array(u32s: [u32; #id_name::SLAB_SIZE]) -> #id_name {
                #id_name { inner: u32s[0usize] }
            }

            pub fn to_array(d: #id_name) -> [u32; #id_name::SLAB_SIZE] {
                [d.inner]
            }
        }
    };

    vec![id_struct, id_impl]
}

/// Generate the array struct and its impl block for a `#[slab_item]` type.
fn generate_array_type(
    type_name: &syn::Ident,
    id_name: &syn::Ident,
    array_name: &syn::Ident,
) -> Vec<syn::Item> {
    let array_struct: syn::Item = syn::parse_quote! {
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
        pub struct #array_name {
            pub id: #id_name,
            pub len: u32,
        }
    };

    let array_impl: syn::Item = syn::parse_quote! {
        impl #array_name {
            pub const NONE: #array_name = #array_name {
                id: #id_name::NONE,
                len: 0u32,
            };

            pub const SLAB_SIZE: usize = #id_name::SLAB_SIZE + 1usize;

            pub fn at(arr: #array_name, index: u32) -> #id_name {
                if index >= arr.len {
                    #id_name::NONE
                } else {
                    #id_name::new(arr.id.inner + #type_name::SLAB_SIZE as u32 * index)
                }
            }

            pub fn from_array(
                u32s: [u32; #array_name::SLAB_SIZE],
            ) -> #array_name {
                #array_name {
                    id: #id_name { inner: u32s[0usize] },
                    len: u32s[0usize + #id_name::SLAB_SIZE],
                }
            }

            pub fn to_array(
                d: #array_name,
            ) -> [u32; #array_name::SLAB_SIZE] {
                let mut arr = [0u32; #array_name::SLAB_SIZE];
                arr[0usize] = d.id.inner;
                arr[0usize + #id_name::SLAB_SIZE] = d.len;
                arr
            }
        }
    };

    vec![array_struct, array_impl]
}

/// Generate CPU-side `impl crabslab::SlabItem` proxies for a type, its
/// ID type, and its array type.
fn generate_trait_impls(
    type_name: &syn::Ident,
    id_name: &syn::Ident,
    array_name: &syn::Ident,
) -> Vec<syn::Item> {
    vec![
        generate_slab_item_proxy(type_name),
        generate_slab_item_proxy(id_name),
        generate_slab_item_proxy(array_name),
    ]
}

/// Parse a simple integer literal expression to a `u32`.
fn parse_u32_expr(expr: &syn::Expr) -> syn::Result<u32> {
    match expr {
        syn::Expr::Lit(syn::ExprLit {
            lit: syn::Lit::Int(lit_int),
            ..
        }) => lit_int.base10_parse::<u32>().map_err(|e| {
            syn::Error::new(
                lit_int.span(),
                format!("enum discriminant must be a u32 literal: {e}"),
            )
        }),
        _ => Err(syn::Error::new(
            expr.span(),
            "enum discriminant must be a simple integer literal",
        )),
    }
}

/// Generate a `impl crabslab::SlabItem for Type` proxy that delegates
/// to the inherent `Type::SLAB_SIZE`, `Type::from_array`, `Type::to_array`.
///
/// Emitted inside the module so that `#[wgsl]` can pass it through to
/// Rust without generating WGSL.
fn generate_slab_item_proxy(type_name: &syn::Ident) -> syn::Item {
    syn::parse_quote! {
        impl crabslab::SlabItem for #type_name {
            const SLAB_SIZE: usize = #type_name::SLAB_SIZE;
            type Array = [u32; #type_name::SLAB_SIZE];

            fn to_array(&self) -> Self::Array {
                #type_name::to_array(*self)
            }

            fn from_array(arr: Self::Array) -> Self {
                #type_name::from_array(arr)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Field code generation
// ---------------------------------------------------------------------------

/// Return a `TokenStream2` expression for the slab size of a field type.
/// Primitives and wgsl-rs vector/matrix types are inlined as literals;
/// other types use `Type::SLAB_SIZE`.
fn field_slab_size_expr(ty: &syn::Type) -> TokenStream2 {
    match type_name_str(ty).as_deref() {
        Some("u32" | "i32" | "f32" | "bool") => quote! { 1usize },
        // wgsl-rs vector types
        Some("Vec2f" | "Vec2i" | "Vec2u" | "Vec2b") => quote! { 2usize },
        Some("Vec3f" | "Vec3i" | "Vec3u" | "Vec3b") => quote! { 3usize },
        Some("Vec4f" | "Vec4i" | "Vec4u" | "Vec4b") => quote! { 4usize },
        // wgsl-rs matrix types
        Some("Mat2x2f" | "Mat2f") => quote! { 4usize },
        Some("Mat2x3f") => quote! { 6usize },
        Some("Mat2x4f") => quote! { 8usize },
        Some("Mat3x2f") => quote! { 6usize },
        Some("Mat3x3f" | "Mat3f") => quote! { 9usize },
        Some("Mat3x4f") => quote! { 12usize },
        Some("Mat4x2f") => quote! { 8usize },
        Some("Mat4x3f") => quote! { 12usize },
        Some("Mat4x4f" | "Mat4f") => quote! { 16usize },
        _ => quote! { #ty::SLAB_SIZE },
    }
}

/// Return pre-computation statements and a value expression to read a
/// field from a `u32s` array at the given offset.
///
/// For primitive types the pre-statements are empty and the expression
/// reads directly from `u32s`. For wgsl-rs vector/matrix types the
/// pre-statements read components and construct the value. For nested
/// `#[slab_item]` types the pre-statements read a sub-array into a
/// local variable so that the struct literal field can use a simple
/// expression (WGSL does not support block expressions as struct field
/// initializers).
fn field_from_array_parts(
    ty: &syn::Type,
    offset: &TokenStream2,
    field_name: &syn::Ident,
) -> (Vec<TokenStream2>, TokenStream2) {
    match type_name_str(ty).as_deref() {
        Some("u32") => (vec![], quote! { u32s[#offset] }),
        Some("f32") => (vec![], quote! { bitcast_f32(u32s[#offset]) }),
        Some("i32") => (vec![], quote! { bitcast_i32(u32s[#offset]) }),
        Some("bool") => (vec![], quote! { u32s[#offset] != 0u32 }),

        // -- wgsl-rs Vec2 types -------------------------------------------
        Some("Vec2f") => {
            let val = format_ident!("val_{}", field_name);
            let pre = vec![quote! {
                let #val = vec2f(
                    bitcast_f32(u32s[#offset]),
                    bitcast_f32(u32s[#offset + 1usize]),
                );
            }];
            (pre, quote! { #val })
        }
        Some("Vec2i") => {
            let val = format_ident!("val_{}", field_name);
            let pre = vec![quote! {
                let #val = vec2i(
                    bitcast_i32(u32s[#offset]),
                    bitcast_i32(u32s[#offset + 1usize]),
                );
            }];
            (pre, quote! { #val })
        }
        Some("Vec2u") => {
            let val = format_ident!("val_{}", field_name);
            let pre = vec![quote! {
                let #val = vec2u(
                    u32s[#offset],
                    u32s[#offset + 1usize],
                );
            }];
            (pre, quote! { #val })
        }
        Some("Vec2b") => {
            let val = format_ident!("val_{}", field_name);
            let pre = vec![quote! {
                let #val = vec2b(
                    u32s[#offset] != 0u32,
                    u32s[#offset + 1usize] != 0u32,
                );
            }];
            (pre, quote! { #val })
        }

        // -- wgsl-rs Vec3 types -------------------------------------------
        Some("Vec3f") => {
            let val = format_ident!("val_{}", field_name);
            let pre = vec![quote! {
                let #val = vec3f(
                    bitcast_f32(u32s[#offset]),
                    bitcast_f32(u32s[#offset + 1usize]),
                    bitcast_f32(u32s[#offset + 2usize]),
                );
            }];
            (pre, quote! { #val })
        }
        Some("Vec3i") => {
            let val = format_ident!("val_{}", field_name);
            let pre = vec![quote! {
                let #val = vec3i(
                    bitcast_i32(u32s[#offset]),
                    bitcast_i32(u32s[#offset + 1usize]),
                    bitcast_i32(u32s[#offset + 2usize]),
                );
            }];
            (pre, quote! { #val })
        }
        Some("Vec3u") => {
            let val = format_ident!("val_{}", field_name);
            let pre = vec![quote! {
                let #val = vec3u(
                    u32s[#offset],
                    u32s[#offset + 1usize],
                    u32s[#offset + 2usize],
                );
            }];
            (pre, quote! { #val })
        }
        Some("Vec3b") => {
            let val = format_ident!("val_{}", field_name);
            let pre = vec![quote! {
                let #val = vec3b(
                    u32s[#offset] != 0u32,
                    u32s[#offset + 1usize] != 0u32,
                    u32s[#offset + 2usize] != 0u32,
                );
            }];
            (pre, quote! { #val })
        }

        // -- wgsl-rs Vec4 types -------------------------------------------
        Some("Vec4f") => {
            let val = format_ident!("val_{}", field_name);
            let pre = vec![quote! {
                let #val = vec4f(
                    bitcast_f32(u32s[#offset]),
                    bitcast_f32(u32s[#offset + 1usize]),
                    bitcast_f32(u32s[#offset + 2usize]),
                    bitcast_f32(u32s[#offset + 3usize]),
                );
            }];
            (pre, quote! { #val })
        }
        Some("Vec4i") => {
            let val = format_ident!("val_{}", field_name);
            let pre = vec![quote! {
                let #val = vec4i(
                    bitcast_i32(u32s[#offset]),
                    bitcast_i32(u32s[#offset + 1usize]),
                    bitcast_i32(u32s[#offset + 2usize]),
                    bitcast_i32(u32s[#offset + 3usize]),
                );
            }];
            (pre, quote! { #val })
        }
        Some("Vec4u") => {
            let val = format_ident!("val_{}", field_name);
            let pre = vec![quote! {
                let #val = vec4u(
                    u32s[#offset],
                    u32s[#offset + 1usize],
                    u32s[#offset + 2usize],
                    u32s[#offset + 3usize],
                );
            }];
            (pre, quote! { #val })
        }
        Some("Vec4b") => {
            let val = format_ident!("val_{}", field_name);
            let pre = vec![quote! {
                let #val = vec4b(
                    u32s[#offset] != 0u32,
                    u32s[#offset + 1usize] != 0u32,
                    u32s[#offset + 2usize] != 0u32,
                    u32s[#offset + 3usize] != 0u32,
                );
            }];
            (pre, quote! { #val })
        }

        // -- wgsl-rs matrix types -----------------------------------------
        Some(name @ ("Mat2x2f" | "Mat2f")) => {
            vec_matrix_from_array_parts(name, field_name, offset, 2, 2)
        }
        Some(name @ "Mat2x3f") => vec_matrix_from_array_parts(name, field_name, offset, 2, 3),
        Some(name @ "Mat2x4f") => vec_matrix_from_array_parts(name, field_name, offset, 2, 4),
        Some(name @ "Mat3x2f") => vec_matrix_from_array_parts(name, field_name, offset, 3, 2),
        Some(name @ ("Mat3x3f" | "Mat3f")) => {
            vec_matrix_from_array_parts(name, field_name, offset, 3, 3)
        }
        Some(name @ "Mat3x4f") => vec_matrix_from_array_parts(name, field_name, offset, 3, 4),
        Some(name @ "Mat4x2f") => vec_matrix_from_array_parts(name, field_name, offset, 4, 2),
        Some(name @ "Mat4x3f") => vec_matrix_from_array_parts(name, field_name, offset, 4, 3),
        Some(name @ ("Mat4x4f" | "Mat4f")) => {
            vec_matrix_from_array_parts(name, field_name, offset, 4, 4)
        }

        _ => {
            // Nested slab_item type: generate pre-statements to read the
            // sub-array, then use the local variable in the struct literal.
            let sub_ident = format_ident!("sub_{}", field_name);
            let val_ident = format_ident!("val_{}", field_name);
            let pre = vec![
                quote! { let mut #sub_ident = [0u32; #ty::SLAB_SIZE]; },
                quote! { slab_read_array!(u32s, #offset, #sub_ident, #ty::SLAB_SIZE); },
                quote! { let #val_ident = #ty::from_array(#sub_ident); },
            ];
            (pre, quote! { #val_ident })
        }
    }
}

/// Generate pre-computation statements and a value expression to read a
/// wgsl-rs matrix field from a `u32s` array.
///
/// Matrices are column-major. Each column is a `VecRf` with `rows`
/// components. The matrix has `cols` columns.
fn vec_matrix_from_array_parts(
    _mat_name: &str,
    field_name: &syn::Ident,
    offset: &TokenStream2,
    cols: usize,
    rows: usize,
) -> (Vec<TokenStream2>, TokenStream2) {
    let val = format_ident!("val_{}", field_name);

    // Generate column read expressions.
    let col_exprs: Vec<TokenStream2> = (0..cols)
        .map(|c| {
            let col_offset = c * rows;
            vec_read_expr(rows, offset, col_offset)
        })
        .collect();

    // Pick the correct constructor name (e.g., mat3x4f).
    let ctor = format_ident!("mat{}x{}f", cols, rows);

    let pre = vec![quote! {
        let #val = #ctor(#(#col_exprs),*);
    }];

    (pre, quote! { #val })
}

/// Generate a token stream expression that reads a `VecNf` from `u32s`
/// at `base_offset + component_offset`.
fn vec_read_expr(
    components: usize,
    base_offset: &TokenStream2,
    component_offset: usize,
) -> TokenStream2 {
    match components {
        2 => {
            let o0 = component_offset;
            let o1 = component_offset + 1;
            quote! {
                vec2f(
                    bitcast_f32(u32s[#base_offset + #o0]),
                    bitcast_f32(u32s[#base_offset + #o1]),
                )
            }
        }
        3 => {
            let o0 = component_offset;
            let o1 = component_offset + 1;
            let o2 = component_offset + 2;
            quote! {
                vec3f(
                    bitcast_f32(u32s[#base_offset + #o0]),
                    bitcast_f32(u32s[#base_offset + #o1]),
                    bitcast_f32(u32s[#base_offset + #o2]),
                )
            }
        }
        4 => {
            let o0 = component_offset;
            let o1 = component_offset + 1;
            let o2 = component_offset + 2;
            let o3 = component_offset + 3;
            quote! {
                vec4f(
                    bitcast_f32(u32s[#base_offset + #o0]),
                    bitcast_f32(u32s[#base_offset + #o1]),
                    bitcast_f32(u32s[#base_offset + #o2]),
                    bitcast_f32(u32s[#base_offset + #o3]),
                )
            }
        }
        _ => unreachable!("invalid vector component count: {components}"),
    }
}

/// Return a `TokenStream2` statement that writes a field into an `arr` array
/// at the given offset expression. The field is accessed as `d.{field_name}`.
fn field_to_array_stmt(
    field_name: &syn::Ident,
    ty: &syn::Type,
    offset: &TokenStream2,
) -> TokenStream2 {
    match type_name_str(ty).as_deref() {
        Some("u32") => quote! { arr[#offset] = d.#field_name; },
        Some("f32") => quote! { arr[#offset] = bitcast_u32(d.#field_name); },
        Some("i32") => quote! { arr[#offset] = bitcast_u32(d.#field_name); },
        Some("bool") => {
            quote! { arr[#offset] = if d.#field_name { 1u32 } else { 0u32 }; }
        }

        // -- wgsl-rs Vec2 types -------------------------------------------
        Some("Vec2f") => quote! {
            arr[#offset] = bitcast_u32(d.#field_name.x);
            arr[#offset + 1usize] = bitcast_u32(d.#field_name.y);
        },
        Some("Vec2i") => quote! {
            arr[#offset] = bitcast_u32(d.#field_name.x);
            arr[#offset + 1usize] = bitcast_u32(d.#field_name.y);
        },
        Some("Vec2u") => quote! {
            arr[#offset] = d.#field_name.x;
            arr[#offset + 1usize] = d.#field_name.y;
        },
        Some("Vec2b") => quote! {
            arr[#offset] = if d.#field_name.x { 1u32 } else { 0u32 };
            arr[#offset + 1usize] = if d.#field_name.y { 1u32 } else { 0u32 };
        },

        // -- wgsl-rs Vec3 types -------------------------------------------
        Some("Vec3f") => quote! {
            arr[#offset] = bitcast_u32(d.#field_name.x);
            arr[#offset + 1usize] = bitcast_u32(d.#field_name.y);
            arr[#offset + 2usize] = bitcast_u32(d.#field_name.z);
        },
        Some("Vec3i") => quote! {
            arr[#offset] = bitcast_u32(d.#field_name.x);
            arr[#offset + 1usize] = bitcast_u32(d.#field_name.y);
            arr[#offset + 2usize] = bitcast_u32(d.#field_name.z);
        },
        Some("Vec3u") => quote! {
            arr[#offset] = d.#field_name.x;
            arr[#offset + 1usize] = d.#field_name.y;
            arr[#offset + 2usize] = d.#field_name.z;
        },
        Some("Vec3b") => quote! {
            arr[#offset] = if d.#field_name.x { 1u32 } else { 0u32 };
            arr[#offset + 1usize] = if d.#field_name.y { 1u32 } else { 0u32 };
            arr[#offset + 2usize] = if d.#field_name.z { 1u32 } else { 0u32 };
        },

        // -- wgsl-rs Vec4 types -------------------------------------------
        Some("Vec4f") => quote! {
            arr[#offset] = bitcast_u32(d.#field_name.x);
            arr[#offset + 1usize] = bitcast_u32(d.#field_name.y);
            arr[#offset + 2usize] = bitcast_u32(d.#field_name.z);
            arr[#offset + 3usize] = bitcast_u32(d.#field_name.w);
        },
        Some("Vec4i") => quote! {
            arr[#offset] = bitcast_u32(d.#field_name.x);
            arr[#offset + 1usize] = bitcast_u32(d.#field_name.y);
            arr[#offset + 2usize] = bitcast_u32(d.#field_name.z);
            arr[#offset + 3usize] = bitcast_u32(d.#field_name.w);
        },
        Some("Vec4u") => quote! {
            arr[#offset] = d.#field_name.x;
            arr[#offset + 1usize] = d.#field_name.y;
            arr[#offset + 2usize] = d.#field_name.z;
            arr[#offset + 3usize] = d.#field_name.w;
        },
        Some("Vec4b") => quote! {
            arr[#offset] = if d.#field_name.x { 1u32 } else { 0u32 };
            arr[#offset + 1usize] = if d.#field_name.y { 1u32 } else { 0u32 };
            arr[#offset + 2usize] = if d.#field_name.z { 1u32 } else { 0u32 };
            arr[#offset + 3usize] = if d.#field_name.w { 1u32 } else { 0u32 };
        },

        // -- wgsl-rs matrix types -----------------------------------------
        Some("Mat2x2f" | "Mat2f") => vec_matrix_to_array_stmt(field_name, offset, 2, 2),
        Some("Mat2x3f") => vec_matrix_to_array_stmt(field_name, offset, 2, 3),
        Some("Mat2x4f") => vec_matrix_to_array_stmt(field_name, offset, 2, 4),
        Some("Mat3x2f") => vec_matrix_to_array_stmt(field_name, offset, 3, 2),
        Some("Mat3x3f" | "Mat3f") => vec_matrix_to_array_stmt(field_name, offset, 3, 3),
        Some("Mat3x4f") => vec_matrix_to_array_stmt(field_name, offset, 3, 4),
        Some("Mat4x2f") => vec_matrix_to_array_stmt(field_name, offset, 4, 2),
        Some("Mat4x3f") => vec_matrix_to_array_stmt(field_name, offset, 4, 3),
        Some("Mat4x4f" | "Mat4f") => vec_matrix_to_array_stmt(field_name, offset, 4, 4),

        _ => {
            // Nested slab_item type: copy sub-array via slab_write_array!.
            quote! {
                {
                    let inner = #ty::to_array(d.#field_name);
                    slab_write_array!(arr, #offset, inner, #ty::SLAB_SIZE);
                }
            }
        }
    }
}

/// Generate statements to write a wgsl-rs matrix field into an `arr` array.
///
/// Matrices are column-major. Each column is accessed via `d.field[col_idx]`
/// and each component is bitcast to u32.
fn vec_matrix_to_array_stmt(
    field_name: &syn::Ident,
    offset: &TokenStream2,
    cols: usize,
    rows: usize,
) -> TokenStream2 {
    let mut stmts = Vec::new();

    for c in 0..cols {
        let col_base = c * rows;
        let col_idx = c;
        match rows {
            2 => {
                let o0 = col_base;
                let o1 = col_base + 1;
                stmts.push(quote! {
                    arr[#offset + #o0] = bitcast_u32(d.#field_name[#col_idx].x);
                    arr[#offset + #o1] = bitcast_u32(d.#field_name[#col_idx].y);
                });
            }
            3 => {
                let o0 = col_base;
                let o1 = col_base + 1;
                let o2 = col_base + 2;
                stmts.push(quote! {
                    arr[#offset + #o0] = bitcast_u32(d.#field_name[#col_idx].x);
                    arr[#offset + #o1] = bitcast_u32(d.#field_name[#col_idx].y);
                    arr[#offset + #o2] = bitcast_u32(d.#field_name[#col_idx].z);
                });
            }
            4 => {
                let o0 = col_base;
                let o1 = col_base + 1;
                let o2 = col_base + 2;
                let o3 = col_base + 3;
                stmts.push(quote! {
                    arr[#offset + #o0] = bitcast_u32(d.#field_name[#col_idx].x);
                    arr[#offset + #o1] = bitcast_u32(d.#field_name[#col_idx].y);
                    arr[#offset + #o2] = bitcast_u32(d.#field_name[#col_idx].z);
                    arr[#offset + #o3] = bitcast_u32(d.#field_name[#col_idx].w);
                });
            }
            _ => unreachable!("invalid matrix row count: {rows}"),
        }
    }

    quote! { #(#stmts)* }
}

/// Build cumulative offset expressions for each field in a struct.
///
/// The offset for field `i` is `0usize + field0_size + field1_size + ...`
/// where `field_size` is `1usize` for primitives or `Type::SLAB_SIZE` for
/// nested types.
fn cumulative_offsets(fields: &[FieldInfo]) -> Vec<TokenStream2> {
    let mut offsets = Vec::with_capacity(fields.len());
    let mut terms: Vec<TokenStream2> = vec![quote! { 0usize }];

    for (i, field) in fields.iter().enumerate() {
        // The offset for this field is the sum of all preceding terms.
        offsets.push(quote! { #(#terms)+* });

        // Add this field's size to the running sum for the next field.
        let size = field_slab_size_expr(&field.ty);
        terms.push(size);

        let _ = i; // suppress unused warning
    }

    offsets
}

// ---------------------------------------------------------------------------
// Utility helpers
// ---------------------------------------------------------------------------

/// Check if an item has a `#[slab_item]` attribute.
fn has_slab_item_attr(item: &syn::Item) -> bool {
    let attrs = match item {
        syn::Item::Struct(s) => &s.attrs,
        syn::Item::Enum(e) => &e.attrs,
        _ => return false,
    };
    attrs.iter().any(|a| a.path().is_ident("slab_item"))
}

/// Strip `#[slab_item]` attributes from a list of attributes.
fn strip_slab_item_attr(attrs: &mut Vec<syn::Attribute>) {
    attrs.retain(|a| !a.path().is_ident("slab_item"));
}

/// Check if an item is `use <wgsl_crate>::std::*;`.
fn is_wgsl_std_import(item: &syn::Item, wgsl_crate: &syn::Path) -> bool {
    let syn::Item::Use(use_item) = item else {
        return false;
    };
    let tokens = quote! { #use_item };
    let s = tokens.to_string();
    let crate_str = quote! { #wgsl_crate }.to_string();
    s.contains(&format!("{crate_str} :: std"))
}

/// Extract a simple type name string from a `syn::Type`, if it's a plain
/// path like `u32`, `f32`, `DataChange`, etc. Returns `None` for complex
/// types.
fn type_name_str(ty: &syn::Type) -> Option<String> {
    let syn::Type::Path(tp) = ty else {
        return None;
    };
    if tp.qself.is_some() {
        return None;
    }
    let seg = tp.path.segments.last()?;
    if seg.arguments.is_none() || matches!(seg.arguments, syn::PathArguments::None) {
        Some(seg.ident.to_string())
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// slab_read! / slab_write! expansion
// ---------------------------------------------------------------------------

/// Visitor that expands `slab_read!(Type, slab, offset)` and
/// `slab_write!(Type, slab, offset, value)` macro calls into their
/// multi-statement equivalents.
///
/// The expansion happens in-place via `syn::visit_mut`, before the module
/// is emitted with `#[wgsl]`. This means `#[wgsl]` only sees
/// `slab_read_array!`/`slab_write_array!` calls which it already supports.
///
/// Because WGSL does not support block expressions, the expansion works at
/// the **statement list** level rather than the expression level.
///
/// For `let x = slab_read!(Type, slab, offset);` the single statement is
/// replaced by three statements:
///
/// ```ignore
/// let mut slab_read_arr = [0u32; Type::SLAB_SIZE];
/// slab_read_array!(slab, offset, slab_read_arr, Type::SLAB_SIZE);
/// let x = Type::from_array(slab_read_arr);
/// ```
///
/// For `slab_write!(Type, slab, offset, value);` the single statement is
/// replaced by two statements:
///
/// ```ignore
/// let slab_write_arr = Type::to_array(value);
/// slab_write_array!(slab, offset, slab_write_arr, Type::SLAB_SIZE);
/// ```
struct SlabMacroExpander {
    /// Counter to generate unique variable names across multiple expansions
    /// in the same block.
    counter: usize,
}

impl SlabMacroExpander {
    fn new() -> Self {
        Self { counter: 0 }
    }

    /// Generate a unique identifier for temporary variables.
    fn unique_ident(&mut self, prefix: &str) -> syn::Ident {
        let id = format_ident!("{}_{}", prefix, self.counter);
        self.counter += 1;
        id
    }

    /// Check if a macro path is `slab_read` (possibly qualified).
    fn is_slab_read(mac: &syn::Macro) -> bool {
        mac.path.is_ident("slab_read")
    }

    /// Check if a macro path is `slab_write` (possibly qualified).
    fn is_slab_write(mac: &syn::Macro) -> bool {
        mac.path.is_ident("slab_write")
    }
}

/// Parsed arguments for `slab_read!(Type, slab, offset)`.
struct SlabReadArgs {
    ty: syn::Type,
    slab: syn::Expr,
    offset: syn::Expr,
}

impl syn::parse::Parse for SlabReadArgs {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let ty: syn::Type = input.parse()?;
        input.parse::<syn::Token![,]>()?;
        let slab: syn::Expr = input.parse()?;
        input.parse::<syn::Token![,]>()?;
        let offset: syn::Expr = input.parse()?;
        // Allow optional trailing comma.
        let _ = input.parse::<syn::Token![,]>();
        Ok(SlabReadArgs { ty, slab, offset })
    }
}

/// Parsed arguments for `slab_write!(Type, slab, offset, value)`.
struct SlabWriteArgs {
    ty: syn::Type,
    slab: syn::Expr,
    offset: syn::Expr,
    value: syn::Expr,
}

impl syn::parse::Parse for SlabWriteArgs {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let ty: syn::Type = input.parse()?;
        input.parse::<syn::Token![,]>()?;
        let slab: syn::Expr = input.parse()?;
        input.parse::<syn::Token![,]>()?;
        let offset: syn::Expr = input.parse()?;
        input.parse::<syn::Token![,]>()?;
        let value: syn::Expr = input.parse()?;
        // Allow optional trailing comma.
        let _ = input.parse::<syn::Token![,]>();
        Ok(SlabWriteArgs {
            ty,
            slab,
            offset,
            value,
        })
    }
}

impl syn::visit_mut::VisitMut for SlabMacroExpander {
    fn visit_block_mut(&mut self, block: &mut syn::Block) {
        // First recurse into nested blocks/expressions.
        syn::visit_mut::visit_block_mut(self, block);

        // Then scan the statement list for slab_read!/slab_write! and
        // expand each into multiple statements.
        let mut new_stmts: Vec<syn::Stmt> = Vec::with_capacity(block.stmts.len());
        for stmt in block.stmts.drain(..) {
            match &stmt {
                // Case 1: `let <pat> = slab_read!(Type, slab, offset);`
                syn::Stmt::Local(local) if Self::is_let_slab_read(local) => {
                    if let Some(expanded) = self.expand_let_slab_read(local) {
                        new_stmts.extend(expanded);
                    } else {
                        new_stmts.push(stmt);
                    }
                }
                // Case 2: `slab_write!(Type, slab, offset, value);`
                syn::Stmt::Macro(stmt_macro) if Self::is_slab_write(&stmt_macro.mac) => {
                    if let Some(expanded) = self.expand_stmt_slab_write(stmt_macro) {
                        new_stmts.extend(expanded);
                    } else {
                        new_stmts.push(stmt);
                    }
                }
                // Case 3: `slab_read!(Type, slab, offset);` in statement
                // position (result discarded — unusual but allowed).
                syn::Stmt::Macro(stmt_macro) if Self::is_slab_read(&stmt_macro.mac) => {
                    if let Some(expanded) = self.expand_stmt_slab_read(stmt_macro) {
                        new_stmts.extend(expanded);
                    } else {
                        new_stmts.push(stmt);
                    }
                }
                _ => new_stmts.push(stmt),
            }
        }
        block.stmts = new_stmts;
    }
}

impl SlabMacroExpander {
    /// Check if a `let` statement's initializer is a `slab_read!()` call.
    fn is_let_slab_read(local: &syn::Local) -> bool {
        let Some(init) = &local.init else {
            return false;
        };
        matches!(&*init.expr, syn::Expr::Macro(em) if Self::is_slab_read(&em.mac))
    }

    /// Expand `let <pat> = slab_read!(Type, slab, offset);` into:
    ///
    /// ```ignore
    /// let mut slab_read_arr_N = [0u32; Type::SLAB_SIZE];
    /// slab_read_array!(slab, offset, slab_read_arr_N, Type::SLAB_SIZE);
    /// let <pat> = Type::from_array(slab_read_arr_N);
    /// ```
    fn expand_let_slab_read(&mut self, local: &syn::Local) -> Option<Vec<syn::Stmt>> {
        let init = local.init.as_ref()?;
        let syn::Expr::Macro(em) = &*init.expr else {
            return None;
        };
        let args: SlabReadArgs = syn::parse2(em.mac.tokens.clone()).ok()?;
        let ty = &args.ty;
        let slab = &args.slab;
        let offset = &args.offset;
        let arr_ident = self.unique_ident("slab_read_arr");
        let pat = &local.pat;

        let stmts: Vec<syn::Stmt> = vec![
            syn::parse_quote! {
                let mut #arr_ident = [0u32; #ty::SLAB_SIZE];
            },
            syn::parse_quote! {
                slab_read_array!(#slab, #offset, #arr_ident, #ty::SLAB_SIZE);
            },
            syn::parse_quote! {
                let #pat = #ty::from_array(#arr_ident);
            },
        ];
        Some(stmts)
    }

    /// Expand `slab_write!(Type, slab, offset, value);` into:
    ///
    /// ```ignore
    /// let slab_write_arr_N = Type::to_array(value);
    /// slab_write_array!(slab, offset, slab_write_arr_N, Type::SLAB_SIZE);
    /// ```
    fn expand_stmt_slab_write(&mut self, stmt_macro: &syn::StmtMacro) -> Option<Vec<syn::Stmt>> {
        let args: SlabWriteArgs = syn::parse2(stmt_macro.mac.tokens.clone()).ok()?;
        let ty = &args.ty;
        let slab = &args.slab;
        let offset = &args.offset;
        let value = &args.value;
        let arr_ident = self.unique_ident("slab_write_arr");

        let stmts: Vec<syn::Stmt> = vec![
            syn::parse_quote! {
                let #arr_ident = #ty::to_array(#value);
            },
            syn::parse_quote! {
                slab_write_array!(#slab, #offset, #arr_ident, #ty::SLAB_SIZE);
            },
        ];
        Some(stmts)
    }

    /// Expand `slab_read!(Type, slab, offset);` in statement position
    /// (result discarded).
    fn expand_stmt_slab_read(&mut self, stmt_macro: &syn::StmtMacro) -> Option<Vec<syn::Stmt>> {
        let args: SlabReadArgs = syn::parse2(stmt_macro.mac.tokens.clone()).ok()?;
        let ty = &args.ty;
        let slab = &args.slab;
        let offset = &args.offset;
        let arr_ident = self.unique_ident("slab_read_arr");

        let stmts: Vec<syn::Stmt> = vec![
            syn::parse_quote! {
                let mut #arr_ident = [0u32; #ty::SLAB_SIZE];
            },
            syn::parse_quote! {
                slab_read_array!(#slab, #offset, #arr_ident, #ty::SLAB_SIZE);
            },
            syn::parse_quote! {
                let _ = #ty::from_array(#arr_ident);
            },
        ];
        Some(stmts)
    }
}

/// Expand all `slab_read!` / `slab_write!` macro calls in a list of items.
fn expand_slab_macros(items: &mut [syn::Item]) {
    use syn::visit_mut::VisitMut;
    let mut expander = SlabMacroExpander::new();
    for item in items.iter_mut() {
        expander.visit_item_mut(item);
    }
}
