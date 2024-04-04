//! Provides derive macros for `crabslab`.
use quote::{format_ident, quote};
use syn::{
    spanned::Spanned, Data, DataEnum, DataStruct, DeriveInput, Fields, FieldsNamed, FieldsUnnamed,
    Ident, Index, Type, TypeTuple, WhereClause, WherePredicate,
};

enum FieldName {
    Index(Index),
    Ident(Ident),
}

struct FieldParams {
    field_tys: Vec<Type>,
    field_names: Vec<FieldName>,
}

impl FieldParams {
    fn new(fields: &syn::punctuated::Punctuated<syn::Field, syn::token::Comma>) -> Self {
        let field_tys: Vec<_> = fields.iter().map(|field| field.ty.clone()).collect();
        let field_names: Vec<_> = fields
            .iter()
            .enumerate()
            .map(|(i, field)| {
                field
                    .ident
                    .clone()
                    .map(FieldName::Ident)
                    .unwrap_or_else(|| {
                        FieldName::Index(Index {
                            index: i as u32,
                            span: field.span(),
                        })
                    })
            })
            .collect();
        Self {
            field_tys,
            field_names,
        }
    }
}

fn get_struct_params(ds: &DataStruct) -> FieldParams {
    let empty_punctuated = syn::punctuated::Punctuated::new();
    let fields = match ds {
        DataStruct {
            fields: Fields::Named(FieldsNamed { named: ref x, .. }),
            ..
        } => x,
        DataStruct {
            fields: Fields::Unnamed(FieldsUnnamed { unnamed: ref x, .. }),
            ..
        } => x,
        DataStruct {
            fields: Fields::Unit,
            ..
        } => &empty_punctuated,
    };

    FieldParams::new(fields)
}

struct EnumVariant {
    variant: syn::Variant,
    fields: FieldParams,
}

struct EnumParams {
    variants: Vec<EnumVariant>,
    slab_size: proc_macro2::TokenStream,
}

fn get_enum_params(de: &DataEnum) -> EnumParams {
    let DataEnum {
        enum_token: _,
        brace_token: _,
        variants,
    } = de;
    let variants = variants
        .iter()
        .map(|variant| {
            let empty_fields = syn::punctuated::Punctuated::new();
            let fields = match &variant.fields {
                Fields::Named(FieldsNamed { named: ref x, .. }) => x,
                Fields::Unnamed(FieldsUnnamed { unnamed: ref x, .. }) => x,
                Fields::Unit => &empty_fields,
            };
            let fields = FieldParams::new(fields);
            EnumVariant {
                variant: variant.clone(),
                fields,
            }
        })
        .collect::<Vec<_>>();
    let slab_size_def = quote! {
        let mut __size = 0usize;
    };
    let slab_size_increments = variants
        .iter()
        .map(|variant| {
            let tys = &variant.fields.field_tys;
            if tys.is_empty() {
                quote! {}
            } else {
                quote! {{
                    let __field_size = #( <#tys as crabslab::SlabItem>::SLAB_SIZE )+*;
                    __size += crabslab::__saturating_sub(__field_size,__size);
                }}
            }
        })
        .collect::<Vec<_>>();
    EnumParams {
        slab_size: quote! {
            #slab_size_def
            #(#slab_size_increments)*
            // Add one for the enum variant
            __size + 1
        },
        variants,
    }
}

enum Params {
    Struct(FieldParams),
    Enum(EnumParams),
}

fn get_params(input: &DeriveInput) -> syn::Result<Params> {
    match &input.data {
        Data::Struct(ds) => Ok(Params::Struct(get_struct_params(ds))),
        Data::Enum(de) => Ok(Params::Enum(get_enum_params(de))),
        _ => Err(syn::Error::new(
            input.span(),
            "deriving SlabItem does not support unions".to_string(),
        )),
    }
}

/// Derives `SlabItem` for a struct.
///
/// For structs this will also implement `offset_of_{field}` and `slab_size_of_{field}` functions for each
/// field, which returns the offset of that field relative to the start of the struct or the size of the field (each in u32):
///
/// ```rust
/// use crabslab::{CpuSlab, GrowableSlab, Slab, SlabItem};
///
/// #[derive(Debug, Default, PartialEq, SlabItem)]
/// struct Foo {
///     a: u32,
///     b: u32,
///     c: u32,
/// }
///
/// let foo_one = Foo { a: 1, b: 2, c: 3 };
/// let foo_two = Foo { a: 4, b: 5, c: 6 };
///
/// let mut slab = CpuSlab::new(vec![]);
/// let foo_one_id = slab.append(&foo_one);
/// let foo_two_id = slab.append(&foo_two);
///
/// // Overwrite the second item of the second `Foo`:
/// slab.write(foo_two_id + Foo::offset_of_b(), &42);
/// assert_eq!(Foo { a: 4, b: 42, c: 6 }, slab.read(foo_two_id));
/// ```
///
/// No such offsets are derived for enums.
///
/// ```rust
/// use crabslab::{CpuSlab, GrowableSlab, Slab, SlabItem};
///
/// #[derive(Debug, Default, PartialEq, SlabItem)]
/// struct Bar {
///     a: u32,
/// }
///
/// #[derive(Debug, Default, PartialEq, SlabItem)]
/// enum Baz {
///     #[default]
///     One,
///     Two {
///         a: u32,
///         b: u32,
///     },
///     Three(u32, u32),
///     Four(Bar),
/// }
///
/// assert_eq!(3, Baz::SLAB_SIZE);
///
/// let mut slab = CpuSlab::new(vec![]);
///
/// let one_id = slab.append(&Baz::One);
/// let two_id = slab.append(&Baz::Two { a: 1, b: 2 });
/// let three_id = slab.append(&Baz::Three(3, 4));
/// let four_id = slab.append(&Baz::Four(Bar { a: 5 }));
///
/// assert_eq!(Baz::One, slab.read(one_id));
/// assert_eq!(Baz::Two { a: 1, b: 2 }, slab.read(two_id));
/// assert_eq!(Baz::Three(3, 4), slab.read(three_id));
/// assert_eq!(Baz::Four(Bar { a: 5 }), slab.read(four_id));
/// ```
#[proc_macro_derive(SlabItem)]
pub fn derive_from_slab(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input: DeriveInput = syn::parse_macro_input!(input);

    match get_params(&input) {
        Ok(Params::Struct(p)) => derive_from_slab_struct(input, p),
        Ok(Params::Enum(p)) => derive_from_slab_enum(input, p),
        Err(e) => e.into_compile_error().into(),
    }
}

fn derive_from_slab_enum(input: DeriveInput, params: EnumParams) -> proc_macro::TokenStream {
    let EnumParams {
        variants,
        slab_size,
    } = params;
    let name = &input.ident;
    let field_tys = variants
        .iter()
        .flat_map(|v| v.fields.field_tys.clone())
        .collect::<Vec<_>>();
    let mut generics = input.generics;
    {
        fn constrain_system_data_types(clause: &mut WhereClause, tys: &[Type]) {
            for ty in tys.iter() {
                let where_predicate: WherePredicate = syn::parse_quote!(#ty : crabslab::SlabItem);
                clause.predicates.push(where_predicate);
            }
        }

        let where_clause = generics.make_where_clause();
        constrain_system_data_types(where_clause, &field_tys)
    }
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    let variant_reads = variants.iter().map(|variant| {
        let ident = &variant.variant.ident;
        let field_names = variant
            .fields
            .field_names
            .iter()
            .map(|name| match name {
                FieldName::Index(i) => Ident::new(&format!("__{}", i.index), i.span),
                FieldName::Ident(field) => field.clone(),
            })
            .collect::<Vec<_>>();
        let field_tys = &variant.fields.field_tys;
        let num_fields = field_names.len();
        let reads = field_names
            .iter()
            .zip(field_tys.iter())
            .enumerate()
            .map(|(i, (name, ty))| {
                let def = quote! {
                    let #name = <#ty as crabslab::SlabItem>::read_slab(index, slab);
                };
                let increment_index = if i + 1 < num_fields {
                    quote! {
                        index += <#ty as crabslab::SlabItem>::SLAB_SIZE;
                    }
                } else {
                    quote! {}
                };
                quote! {
                    #def
                    #increment_index
                }
            })
            .collect::<Vec<_>>();

        match variant.variant.fields {
            Fields::Named(_) => {
                quote! {{
                    #(#reads)*
                     #name::#ident {
                         #(#field_names),*
                     }
                }}
            }
            Fields::Unnamed(_) => {
                quote! {{
                    #(#reads)*
                    #name::#ident(
                        #(#field_names),*
                    )
                }}
            }
            Fields::Unit => quote! {
                #name::#ident,
            },
        }
    });
    let read_variants_matches: Vec<proc_macro2::TokenStream> = variants
        .iter()
        .enumerate()
        .zip(variant_reads)
        .map(|((i, variant), read)| {
            let hash = syn::LitInt::new(&i.to_string(), variant.variant.span());
            quote! {
                #hash => #read
            }
        })
        .collect();
    let variant_writes = variants.iter().map(|variant| {
        let field_names = variant
            .fields
            .field_names
            .iter()
            .map(|name| match name {
                FieldName::Index(i) => Ident::new(&format!("__{}", i.index), i.span),
                FieldName::Ident(field) => field.clone(),
            })
            .collect::<Vec<_>>();
        quote! {
            #(let index = #field_names.write_slab(index, slab);)*
        }
    });
    let write_variants_matches: Vec<proc_macro2::TokenStream> = variants
        .iter()
        .enumerate()
        .zip(variant_writes)
        .map(|((i, variant), write)| {
            let hash = syn::LitInt::new(&i.to_string(), variant.variant.span());
            let field_names = variant
                .fields
                .field_names
                .iter()
                .map(|name| match name {
                    FieldName::Index(i) => Ident::new(&format!("__{}", i.index), i.span),
                    FieldName::Ident(field) => field.clone(),
                })
                .collect::<Vec<_>>();
            let ident = &variant.variant.ident;
            let pat_match = match variant.variant.fields {
                Fields::Named(_) => {
                    quote! {
                        #name::#ident {
                            #(#field_names,)*
                        }
                    }
                }
                Fields::Unnamed(_) => {
                    quote! {
                        #name::#ident(
                            #(#field_names,)*
                        )
                    }
                }
                Fields::Unit => quote! {
                    #name::#ident
                },
            };
            quote! {
                #pat_match => {
                    let __hash: u32 = #hash;
                    let index = __hash.write_slab(index, slab);
                    #write
                    original_index + slab_size
                }
            }
        })
        .collect();

    let output = quote! {
        #[automatically_derived]
        impl #impl_generics crabslab::SlabItem for #name #ty_generics #where_clause
        {
            const SLAB_SIZE: usize = {#slab_size};

            fn read_slab(mut index: usize, slab: &[u32]) -> Self {
                // Read the hash to tell which variant we're in.
                let hash =  u32::read_slab(index, slab);
                index += 1;
                match hash {
                    #(#read_variants_matches)*
                    _ => Default::default(),
                }
            }

            fn write_slab(&self, index: usize, slab: &mut [u32]) -> usize {
                let slab_size = Self::SLAB_SIZE;
                let original_index = index;
                match self {
                    #(#write_variants_matches)*
                }
            }
        }
    };
    output.into()
}

fn derive_from_slab_struct(input: DeriveInput, params: FieldParams) -> proc_macro::TokenStream {
    let FieldParams {
        field_tys,
        field_names,
    } = params;

    let name = &input.ident;
    let is_struct_style = match field_names.first() {
        Some(FieldName::Index(_)) => false,
        _ => true,
    };
    let mut generics = input.generics;
    {
        /// Adds a `CanFetch<'lt>` bound on each of the system data types.
        fn constrain_system_data_types(clause: &mut WhereClause, tys: &[Type]) {
            for ty in tys.iter() {
                let where_predicate: WherePredicate = syn::parse_quote!(#ty : crabslab::SlabItem);
                clause.predicates.push(where_predicate);
            }
        }

        let where_clause = generics.make_where_clause();
        constrain_system_data_types(where_clause, &field_tys)
    }
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
    let read_field_names = field_names
        .iter()
        .zip(field_tys.iter())
        .enumerate()
        .map(|(i, (name, ty))| {
            let var = Ident::new(&format!("__{i}"), ty.span());
            let inner = quote! {{
                let #var = <#ty as crabslab::SlabItem>::read_slab(index, slab);
                index += <#ty as crabslab::SlabItem>::SLAB_SIZE;
                #var
            }};
            match name {
                FieldName::Index(_) => inner,
                FieldName::Ident(n) => {
                    quote! {
                        #n: #inner
                    }
                }
            }
        })
        .collect::<Vec<_>>();
    let read_impl = if is_struct_style {
        quote! {
            Self { #(#read_field_names),* }
        }
    } else {
        quote! {
            Self( #(#read_field_names),* )
        }
    };
    let write_index_field_names = field_names
        .iter()
        .map(|name| match name {
            FieldName::Index(i) => quote! {
                let index = self.#i.write_slab(index, slab);
            },
            FieldName::Ident(field) => quote! {
                let index = self.#field.write_slab(index, slab);
            },
        })
        .collect::<Vec<_>>();

    let mut offset_tys = vec![];
    let mut offsets = vec![];
    for (name, ty) in field_names.iter().zip(field_tys.iter()) {
        let (offset_of_ident, slab_size_of_ident) = match name {
            FieldName::Index(i) => (
                Ident::new(&format!("offset_of_{}", i.index), i.span),
                Ident::new(&format!("slab_size_of_{}", i.index), i.span),
            ),
            FieldName::Ident(field) => (
                Ident::new(&format!("offset_of_{}", field), field.span()),
                Ident::new(&format!("slab_size_of_{}", field), field.span()),
            ),
        };
        offsets.push(quote! {
            pub fn #offset_of_ident() -> crabslab::Offset<#ty, Self> {
                crabslab::Offset::new(
                    #(<#offset_tys as crabslab::SlabItem>::SLAB_SIZE+)*
                    0
                )
            }

            pub fn #slab_size_of_ident() -> usize {
                <#ty as crabslab::SlabItem>::SLAB_SIZE
            }
        });
        offset_tys.push(ty.clone());
    }

    let output = quote! {
        #[automatically_derived]
        /// Offsets into the slab buffer for each field.
        impl #impl_generics #name #ty_generics {
            #(#offsets)*
        }

        #[automatically_derived]
        impl #impl_generics crabslab::SlabItem for #name #ty_generics #where_clause
        {
            const SLAB_SIZE: usize = {
                #( <#field_tys as crabslab::SlabItem>::SLAB_SIZE )+*
            };

            fn read_slab(mut index: usize, slab: &[u32]) -> Self {
                #read_impl
            }

            fn write_slab(&self, index: usize, slab: &mut [u32]) -> usize {
                if slab.len() < index + Self::SLAB_SIZE {
                    return index;
                }

                #(#write_index_field_names)*

                index
            }
        }
    };
    output.into()
}

#[proc_macro]
pub fn impl_slabitem_tuples(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let tuple: TypeTuple = syn::parse_macro_input!(input);
    let tys = tuple.elems.iter().collect::<Vec<_>>();
    let indices = tys
        .iter()
        .enumerate()
        .map(|(i, _)| Index::from(i))
        .collect::<Vec<_>>();
    let reads = tys
        .iter()
        .enumerate()
        .map(|(i, ty)| {
            let var = Ident::new(&format!("__{i}"), ty.span());
            quote! {{
                    let #var = <#ty as crabslab::SlabItem>::read_slab(index, slab);
                    index += <#ty as crabslab::SlabItem>::SLAB_SIZE;
                    #var
            }}
        })
        .collect::<Vec<_>>();
    let output = quote! {
        impl<#(#tys),*> crabslab::SlabItem for #tuple
        where
            #(#tys: crabslab::SlabItem),*,
        {
            const SLAB_SIZE: usize = {
                #(#tys::SLAB_SIZE )+*
            };
            fn read_slab(mut index: usize, slab: &[u32]) -> Self {
                (
                    #( #reads ,)*
                )
            }
            fn write_slab(&self, index: usize, slab: &mut [u32]) -> usize {
                #(let index = self.#indices.write_slab(index, slab);)*
                index
            }
        }
    };
    output.into()
}
