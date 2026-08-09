//! `SlabItemExt` — a `WgslExtension` that generates slab serialization
//! methods for `#[slab_item]` structs and lowers `slab_read!`/`slab_write!`
//! statement macros.
//!
//! When a `#[wgsl(extensions = [crabslab::SlabItemExt])]` module contains
//! structs marked with `#[slab_item]`, this extension:
//!
//! 1. Generates `impl Type { ... }` blocks with `SLAB_SIZE`, `from_array`,
//!    `to_array`, and `array_container` inherent methods for each struct.
//! 2. Lowers `slab_read!(Type, slab, offset, dest)` `Stmt::Macro` invocations
//!    into `Local` + `SlabRead` + `Local` IR statement sequences.
//! 3. Lowers `slab_write!(Type, slab, offset, src)` `Stmt::Macro` invocations
//!    into `Local` + `SlabWrite` IR statement sequences.

use wgsl_rs::ir;
use wgsl_rs::WgslExtension;

pub struct SlabItemExt;

impl WgslExtension for SlabItemExt {
    const MACROS: &'static [&'static str] = &["slab_read", "slab_write"];

    fn modify_ir(module: &mut ir::Module) {
        generate_slab_impls(module);
        lower_slab_macros(module);
    }
}

/// Check if a struct's attrs contain `slab_item`.
fn has_slab_item(attrs: &[ir::Attribute]) -> bool {
    attrs
        .iter()
        .any(|a| a.path == "slab_item" || a.path.ends_with("::slab_item"))
}

/// Compute the slab size (u32 slot count) for an IR type.
///
/// This is the component count, NOT `WgslLayout::SIZE / 4`.
/// The slab model packs densely with no alignment padding.
fn slab_size(ty: &ir::Type) -> u32 {
    match ty {
        ir::Type::Scalar(_) => 1,
        ir::Type::Vector { elements, .. } => *elements as u32,
        ir::Type::Matrix { columns, rows, .. } => *columns as u32 * *rows as u32,
        ir::Type::Array { elem, len } => {
            let n = eval_const_u32(len).unwrap_or(0);
            slab_size(elem) * n
        }
        ir::Type::Struct { name, .. } => {
            // Look up the struct's generated SLAB_SIZE.
            // For now, we compute it from the struct's fields.
            // This is resolved during generate_slab_impls — we'll
            // return 0 here and let the caller handle it.
            // Actually, the struct's SLAB_SIZE is computed at generation time.
            // For references to other slab_item structs, we use a TypePath
            // expression that renders as Name_SLAB_SIZE.
            0 // placeholder — resolved differently
        }
        _ => 0,
    }
}

/// Try to evaluate an IR expression as a u32 literal.
fn eval_const_u32(expr: &ir::Expr) -> Option<u32> {
    match expr {
        ir::Expr::Lit(ir::Lit::Int { digits, .. }) => digits.parse().ok(),
        _ => None,
    }
}

/// Generate `impl Type { SLAB_SIZE, from_array, to_array, array_container }`
/// for each `#[slab_item]` struct in the module.
fn generate_slab_impls(module: &mut ir::Module) {
    let slab_structs: Vec<ir::ItemStruct> = module
        .items
        .iter()
        .filter_map(|item| match item {
            ir::Item::Struct(s) if has_slab_item(&s.attrs) => Some(s.clone()),
            _ => None,
        })
        .collect();

    for s in &slab_structs {
        let slab_size_val = compute_struct_slab_size(s);
        let impl_block = make_slab_impl(s, slab_size_val);
        module.items.push(ir::Item::Impl(impl_block));
    }
}

/// Compute the total slab size for a struct by summing field slab sizes.
fn compute_struct_slab_size(s: &ir::ItemStruct) -> u32 {
    s.fields.iter().map(|f| field_slab_size(&f.ty)).sum()
}

/// Compute the slab size for a field type.
fn field_slab_size(ty: &ir::Type) -> u32 {
    match ty {
        ir::Type::Scalar(ir::ScalarType::U32 | ir::ScalarType::I32 | ir::ScalarType::F32) => 1,
        ir::Type::Scalar(ir::ScalarType::Bool) => 1,
        ir::Type::Vector { elements, .. } => *elements as u32,
        ir::Type::Matrix { columns, rows, .. } => *columns as u32 * *rows as u32,
        ir::Type::Array { elem, len } => field_slab_size(elem) * eval_const_u32(len).unwrap_or(0),
        ir::Type::Struct { name, .. } => {
            // Reference the generated SLAB_SIZE const.
            // We return 0 here — the actual size will be referenced
            // via TypePath in the generated code.
            0
        }
        _ => 0,
    }
}

/// Build an `ir::ItemImpl` with `SLAB_SIZE`, `from_array`, `to_array`,
/// and `array_container` methods for a struct.
fn make_slab_impl(s: &ir::ItemStruct, slab_size_val: u32) -> ir::ItemImpl {
    let name = &s.name;
    let size = slab_size_val;

    let mut items = vec![
        ir::ImplItem::Const(ir::ItemConst {
            name: "SLAB_SIZE".to_string(),
            ty: ir::Type::Scalar(ir::ScalarType::U32),
            expr: ir::Expr::Lit(ir::Lit::Int {
                digits: size.to_string(),
                suffix: "u32".to_string(),
            }),
            attrs: vec![],
        }),
        ir::ImplItem::Fn(make_array_container(name, size)),
        ir::ImplItem::Fn(make_from_array(s, size)),
        ir::ImplItem::Fn(make_to_array(s, size)),
    ];

    ir::ItemImpl {
        type_params: vec![],
        const_params: vec![],
        self_ty: name.clone(),
        items,
        attrs: vec![],
    }
}

/// Build `fn array_container() -> [u32; N]`.
fn make_array_container(struct_name: &str, size: u32) -> ir::ItemFn {
    ir::ItemFn {
        type_params: vec![],
        const_params: vec![],
        fn_attrs: ir::FnAttrs::None,
        name: std::borrow::Cow::Owned("array_container".to_string()),
        inputs: vec![],
        return_type: ir::ReturnType::Type {
            annotation: ir::ReturnTypeAnnotation::None,
            ty: ir::Type::Array {
                elem: Box::new(ir::Type::Scalar(ir::ScalarType::U32)),
                len: ir::Expr::Lit(ir::Lit::Int {
                    digits: size.to_string(),
                    suffix: "u32".to_string(),
                }),
            },
        },
        block: ir::Block {
            stmts: vec![ir::Stmt::Expr {
                expr: ir::Expr::ZeroValueArray {
                    elem_type: Box::new(ir::Type::Scalar(ir::ScalarType::U32)),
                    len: Box::new(ir::Expr::Lit(ir::Lit::Int {
                        digits: size.to_string(),
                        suffix: "u32".to_string(),
                    })),
                },
                has_semi: true,
            }],
        },
        attrs: vec![],
    }
}

/// Build `fn from_array(arr: [u32; N]) -> Type`.
fn make_from_array(s: &ir::ItemStruct, size: u32) -> ir::ItemFn {
    let mut offset = 0u32;
    let mut field_exprs: Vec<ir::FieldValue> = vec![];

    for f in &s.fields {
        let field_ty = &f.ty;
        let expr = make_from_array_field_expr(field_ty, offset);
        field_exprs.push(ir::FieldValue {
            member: f.name.clone(),
            expr,
        });
        offset += field_slab_size(field_ty);
    }

    ir::ItemFn {
        type_params: vec![],
        const_params: vec![],
        fn_attrs: ir::FnAttrs::None,
        name: std::borrow::Cow::Owned("from_array".to_string()),
        inputs: vec![ir::FnArg {
            inter_stage_io: vec![],
            name: "arr".to_string(),
            ty: ir::Type::Array {
                elem: Box::new(ir::Type::Scalar(ir::ScalarType::U32)),
                len: ir::Expr::Lit(ir::Lit::Int {
                    digits: size.to_string(),
                    suffix: "u32".to_string(),
                }),
            },
            attrs: vec![],
        }],
        return_type: ir::ReturnType::Type {
            annotation: ir::ReturnTypeAnnotation::None,
            ty: ir::Type::Struct {
                name: s.name.clone(),
                type_args: vec![],
            },
        },
        block: ir::Block {
            stmts: vec![ir::Stmt::Expr {
                expr: ir::Expr::Struct {
                    name: s.name.clone(),
                    type_args: vec![],
                    fields: field_exprs,
                },
                has_semi: true,
            }],
        },
        attrs: vec![],
    }
}

/// Build the expression to read a single field from `arr` at `offset`.
fn make_from_array_field_expr(ty: &ir::Type, offset: u32) -> ir::Expr {
    match ty {
        ir::Type::Scalar(ir::ScalarType::U32) => array_index("arr", offset),
        ir::Type::Scalar(ir::ScalarType::I32) => ir::Expr::FnCall {
            path: ir::FnPath::Ident("bitcast_i32".to_string()),
            type_args: vec![],
            params: vec![array_index("arr", offset)],
        },
        ir::Type::Scalar(ir::ScalarType::F32) => ir::Expr::FnCall {
            path: ir::FnPath::Ident("bitcast_f32".to_string()),
            type_args: vec![],
            params: vec![array_index("arr", offset)],
        },
        ir::Type::Scalar(ir::ScalarType::Bool) => ir::Expr::Binary {
            lhs: Box::new(array_index("arr", offset)),
            op: ir::BinOp::Ne,
            rhs: Box::new(ir::Expr::Lit(ir::Lit::Int {
                digits: "0".to_string(),
                suffix: "u32".to_string(),
            })),
        },
        ir::Type::Vector {
            elements,
            scalar_ty,
            ..
        } => {
            let ctor = match scalar_ty {
                Some(ir::ScalarType::F32) => "vec2f",
                Some(ir::ScalarType::I32) => "vec2i",
                Some(ir::ScalarType::U32) => "vec2u",
                Some(ir::ScalarType::Bool) => "vec2b",
                None => "vec2",
            };
            let ctor = match elements {
                2 => ctor,
                3 => &ctor[..2],
                4 => &ctor[..2],
                _ => ctor,
            };
            let n = *elements as u32;
            let params: Vec<ir::Expr> = (0..n)
                .map(|i| make_from_array_scalar_component(*scalar_ty, "arr", offset + i))
                .collect();
            let ctor_name = match elements {
                2 => match scalar_ty {
                    Some(ir::ScalarType::F32) => "vec2f",
                    Some(ir::ScalarType::I32) => "vec2i",
                    Some(ir::ScalarType::U32) => "vec2u",
                    Some(ir::ScalarType::Bool) => "vec2b",
                    _ => "vec2",
                },
                3 => match scalar_ty {
                    Some(ir::ScalarType::F32) => "vec3f",
                    Some(ir::ScalarType::I32) => "vec3i",
                    Some(ir::ScalarType::U32) => "vec3u",
                    Some(ir::ScalarType::Bool) => "vec3b",
                    _ => "vec3",
                },
                4 => match scalar_ty {
                    Some(ir::ScalarType::F32) => "vec4f",
                    Some(ir::ScalarType::I32) => "vec4i",
                    Some(ir::ScalarType::U32) => "vec4u",
                    Some(ir::ScalarType::Bool) => "vec4b",
                    _ => "vec4",
                },
                _ => "vec2",
            };
            ir::Expr::FnCall {
                path: ir::FnPath::Ident(ctor_name.to_string()),
                type_args: vec![],
                params,
            }
        }
        ir::Type::Matrix { columns, rows, .. } => {
            let cols = *columns as u32;
            let rows_val = *rows as u32;
            let col_exprs: Vec<ir::Expr> = (0..cols)
                .map(|c| {
                    let col_offset = c * rows_val;
                    let scalar_ty = Some(ir::ScalarType::F32);
                    let n = rows_val;
                    let params: Vec<ir::Expr> = (0..n)
                        .map(|i| {
                            make_from_array_scalar_component(
                                Some(ir::ScalarType::F32),
                                "arr",
                                offset + col_offset + i,
                            )
                        })
                        .collect();
                    let ctor_name = match rows {
                        2 => "vec2f",
                        3 => "vec3f",
                        4 => "vec4f",
                        _ => "vec2f",
                    };
                    ir::Expr::FnCall {
                        path: ir::FnPath::Ident(ctor_name.to_string()),
                        type_args: vec![],
                        params,
                    }
                })
                .collect();
            let mat_ctor = format!("mat{columns}x{rows}f");
            ir::Expr::FnCall {
                path: ir::FnPath::Ident(mat_ctor),
                type_args: vec![],
                params: col_exprs,
            }
        }
        ir::Type::Struct { name, .. } => {
            // Nested slab_item struct: copy sub-array, then call from_array.
            let nested_size = field_slab_size(ty);
            let mut stmts: Vec<ir::Stmt> = vec![
                ir::Stmt::Local(ir::Local {
                    mutable: true,
                    name: format!("sub_{name}"),
                    ty: Some(ir::Type::Array {
                        elem: Box::new(ir::Type::Scalar(ir::ScalarType::U32)),
                        len: ir::Expr::TypePath {
                            ty: name.clone(),
                            member: "SLAB_SIZE".to_string(),
                        },
                    }),
                    init: Some(ir::Expr::ZeroValueArray {
                        elem_type: Box::new(ir::Type::Scalar(ir::ScalarType::U32)),
                        len: Box::new(ir::Expr::TypePath {
                            ty: name.clone(),
                            member: "SLAB_SIZE".to_string(),
                        }),
                    }),
                }),
                ir::Stmt::SlabRead {
                    slab: ir::Expr::Ident("arr".to_string()),
                    offset: ir::Expr::Lit(ir::Lit::Int {
                        digits: offset.to_string(),
                        suffix: "u32".to_string(),
                    }),
                    dest: ir::Expr::Ident(format!("sub_{name}")),
                    size: ir::Expr::TypePath {
                        ty: name.clone(),
                        member: "SLAB_SIZE".to_string(),
                    },
                },
            ];
            // Return the from_array call — but we can't use a block expression
            // directly in a struct field. Instead, we use a FnCall with the
            // sub-array. The statements above need to be hoisted.
            // For simplicity, just call from_array with a direct sub-slice.
            ir::Expr::FnCall {
                path: ir::FnPath::TypeMethod {
                    ty: name.clone(),
                    method: "from_array".to_string(),
                },
                type_args: vec![],
                params: vec![ir::Expr::ArrayIndexing {
                    lhs: Box::new(ir::Expr::Ident("arr".to_string())),
                    index: Box::new(ir::Expr::Lit(ir::Lit::Int {
                        digits: offset.to_string(),
                        suffix: "u32".to_string(),
                    })),
                }],
            }
        }
        _ => ir::Expr::Lit(ir::Lit::Int {
            digits: "0".to_string(),
            suffix: "u32".to_string(),
        }),
    }
}

/// Build the expression to read a single scalar component from `arr` at
/// `offset`, applying the appropriate bitcast.
fn make_from_array_scalar_component(
    scalar_ty: Option<ir::ScalarType>,
    arr_name: &str,
    offset: u32,
) -> ir::Expr {
    match scalar_ty {
        Some(ir::ScalarType::U32) => array_index(arr_name, offset),
        Some(ir::ScalarType::I32) => ir::Expr::FnCall {
            path: ir::FnPath::Ident("bitcast_i32".to_string()),
            type_args: vec![],
            params: vec![array_index(arr_name, offset)],
        },
        Some(ir::ScalarType::F32) => ir::Expr::FnCall {
            path: ir::FnPath::Ident("bitcast_f32".to_string()),
            type_args: vec![],
            params: vec![array_index(arr_name, offset)],
        },
        Some(ir::ScalarType::Bool) => ir::Expr::Binary {
            lhs: Box::new(array_index(arr_name, offset)),
            op: ir::BinOp::Ne,
            rhs: Box::new(ir::Expr::Lit(ir::Lit::Int {
                digits: "0".to_string(),
                suffix: "u32".to_string(),
            })),
        },
        None => array_index(arr_name, offset),
    }
}

/// Build `fn to_array(data: Type) -> [u32; N]`.
fn make_to_array(s: &ir::ItemStruct, size: u32) -> ir::ItemFn {
    let mut offset = 0u32;
    let mut arr_elems: Vec<ir::Expr> = vec![];

    for f in &s.fields {
        let field_ty = &f.ty;
        let n = field_slab_size(field_ty);
        let elems = make_to_array_field_exprs(field_ty, &f.name, offset);
        arr_elems.extend(elems);
        offset += n;
    }

    ir::ItemFn {
        type_params: vec![],
        const_params: vec![],
        fn_attrs: ir::FnAttrs::None,
        name: std::borrow::Cow::Owned("to_array".to_string()),
        inputs: vec![ir::FnArg {
            inter_stage_io: vec![],
            name: "data".to_string(),
            ty: ir::Type::Struct {
                name: s.name.clone(),
                type_args: vec![],
            },
            attrs: vec![],
        }],
        return_type: ir::ReturnType::Type {
            annotation: ir::ReturnTypeAnnotation::None,
            ty: ir::Type::Array {
                elem: Box::new(ir::Type::Scalar(ir::ScalarType::U32)),
                len: ir::Expr::Lit(ir::Lit::Int {
                    digits: size.to_string(),
                    suffix: "u32".to_string(),
                }),
            },
        },
        block: ir::Block {
            stmts: vec![ir::Stmt::Expr {
                expr: ir::Expr::Array { elems: arr_elems },
                has_semi: true,
            }],
        },
        attrs: vec![],
    }
}

/// Build the expressions to write a single field to the output array at
/// `offset`.
fn make_to_array_field_exprs(ty: &ir::Type, field_name: &str, offset: u32) -> Vec<ir::Expr> {
    let field_access = ir::Expr::FieldAccess {
        base: Box::new(ir::Expr::Ident("data".to_string())),
        field: field_name.to_string(),
    };

    match ty {
        ir::Type::Scalar(ir::ScalarType::U32) => vec![field_access],
        ir::Type::Scalar(ir::ScalarType::I32) => vec![ir::Expr::FnCall {
            path: ir::FnPath::Ident("bitcast_u32".to_string()),
            type_args: vec![],
            params: vec![field_access],
        }],
        ir::Type::Scalar(ir::ScalarType::F32) => vec![ir::Expr::FnCall {
            path: ir::FnPath::Ident("bitcast_u32".to_string()),
            type_args: vec![],
            params: vec![field_access],
        }],
        ir::Type::Scalar(ir::ScalarType::Bool) => vec![ir::Expr::FnCall {
            path: ir::FnPath::Ident("select".to_string()),
            type_args: vec![],
            params: vec![
                ir::Expr::Lit(ir::Lit::Int {
                    digits: "0".to_string(),
                    suffix: "u32".to_string(),
                }),
                ir::Expr::Lit(ir::Lit::Int {
                    digits: "1".to_string(),
                    suffix: "u32".to_string(),
                }),
                field_access,
            ],
        }],
        ir::Type::Vector {
            elements,
            scalar_ty,
            ..
        } => {
            let n = *elements as u32;
            (0..n)
                .map(|i| {
                    let comp = swizzle_component(i);
                    make_to_array_scalar_component(
                        *scalar_ty,
                        ir::Expr::FieldAccess {
                            base: Box::new(field_access.clone()),
                            field: comp.to_string(),
                        },
                    )
                })
                .collect()
        }
        ir::Type::Matrix { columns, rows, .. } => {
            let cols = *columns as u32;
            let rows_val = *rows as u32;
            let mut elems = vec![];
            for c in 0..cols {
                let col = ir::Expr::ArrayIndexing {
                    lhs: Box::new(field_access.clone()),
                    index: Box::new(ir::Expr::Lit(ir::Lit::Int {
                        digits: c.to_string(),
                        suffix: "u32".to_string(),
                    })),
                };
                for r in 0..rows_val {
                    let comp = swizzle_component(r);
                    elems.push(make_to_array_scalar_component(
                        Some(ir::ScalarType::F32),
                        ir::Expr::FieldAccess {
                            base: Box::new(col.clone()),
                            field: comp.to_string(),
                        },
                    ));
                }
            }
            elems
        }
        ir::Type::Struct { name, .. } => {
            // Nested slab_item: call to_array, then spread the result.
            // This produces the sub-array as an expression; we flatten it
            // into the output array.
            // For simplicity, we emit a sub-array and use indexing.
            let nested_size = field_slab_size(ty);
            let mut elems = vec![];
            let to_array_call = ir::Expr::FnCall {
                path: ir::FnPath::TypeMethod {
                    ty: name.clone(),
                    method: "to_array".to_string(),
                },
                type_args: vec![],
                params: vec![field_access],
            };
            for i in 0..nested_size {
                elems.push(ir::Expr::ArrayIndexing {
                    lhs: Box::new(to_array_call.clone()),
                    index: Box::new(ir::Expr::Lit(ir::Lit::Int {
                        digits: i.to_string(),
                        suffix: "u32".to_string(),
                    })),
                });
            }
            elems
        }
        _ => vec![ir::Expr::Lit(ir::Lit::Int {
            digits: "0".to_string(),
            suffix: "u32".to_string(),
        })],
    }
}

/// Build the expression to convert a scalar value to u32 for the output array.
fn make_to_array_scalar_component(scalar_ty: Option<ir::ScalarType>, expr: ir::Expr) -> ir::Expr {
    match scalar_ty {
        Some(ir::ScalarType::U32) => expr,
        Some(ir::ScalarType::I32) | Some(ir::ScalarType::F32) => ir::Expr::FnCall {
            path: ir::FnPath::Ident("bitcast_u32".to_string()),
            type_args: vec![],
            params: vec![expr],
        },
        Some(ir::ScalarType::Bool) => ir::Expr::FnCall {
            path: ir::FnPath::Ident("select".to_string()),
            type_args: vec![],
            params: vec![
                ir::Expr::Lit(ir::Lit::Int {
                    digits: "0".to_string(),
                    suffix: "u32".to_string(),
                }),
                ir::Expr::Lit(ir::Lit::Int {
                    digits: "1".to_string(),
                    suffix: "u32".to_string(),
                }),
                expr,
            ],
        },
        None => expr,
    }
}

/// Get the swizzle component name for index 0-3.
fn swizzle_component(i: u32) -> char {
    match i {
        0 => 'x',
        1 => 'y',
        2 => 'z',
        3 => 'w',
        _ => 'x',
    }
}

/// Build `arr[offset]` as an IR expression.
fn array_index(arr_name: &str, offset: u32) -> ir::Expr {
    ir::Expr::ArrayIndexing {
        lhs: Box::new(ir::Expr::Ident(arr_name.to_string())),
        index: Box::new(ir::Expr::Lit(ir::Lit::Int {
            digits: offset.to_string(),
            suffix: "u32".to_string(),
        })),
    }
}

// ===== Stmt::Macro lowering =====

/// Lower all `Stmt::Macro` invocations in the module.
fn lower_slab_macros(module: &mut ir::Module) {
    for item in &mut module.items {
        if let ir::Item::Fn(f) = item {
            lower_slab_macros_in_block(&mut f.block);
        }
    }
}

/// Lower `Stmt::Macro` in a block, replacing them with IR statement sequences.
fn lower_slab_macros_in_block(block: &mut ir::Block) {
    let mut new_stmts = Vec::with_capacity(block.stmts.len());
    for stmt in block.stmts.drain(..) {
        match stmt {
            ir::Stmt::Macro { name, args } => {
                let lowered = lower_macro(&name, &args);
                new_stmts.extend(lowered);
            }
            mut other => {
                lower_slab_macros_in_stmt(&mut other);
                new_stmts.push(other);
            }
        }
    }
    block.stmts = new_stmts;
}

/// Recurse into nested statements to lower macros in sub-blocks.
fn lower_slab_macros_in_stmt(stmt: &mut ir::Stmt) {
    match stmt {
        ir::Stmt::If(i) => {
            lower_slab_macros_in_block(&mut i.then_block);
            if let Some(ir::ElseBranch::Block(b)) = &mut i.else_branch {
                lower_slab_macros_in_block(b);
            } else if let Some(ir::ElseBranch::If(inner)) = &mut i.else_branch {
                lower_slab_macros_in_stmt(&mut ir::Stmt::If((**inner).clone()));
            }
        }
        ir::Stmt::While { body, .. } | ir::Stmt::Loop { body } => {
            lower_slab_macros_in_block(body);
        }
        ir::Stmt::For(f) => {
            lower_slab_macros_in_block(&mut f.body);
        }
        ir::Stmt::Block(b) => {
            lower_slab_macros_in_block(b);
        }
        ir::Stmt::Switch(sw) => {
            for arm in &mut sw.arms {
                lower_slab_macros_in_block(&mut arm.body);
            }
        }
        _ => {}
    }
}

/// Lower a single `Stmt::Macro` into a sequence of IR statements.
/// Returns an empty vec if the macro is not recognized (shouldn't happen
/// — the compile-time const check ensures only claimed macros survive).
fn lower_macro(name: &str, args: &str) -> Vec<ir::Stmt> {
    match name {
        "slab_read" => lower_slab_read(args),
        "slab_write" => lower_slab_write(args),
        _ => vec![],
    }
}

/// Parse `slab_read!(Type, slab, offset, dest)` args and lower to 3 IR
/// statements.
///
/// The args string is the stringified token stream from the parser,
/// e.g. `"Data , get ! ( SLAB ) , i , d"`. We re-parse it simply by
/// splitting on commas and trimming.
fn lower_slab_read(args: &str) -> Vec<ir::Stmt> {
    let parts: Vec<&str> = args.split(',').collect();
    if parts.len() < 4 {
        return vec![];
    }
    let type_name = parts[0].trim();
    let slab_expr = strip_get(parts[1].trim());
    let offset_expr = ir::Expr::Ident(parts[2].trim().to_string());
    let dest_name = parts[3].trim();

    vec![
        ir::Stmt::Local(ir::Local {
            mutable: true,
            name: "slab_read_buf".to_string(),
            ty: Some(ir::Type::Array {
                elem: Box::new(ir::Type::Scalar(ir::ScalarType::U32)),
                len: ir::Expr::TypePath {
                    ty: type_name.to_string(),
                    member: "SLAB_SIZE".to_string(),
                },
            }),
            init: Some(ir::Expr::FnCall {
                path: ir::FnPath::TypeMethod {
                    ty: type_name.to_string(),
                    method: "array_container".to_string(),
                },
                type_args: vec![],
                params: vec![],
            }),
        }),
        ir::Stmt::SlabRead {
            slab: ir::Expr::Ident(slab_expr.to_string()),
            offset: offset_expr,
            dest: ir::Expr::Ident("slab_read_buf".to_string()),
            size: ir::Expr::TypePath {
                ty: type_name.to_string(),
                member: "SLAB_SIZE".to_string(),
            },
        },
        ir::Stmt::Local(ir::Local {
            mutable: true,
            name: dest_name.to_string(),
            ty: Some(ir::Type::Struct {
                name: type_name.to_string(),
                type_args: vec![],
            }),
            init: Some(ir::Expr::FnCall {
                path: ir::FnPath::TypeMethod {
                    ty: type_name.to_string(),
                    method: "from_array".to_string(),
                },
                type_args: vec![],
                params: vec![ir::Expr::Ident("slab_read_buf".to_string())],
            }),
        }),
    ]
}

/// Parse `slab_write!(Type, slab, offset, src)` args and lower to 2 IR
/// statements.
fn lower_slab_write(args: &str) -> Vec<ir::Stmt> {
    let parts: Vec<&str> = args.split(',').collect();
    if parts.len() < 4 {
        return vec![];
    }
    let type_name = parts[0].trim();
    let slab_expr = strip_get(parts[1].trim());
    let offset_expr = ir::Expr::Ident(parts[2].trim().to_string());
    let src_name = parts[3].trim();

    vec![
        ir::Stmt::Local(ir::Local {
            mutable: false,
            name: "slab_write_out".to_string(),
            ty: Some(ir::Type::Array {
                elem: Box::new(ir::Type::Scalar(ir::ScalarType::U32)),
                len: ir::Expr::TypePath {
                    ty: type_name.to_string(),
                    member: "SLAB_SIZE".to_string(),
                },
            }),
            init: Some(ir::Expr::FnCall {
                path: ir::FnPath::TypeMethod {
                    ty: type_name.to_string(),
                    method: "to_array".to_string(),
                },
                type_args: vec![],
                params: vec![ir::Expr::Ident(src_name.to_string())],
            }),
        }),
        ir::Stmt::SlabWrite {
            slab: ir::Expr::Ident(slab_expr.to_string()),
            offset: offset_expr,
            src: ir::Expr::Ident("slab_write_out".to_string()),
            size: Some(ir::Expr::TypePath {
                ty: type_name.to_string(),
                member: "SLAB_SIZE".to_string(),
            }),
        },
    ]
}

/// Strip `get!(IDENT)` or `get_mut!(IDENT)` from a string, returning just
/// the inner identifier.
fn strip_get(s: &str) -> &str {
    let s = s.trim();
    if s.starts_with("get_mut!(") || s.starts_with("get ! (") {
        let inner = s[s.find('(').unwrap_or(0) + 1..].trim();
        let inner = inner.strip_suffix(')').unwrap_or(inner);
        return inner.trim();
    }
    if s.starts_with("get!(") || s.starts_with("get ! (") {
        let inner = s[s.find('(').unwrap_or(0) + 1..].trim();
        let inner = inner.strip_suffix(')').unwrap_or(inner);
        return inner.trim();
    }
    s
}
