//! `SlabItemExt` — a `WgslExtension` that gives `#[derive(SlabItem)]`
//! structs their GPU-side serialization impls, and lowers the
//! `slab_read!`/`slab_write!` statement macros.
//!
//! # Why an extension?
//!
//! Derive macros run AFTER `#[wgsl]`, so the transpiler never sees the
//! `impl SlabItem` that `#[derive(SlabItem)]` generates on the CPU. This
//! extension closes the gap: it detects `#[derive(..., SlabItem)]` structs
//! in the module's IR (derive attributes are preserved on
//! [`ir::ItemStruct::attrs`]) and generates the same serialization
//! functions the CPU derive generates — as inherent methods, since WGSL
//! has no traits. Rendering mangles them (`Foo__1to_array`), identical to
//! the hand-written trait-impl path.
//!
//! # Generated shape
//!
//! The generated methods mirror the CPU derive's sequential field walk:
//! a const-sum `SLAB_SIZE`, `array_container`, and `to_array`/`from_array`
//! pairs that walk fields with a running index `i`, calling the field
//! type's own mangled methods and copying with `Stmt::SlabCopy` loops.
//! For
//!
//! ```rust,ignore
//! #[derive(Wgsl, SlabItem)]
//! struct Foo { count: u32, inner: Bar, tags: [u32; 3] }
//! ```
//!
//! the extension generates (rendered):
//!
//! ```wgsl
//! const Foo__1SLAB_SIZE: u32 =
//!     ((u32__1SLAB_SIZE + Bar__1SLAB_SIZE) + (3 * u32__1SLAB_SIZE));
//!
//! fn Foo__1array_container() -> array<u32, ...> {
//!     return array<u32, ...>();
//! }
//!
//! fn Foo__1to_array(data: Foo) -> array<u32, ...> {
//!     var dest = Foo__1array_container();
//!     var i: u32 = 0u;
//!     let count_slab = u32__1to_array(data.count);
//!     for (var _i: u32 = 0u; _i < u32__1SLAB_SIZE; _i++) {
//!         dest[i + _i] = count_slab[0u + _i];
//!     }
//!     i += u32__1SLAB_SIZE;
//!     // `inner` recurses through Bar__1to_array; `tags` unrolls one
//!     // to_array/copy pair per element (arrays are not slab items).
//!     ...
//!     return dest;
//! }
//! ```
//!
//! # Primitives
//!
//! There is no `[u32; N]::to_array` — the slab surface is values-only,
//! and array fields are unrolled element-wise, matching the CPU derive.
//! The mangled primitive methods (`u32__1to_array`, `f32__1SLAB_SIZE`, …)
//! are not generated here: a module whose derived structs use them must
//! import them, e.g. `use crabslab::slab_item::*;` (they live in the
//! `crabslab::slab_item` module's `WGSL_SOURCE`).
//!
//! # Cross-module caveat
//!
//! `SLAB_SIZE` sums and recursion reference the FIELD types' mangled
//! names (`Inner__1SLAB_SIZE`, `Inner__1to_array`). A `#[derive(SlabItem)]`
//! struct imported from another module needs that struct's generated
//! methods to exist in the assembled WGSL: the module that declares the
//! field must also run this extension (list `crabslab::SlabItemExt` in its
//! `extensions`) so `Inner__1SLAB_SIZE` is generated there.
//!
//! # Enums
//!
//! The extension detects derive attrs on structs only. Enum slab items
//! (discriminant + payload) are currently CPU-only; a GPU `Item::Enum`
//! arm can be added later if a concrete need appears.

use std::borrow::Cow;
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

/// Whether a struct's attrs include `SlabItem` in a `#[derive(...)]`
/// list.
///
/// Matching is lenient: qualified spellings (`crabslab::SlabItem`) and
/// stringification whitespace (`crabslab :: SlabItem`) both match, since
/// derive args are stringified token streams rather than parsed paths.
fn has_slab_item(attrs: &[ir::Attribute]) -> bool {
    attrs.iter().any(|a| {
        a.path == "derive"
            && a.args
                .iter()
                .any(|arg| arg.trim().rsplit("::").next().map(str::trim) == Some("SlabItem"))
    })
}

// ===== Size expressions =====

/// The u32 literal expression `N` (rendered `Nu`).
fn lit_u32(n: u32) -> ir::Expr {
    ir::Expr::Lit(ir::Lit::Int {
        digits: n.to_string(),
        suffix: "u32".to_string(),
    })
}

/// The WGSL-side spelling of a scalar type.
fn scalar_name(s: ir::ScalarType) -> String {
    match s {
        ir::ScalarType::U32 => "u32",
        ir::ScalarType::I32 => "i32",
        ir::ScalarType::F32 => "f32",
        ir::ScalarType::Bool => "bool",
    }
    .to_string()
}

/// The type name whose mangled methods implement slab serialization for
/// a field type: scalars serialize through the imported primitive impls,
/// structs through their own generated (or imported) impls.
///
/// Panics for anything else — the CPU derive only compiles when every
/// field is a slab item (scalar, slab struct, or array of those), so a
/// different type here means the two sides have diverged.
fn slab_item_type_name(ty: &ir::Type) -> String {
    match ty {
        ir::Type::Scalar(s) => scalar_name(*s),
        ir::Type::Struct { name, .. } => name.clone(),
        other => panic!(
            "SlabItemExt: field type `{other}` is not a slab item; only \
             scalars, slab structs, and arrays of those are supported"
        ),
    }
}

/// The `SLAB_SIZE` const-expression a field type contributes, mirroring
/// the CPU derive: scalars and structs reference the type's mangled
/// `SLAB_SIZE` const (`f32__1SLAB_SIZE`), arrays contribute
/// `n * <elem size>`.
fn field_size_expr(ty: &ir::Type) -> ir::Expr {
    match ty {
        ir::Type::Scalar(_) | ir::Type::Struct { .. } => ir::Expr::TypePath {
            ty: slab_item_type_name(ty),
            member: "SLAB_SIZE".to_string(),
        },
        ir::Type::Array { elem, len } => ir::Expr::Binary {
            lhs: Box::new(len.clone()),
            op: ir::BinOp::Mul,
            rhs: Box::new(field_size_expr(elem)),
        },
        other => panic!(
            "SlabItemExt: field type `{other}` is not a slab item; only \
             scalars, slab structs, and arrays of those are supported"
        ),
    }
}

/// Fold the per-field size expressions into the struct's `SLAB_SIZE`
/// const-expression — a left-nested `+` chain, mirroring the CPU derive's
/// `A::SLAB_SIZE + B::SLAB_SIZE + ...`. A fieldless struct contributes
/// `0u`.
fn struct_size_expr(s: &ir::ItemStruct) -> ir::Expr {
    let mut sum: Option<ir::Expr> = None;
    for f in &s.fields {
        let field_size = field_size_expr(&f.ty);
        sum = Some(match sum {
            None => field_size,
            Some(acc) => ir::Expr::Binary {
                lhs: Box::new(acc),
                op: ir::BinOp::Add,
                rhs: Box::new(field_size),
            },
        });
    }
    sum.unwrap_or_else(|| lit_u32(0))
}

/// The `Self::Array` type for a struct: `[u32; <SLAB_SIZE sum>]`.
fn slab_array_ty(size_expr: &ir::Expr) -> ir::Type {
    ir::Type::Array {
        elem: Box::new(ir::Type::Scalar(ir::ScalarType::U32)),
        len: size_expr.clone(),
    }
}

/// Evaluate a fixed array length as a literal element count. Array
/// fields must have literal lengths (the CPU derive requires the same,
/// for its const `SLAB_SIZE` contribution).
fn eval_array_len(len: &ir::Expr) -> u32 {
    match len {
        ir::Expr::Lit(ir::Lit::Int { digits, .. }) => digits.parse().unwrap_or_else(|_| {
            panic!("SlabItemExt: array slab field length `{digits}` is not a u32")
        }),
        other => {
            panic!("SlabItemExt: array slab fields must have a literal length, got `{other:?}`")
        }
    }
}

// ===== IR construction helpers =====

/// `<ty>::method(params...)`, rendered as the mangled `ty__1method`.
fn type_method_call(ty: &str, method: &str, params: Vec<ir::Expr>) -> ir::Expr {
    ir::Expr::FnCall {
        path: ir::FnPath::TypeMethod {
            ty: ty.to_string(),
            method: method.to_string(),
        },
        type_args: vec![],
        params,
    }
}

/// `data.<field>`.
fn data_field(name: &str) -> ir::Expr {
    ir::Expr::FieldAccess {
        base: Box::new(ir::Expr::Ident("data".to_string())),
        field: name.to_string(),
    }
}

/// A `Stmt::SlabCopy`, rendered as the offset-corrected copy loop
/// `dest[dest_offset + _i] = src[src_offset + _i]`.
fn slab_copy(
    src: ir::Expr,
    src_offset: ir::Expr,
    dest: ir::Expr,
    dest_offset: ir::Expr,
    size: ir::Expr,
) -> ir::Stmt {
    ir::Stmt::SlabCopy {
        src,
        src_offset,
        dest,
        dest_offset,
        size,
    }
}

/// `i += <size>;` — the running index advance after a field is walked.
fn advance_i(size: ir::Expr) -> ir::Stmt {
    ir::Stmt::CompoundAssignment {
        lhs: ir::Expr::Ident("i".to_string()),
        op: ir::CompoundOp::AddAssign,
        rhs: size,
    }
}

/// Serialize one field (or array element) into `dest` at the running
/// index `i`: a `to_array` call into a temp, an offset-corrected copy,
/// and an index advance.
fn to_array_field_stmts(
    item_ty: &str,
    value: ir::Expr,
    temp: &str,
    size: ir::Expr,
) -> Vec<ir::Stmt> {
    vec![
        ir::Stmt::Local(ir::Local {
            mutable: false,
            name: temp.to_string(),
            ty: None,
            init: Some(type_method_call(item_ty, "to_array", vec![value])),
        }),
        slab_copy(
            ir::Expr::Ident(temp.to_string()),
            lit_u32(0),
            ir::Expr::Ident("dest".to_string()),
            ir::Expr::Ident("i".to_string()),
            size.clone(),
        ),
        advance_i(size),
    ]
}

/// Deserialize one field (or array element) from `slab` at the running
/// index `i`: an `array_container` temp, an offset-corrected copy into
/// it, a `from_array` call binding the value, and an index advance.
fn from_array_field_stmts(item_ty: &str, temp: &str, bind: &str, size: ir::Expr) -> Vec<ir::Stmt> {
    vec![
        ir::Stmt::Local(ir::Local {
            mutable: true,
            name: temp.to_string(),
            ty: None,
            init: Some(type_method_call(item_ty, "array_container", vec![])),
        }),
        slab_copy(
            ir::Expr::Ident("slab".to_string()),
            ir::Expr::Ident("i".to_string()),
            ir::Expr::Ident(temp.to_string()),
            lit_u32(0),
            size.clone(),
        ),
        ir::Stmt::Local(ir::Local {
            mutable: false,
            name: bind.to_string(),
            ty: None,
            init: Some(type_method_call(
                item_ty,
                "from_array",
                vec![ir::Expr::Ident(temp.to_string())],
            )),
        }),
        advance_i(size),
    ]
}

/// The shared function scaffolding: no generics, no attributes, the
/// given name, inputs, return type, and body.
fn make_fn(
    name: &str,
    inputs: Vec<ir::FnArg>,
    return_type: ir::ReturnType,
    stmts: Vec<ir::Stmt>,
) -> ir::ItemFn {
    ir::ItemFn {
        type_params: vec![],
        const_params: vec![],
        fn_attrs: ir::FnAttrs::None,
        name: Cow::Owned(name.to_string()),
        inputs,
        return_type,
        block: ir::Block { stmts },
        attrs: vec![],
    }
}

// ===== Generated impl items =====

/// Build `fn array_container() -> [u32; N]` with a zero-value array
/// body (`array<u32, N>()`).
fn make_array_container(size_expr: &ir::Expr) -> ir::ItemFn {
    make_fn(
        "array_container",
        vec![],
        ir::ReturnType::Type {
            annotation: ir::ReturnTypeAnnotation::None,
            ty: slab_array_ty(size_expr),
        },
        vec![ir::Stmt::Expr {
            expr: ir::Expr::ZeroValueArray {
                elem_type: Box::new(ir::Type::Scalar(ir::ScalarType::U32)),
                len: Box::new(size_expr.clone()),
            },
            has_semi: false,
        }],
    )
}

/// Build `fn to_array(data: Self) -> [u32; N]`: zeroed `dest`, running
/// index `i`, one to_array/copy/advance triple per field — recursing
/// through struct fields' mangled methods and unrolling array fields
/// element-wise.
fn make_to_array(s: &ir::ItemStruct, size_expr: &ir::Expr) -> ir::ItemFn {
    let mut stmts = vec![
        ir::Stmt::Local(ir::Local {
            mutable: true,
            name: "dest".to_string(),
            ty: None,
            init: Some(type_method_call(&s.name, "array_container", vec![])),
        }),
        ir::Stmt::Local(ir::Local {
            mutable: true,
            name: "i".to_string(),
            ty: Some(ir::Type::Scalar(ir::ScalarType::U32)),
            init: Some(lit_u32(0)),
        }),
    ];

    for f in &s.fields {
        match &f.ty {
            ir::Type::Array { elem, len } => {
                let n = eval_array_len(len);
                let item_ty = slab_item_type_name(elem);
                let elem_size = field_size_expr(elem);
                for k in 0..n {
                    let value = ir::Expr::ArrayIndexing {
                        lhs: Box::new(data_field(&f.name)),
                        index: Box::new(lit_u32(k)),
                    };
                    stmts.extend(to_array_field_stmts(
                        &item_ty,
                        value,
                        &format!("{}_slab_{}", f.name, k),
                        elem_size.clone(),
                    ));
                }
            }
            _ => {
                let item_ty = slab_item_type_name(&f.ty);
                let size = field_size_expr(&f.ty);
                stmts.extend(to_array_field_stmts(
                    &item_ty,
                    data_field(&f.name),
                    &format!("{}_slab", f.name),
                    size,
                ));
            }
        }
    }

    stmts.push(ir::Stmt::Expr {
        expr: ir::Expr::Ident("dest".to_string()),
        has_semi: false,
    });

    make_fn(
        "to_array",
        vec![ir::FnArg {
            inter_stage_io: vec![],
            name: "data".to_string(),
            ty: ir::Type::Struct {
                name: s.name.clone(),
                type_args: vec![],
            },
            attrs: vec![],
        }],
        ir::ReturnType::Type {
            annotation: ir::ReturnTypeAnnotation::None,
            ty: slab_array_ty(size_expr),
        },
        stmts,
    )
}

/// Build `fn from_array(slab: [u32; N]) -> Self`: running index `i`,
/// one array_container/copy/from_array/advance quad per field, then a
/// struct construction. Array fields unroll element-wise and rebuild
/// with an array constructor expression.
fn make_from_array(s: &ir::ItemStruct, size_expr: &ir::Expr) -> ir::ItemFn {
    let mut stmts = vec![ir::Stmt::Local(ir::Local {
        mutable: true,
        name: "i".to_string(),
        ty: Some(ir::Type::Scalar(ir::ScalarType::U32)),
        init: Some(lit_u32(0)),
    })];

    let mut fields: Vec<ir::FieldValue> = vec![];
    for f in &s.fields {
        match &f.ty {
            ir::Type::Array { elem, len } => {
                let n = eval_array_len(len);
                let item_ty = slab_item_type_name(elem);
                let elem_size = field_size_expr(elem);
                let mut elems = vec![];
                for k in 0..n {
                    let temp = format!("{}_array_{}", f.name, k);
                    let bind = format!("{}_{}", f.name, k);
                    stmts.extend(from_array_field_stmts(
                        &item_ty,
                        &temp,
                        &bind,
                        elem_size.clone(),
                    ));
                    elems.push(ir::Expr::Ident(bind));
                }
                fields.push(ir::FieldValue {
                    member: f.name.clone(),
                    expr: ir::Expr::Array { elems },
                });
            }
            _ => {
                let item_ty = slab_item_type_name(&f.ty);
                let size = field_size_expr(&f.ty);
                stmts.extend(from_array_field_stmts(
                    &item_ty,
                    &format!("{}_array", f.name),
                    &f.name,
                    size,
                ));
                fields.push(ir::FieldValue {
                    member: f.name.clone(),
                    expr: ir::Expr::Ident(f.name.clone()),
                });
            }
        }
    }

    stmts.push(ir::Stmt::Expr {
        expr: ir::Expr::Struct {
            name: s.name.clone(),
            type_args: vec![],
            fields,
        },
        has_semi: false,
    });

    make_fn(
        "from_array",
        vec![ir::FnArg {
            inter_stage_io: vec![],
            name: "slab".to_string(),
            ty: slab_array_ty(size_expr),
            attrs: vec![],
        }],
        ir::ReturnType::Type {
            annotation: ir::ReturnTypeAnnotation::None,
            ty: ir::Type::Struct {
                name: s.name.clone(),
                type_args: vec![],
            },
        },
        stmts,
    )
}

/// Build the inherent `impl` block carrying `SLAB_SIZE`,
/// `array_container`, `to_array`, and `from_array`. WGSL has no traits;
/// rendering mangles these to `Foo__1SLAB_SIZE` / `Foo__1to_array` /
/// `Foo__1from_array` / `Foo__1array_container`.
fn make_slab_impl(s: &ir::ItemStruct) -> ir::ItemImpl {
    let size_expr = struct_size_expr(s);
    ir::ItemImpl {
        type_params: vec![],
        const_params: vec![],
        self_ty: s.name.clone(),
        items: vec![
            ir::ImplItem::Const(ir::ItemConst {
                name: "SLAB_SIZE".to_string(),
                ty: ir::Type::Scalar(ir::ScalarType::U32),
                expr: size_expr.clone(),
                attrs: vec![],
            }),
            ir::ImplItem::Fn(make_array_container(&size_expr)),
            ir::ImplItem::Fn(make_to_array(s, &size_expr)),
            ir::ImplItem::Fn(make_from_array(s, &size_expr)),
        ],
        attrs: vec![],
    }
}

/// Generate the slab impl block for each `#[derive(..., SlabItem)]`
/// struct in the module.
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
        module.items.push(ir::Item::Impl(make_slab_impl(s)));
    }
}

// ===== Stmt::Macro lowering =====

/// Lower all `slab_read!`/`slab_write!` `Stmt::Macro` invocations in the
/// module, including inside impl-block methods.
fn lower_slab_macros(module: &mut ir::Module) {
    for item in &mut module.items {
        match item {
            ir::Item::Fn(f) => lower_slab_macros_in_block(&mut f.block),
            ir::Item::Impl(i) => {
                for impl_item in &mut i.items {
                    if let ir::ImplItem::Fn(f) = impl_item {
                        lower_slab_macros_in_block(&mut f.block);
                    }
                }
            }
            _ => {}
        }
    }
}

/// Lower `Stmt::Macro` in a block, replacing them with IR statement
/// sequences.
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
        ir::Stmt::If(i) => lower_slab_macros_in_if(i),
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

/// Lower `Stmt::Macro` in an `if`/`else if` chain, recursing into both
/// branches.
fn lower_slab_macros_in_if(i: &mut ir::StmtIf) {
    lower_slab_macros_in_block(&mut i.then_block);
    match &mut i.else_branch {
        Some(ir::ElseBranch::Block(b)) => lower_slab_macros_in_block(b),
        Some(ir::ElseBranch::If(inner)) => lower_slab_macros_in_if(inner),
        None => {}
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

/// Parse a stringified offset argument. Integer literals (written `0`,
/// `0u32`, … on the CPU side, where WGSL's `0u` spelling is not valid
/// Rust) become u32 `Lit`s (rendered `0u`); anything else is emitted
/// verbatim as an identifier/expression.
fn parse_offset_arg(arg: &str) -> ir::Expr {
    let arg = arg.trim();
    let digits: String = arg.chars().take_while(char::is_ascii_digit).collect();
    let rest = &arg[digits.len()..];
    if !digits.is_empty() && matches!(rest, "" | "u" | "u32" | "usize") {
        ir::Expr::Lit(ir::Lit::Int {
            digits,
            suffix: "u32".to_string(),
        })
    } else {
        ir::Expr::Ident(arg.to_string())
    }
}

/// Parse `slab_read!(Type, slab, offset, dest)` args and lower to 3 IR
/// statements: a zeroed `array_container` temp, a `SlabCopy` from the
/// slab into it, and a `from_array` assignment into the caller-declared
/// destination (the destination variable's declaration is the caller's,
/// mirroring the CPU macro's assignment form).
///
/// The args string is the stringified token stream from the parser,
/// e.g. `"Foo , get ! ( SLAB ) , 0u , d"`. We re-parse it simply by
/// splitting on commas and trimming.
fn lower_slab_read(args: &str) -> Vec<ir::Stmt> {
    let parts: Vec<&str> = args.split(',').collect();
    if parts.len() < 4 {
        return vec![];
    }
    let type_name = parts[0].trim();
    let slab_expr = strip_get(parts[1].trim());
    let offset_expr = parse_offset_arg(parts[2]);
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
        ir::Stmt::SlabCopy {
            src: ir::Expr::Ident(slab_expr.to_string()),
            src_offset: offset_expr,
            dest: ir::Expr::Ident("slab_read_buf".to_string()),
            dest_offset: ir::Expr::Lit(ir::Lit::Int {
                digits: "0".to_string(),
                suffix: "u32".to_string(),
            }),
            size: ir::Expr::TypePath {
                ty: type_name.to_string(),
                member: "SLAB_SIZE".to_string(),
            },
        },
        ir::Stmt::Assignment {
            lhs: ir::Expr::Ident(dest_name.to_string()),
            rhs: ir::Expr::FnCall {
                path: ir::FnPath::TypeMethod {
                    ty: type_name.to_string(),
                    method: "from_array".to_string(),
                },
                type_args: vec![],
                params: vec![ir::Expr::Ident("slab_read_buf".to_string())],
            },
        },
    ]
}

/// Parse `slab_write!(Type, slab, offset, src)` args and lower to 2 IR
/// statements: a `to_array` temp, and a `SlabCopy` from it into the
/// slab.
fn lower_slab_write(args: &str) -> Vec<ir::Stmt> {
    let parts: Vec<&str> = args.split(',').collect();
    if parts.len() < 4 {
        return vec![];
    }
    let type_name = parts[0].trim();
    let slab_expr = strip_get(parts[1].trim());
    let offset_expr = parse_offset_arg(parts[2]);
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
        ir::Stmt::SlabCopy {
            src: ir::Expr::Ident("slab_write_out".to_string()),
            src_offset: ir::Expr::Lit(ir::Lit::Int {
                digits: "0".to_string(),
                suffix: "u32".to_string(),
            }),
            dest: ir::Expr::Ident(slab_expr.to_string()),
            dest_offset: offset_expr,
            size: ir::Expr::TypePath {
                ty: type_name.to_string(),
                member: "SLAB_SIZE".to_string(),
            },
        },
    ]
}

/// Strip `get!(IDENT)` or `get_mut!(IDENT)` from a stringified arg,
/// returning just the inner identifier — on the GPU side a storage
/// access is the bare variable name. Stringification can be glue-packed
/// (`get!(SLAB)`) or spaced (`get ! ( SLAB )`); whitespace is ignored
/// when matching the head.
fn strip_get(s: &str) -> &str {
    let s = s.trim();
    let Some(open) = s.find('(') else {
        return s;
    };
    let Some(close) = s.rfind(')') else {
        return s;
    };
    let head: String = s[..open].chars().filter(|c| !c.is_whitespace()).collect();
    let head = head.trim_end_matches('!');
    if head == "get" || head == "get_mut" {
        s[open + 1..close].trim()
    } else {
        s
    }
}
