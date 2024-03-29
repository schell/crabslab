//! This program builds the rust-gpu shaders and writes the spv files into the
//! main source repo.
use clap::Parser;
use spirv_builder::{CompileResult, MetadataPrintout, ModuleResult, SpirvBuilder};

mod shader;

#[derive(Debug, Parser)]
#[clap(author, version, about)]
struct Cli {
    /// Sets the verbosity level
    #[clap(short, action = clap::ArgAction::Count)]
    verbosity: u8,

    /// Directory containing the shader crate to compile.
    #[clap(long, short, default_value = "example")]
    shader_crate: std::path::PathBuf,

    /// Set cargo default-features.
    #[clap(long)]
    no_default_features: bool,

    /// Set cargo features.
    #[clap(long)]
    features: Vec<String>,

    ///// Cargo features to be passed to the shader crate compilation invocation.
    //#[clap(long, short)]
    //features: Vec<String>,
    /// Path to the output directory for the compiled shaders.
    #[clap(long, short, default_value = "..")]
    output_dir: std::path::PathBuf,

    /// If set the shaders will be compiled but not put into place.
    #[clap(long, short)]
    dry_run: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    println!("options: {cli:#?}");

    let Cli {
        verbosity,
        shader_crate,
        no_default_features,
        features,
        output_dir,
        dry_run,
    } = Cli::parse();
    let level = match verbosity {
        0 => log::LevelFilter::Warn,
        1 => log::LevelFilter::Info,
        2 => log::LevelFilter::Debug,
        _ => log::LevelFilter::Trace,
    };
    env_logger::builder()
        .filter_level(level)
        .try_init()
        .unwrap();

    std::fs::create_dir_all(&output_dir).unwrap();

    let shader_crate = std::path::Path::new("../crates/").join(shader_crate);
    assert!(
        shader_crate.exists(),
        "shader crate '{}' does not exist. Current dir is {}",
        shader_crate.display(),
        std::env::current_dir().unwrap().display()
    );

    let start = std::time::Instant::now();

    let CompileResult {
        entry_points,
        module,
    } = {
        let mut builder = SpirvBuilder::new(shader_crate, "spirv-unknown-vulkan1.2")
            .print_metadata(MetadataPrintout::None)
            .multimodule(true);

        if no_default_features {
            builder = builder.shader_crate_default_features(false);
        }
        if !features.is_empty() {
            builder = builder.shader_crate_features(features);
        }

        builder.build()?
    };

    let dir = output_dir;
    let mut shaders = vec![];
    match module {
        ModuleResult::MultiModule(modules) => {
            assert!(!modules.is_empty(), "No shader modules to compile");
            for (entry, filepath) in modules.into_iter() {
                let path = dir.join(filepath.file_name().unwrap());
                let metadata = std::fs::metadata(&filepath).unwrap();
                println!("{entry} is {}bytes", metadata.len());
                if !dry_run {
                    std::fs::copy(filepath, &path).unwrap();
                }
                shaders.push((entry, path));
            }
        }
        ModuleResult::SingleModule(filepath) => {
            let path = dir.join(filepath.file_name().unwrap());
            if !dry_run {
                std::fs::copy(filepath, &path).unwrap();
            }
            for entry in entry_points {
                shaders.push((entry, path.clone()));
            }
        }
    }

    let tokens = shader::gen_all_shaders(shaders);
    let tokens = syn::parse_file(&tokens).unwrap_or_else(|e| {
        panic!(
            "Failed to parse generated shader.rs: {}\n\n{}",
            e,
            tokens.to_string()
        )
    });
    let tokens = prettyplease::unparse(&tokens);
    if dry_run {
        println!("\nNot writing shaders.rs because --dry-run was set.\n");
    } else {
        println!("\nWriting shaders.rs to {}", dir.display());
    };
    let tokens = tokens.to_string();
    if !dry_run {
        std::fs::write(dir.join("shaders.rs"), tokens).unwrap();
    }

    let end = std::time::Instant::now();
    println!("Finished in {:?}", (end - start));

    Ok(())
}
