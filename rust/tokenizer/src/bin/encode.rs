use anyhow::{Context, Result};
use clap::Parser;
use itertools::{Itertools, enumerate};
use npyz::WriterBuilder;
use rayon::ThreadPoolBuilder;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use tokenizer::tokenizer::Tokenizer;
// use npyz::npz::NpzWriter

/// CLI args (matches the Python defaults)
#[derive(Parser, Debug)]
struct Args {
    /// Tokenizer directory name under tokenizer_data/
    #[arg(long, default_value = "tinystories")]
    tokenizer: String,

    /// Corpus filename under corpus/
    #[arg(long, default_value = "TinyStoriesV2-GPT4-valid.txt")]
    corpus: String,

    /// Output directory
    #[arg(long, default_value = "~/Courses/CS336/assignment1-basics/data/encoded_data/tinystories-valid")]
    output_directory: String,
}

fn expand_user(p: &str) -> PathBuf {
    PathBuf::from(shellexpand::tilde(p).to_string())
}

fn main() -> Result<()> {
    ThreadPoolBuilder::new()
        .num_threads(12) // or read from env/flag
        .build_global()
        .unwrap();

    let args = Args::parse();

    // Replicates: ~/Courses/CS336/assignment1-basics/data/
    let data_root_path = expand_user("~/Courses/CS336/assignment1-basics/data/");
    let corpus = data_root_path.join("corpus").join(&args.corpus);
    let tokenizer_data_dir = data_root_path.join("tokenizer_data").join(&args.tokenizer);

    // --- Load tokenizer (expects train-bpe-vocab.json & train-bpe-merges.json) ---
    // Replace this block with your actual tokenizer module path.
    let tokenizer = Tokenizer::from_files(
        &tokenizer_data_dir.join("vocab.json"),
        &tokenizer_data_dir.join("merges.txt"),
        Some(vec!["<|endoftext|>".to_string()]),
    );

    // --- Read corpus ---
    let texts: Vec<String> = fs::read_to_string(&corpus)
        .with_context(|| format!("Failed to read corpus file: {}", corpus.display()))?
        .chars()
        .collect::<Vec<_>>()
        .chunks(1e8 as usize)
        .map(|c| c.iter().collect())
        .collect();

    // --- Encode ---
    let out_dir = data_root_path.join("encoded_data");
    for (i, text) in enumerate(texts) {
        let tokens: Vec<u64> = tokenizer.encode(&text);

        // --- Save as .npy ---
        fs::create_dir_all(&out_dir)
            .with_context(|| format!("Failed to create output dir: {}", out_dir.display()))?;

        let out_path = out_dir.join(format!("encoded_tinystories_{}.npy", i));
        let f = File::create(&out_path)
            .with_context(|| format!("Failed to create output file: {}", out_path.display()))?;
        let mut buf = BufWriter::new(f);

        // Write a 1D .npy array of u64 using npyz
        let mut npy = npyz::WriteOptions::new()
            .default_dtype() // infer dtype from the Rust element type
            .shape(&[tokens.len() as u64]) // 1D shape
            .writer(&mut buf) // any io::Write
            .begin_nd()?; // start writing

        // You can pass a slice; extend() accepts an iterator of items.
        npy.extend(tokens.iter())?; // or: npy.extend(&tokens)?
        npy.finish()?; // finalize header/footer
        buf.flush()?;

        println!("Saved encoded data to {}", out_path.display());
    }
    Ok(())
}
