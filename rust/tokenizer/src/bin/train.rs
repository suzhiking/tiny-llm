use std::path::PathBuf;

use clap::Parser;
use tokenizer::bpe::BPETrainer;

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// Input path of corpus text
    input_path: PathBuf,

    /// Output directory for trained tokenizer data (optional)
    #[arg(short = 'o', long = "output")]
    output_path: Option<PathBuf>,

    /// Maximum vocabulary size
    #[arg(long = "vocab_size", default_value_t = 256)]
    vocab_size: u64,

    /// Example: --special-token "<|endoftext|>" --special-token "<|pad|>"
    #[arg(long = "special_tokens", num_args = 0.., default_values_t = vec!["<|endoftext|>".to_string()])]
    special_tokens: Vec<String>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Convert Vec<String> → Vec<&str> because our trainer API took &[&str]
    let specials_storage: Vec<String> = args.special_tokens;
    let special_tokens: Vec<&str> = specials_storage.iter().map(|s| s.as_str()).collect();
    let vocab_size = args.vocab_size;
    let input_path = args.input_path;

    // Build trainer
    let mut trainer = BPETrainer::new(&input_path, vocab_size, special_tokens);

    trainer.run_to_limit();
    if let Some(path) = args.output_path {
        trainer.save(&path);
    }

    Ok(())
}
