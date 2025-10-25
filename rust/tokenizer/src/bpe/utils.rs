use indexmap::IndexMap;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::{
    collections::HashMap,
    fs::File,
    io::{Read, Result, Seek, SeekFrom},
    path::Path,
};

use crate::commons::PAT;
use regex::Regex;
use memchr::memmem;

fn count_pretokens(
    start_end: &[(u64, u64)],
    special_tokens: &[&str],
    input_path: &Path,
) -> HashMap<String, u64> {
    let mut result: HashMap<String, u64> = HashMap::new();
    let mut fd = File::open(input_path).expect("Error when opening file");

    let escaped_tokens: Vec<String> = special_tokens
        .iter()
        .map(|&t| regex::escape(t).into())
        .collect();
    let pattern = escaped_tokens.join("|");
    let re = Regex::new(&pattern).unwrap();
    for &(start, end) in start_end {
        let _ = fd.seek(SeekFrom::Start(start));
        let mut content = vec![0u8; (end - start) as usize];
        let n = fd.read(&mut content).expect("Error when reading file");
        content.truncate(n);

        let content_str = std::str::from_utf8(&content).expect("File content is not valid UTF-8");

        let chunks: Vec<&str> = re.split(content_str).collect();

        // println!("{}", chunks.len());
        for chunk in chunks {
            let re = Regex::new(PAT).unwrap();
            for mtch in Regex::find_iter(&re, chunk) {
                let matched = mtch.as_str();
                *result.entry(matched.to_string()).or_insert(0) += 1;
            }
        }
    }

    result
}

pub fn find_chunk_boundaries(
    file: &mut File,
    desired_num_chunks: u64,
    split_special_token: &[u8],
) -> Result<Vec<u64>> {
    assert!(desired_num_chunks > 0);

    let meta_data = file.metadata()?;
    let file_size = meta_data.len();

    let chunk_size = file_size / desired_num_chunks;

    let mut chunk_boundaries: Vec<u64> = (0..=desired_num_chunks).map(|i| i * chunk_size).collect();
    let len = chunk_boundaries.len();
    chunk_boundaries[len - 1] = file_size;

    let mini_chunk_size = 4096;

    for boundary in chunk_boundaries.iter_mut().take(len - 1).skip(1) {
        let mut initial_position = *boundary;
        let _ = file.seek(SeekFrom::Start(initial_position));

        loop {
            let mut mini_chunk = vec![0u8; mini_chunk_size];
            let n = file.read(&mut mini_chunk)?;
            mini_chunk.truncate(n);

            if mini_chunk.is_empty() {
                *boundary = file_size;
                break;
            }

            if let Some(found_at) = memmem::find(&mini_chunk, split_special_token) {
                *boundary = initial_position + found_at as u64;
                break;
            }

            initial_position += mini_chunk_size as u64;
        }
    }

    chunk_boundaries.sort();
    chunk_boundaries.dedup();

    Ok(chunk_boundaries)
}

pub fn pretokenize(input_path: &Path, special_tokens: &[&str]) -> HashMap<String, u64> {
    let mut fd = File::open(input_path).expect("Failed to open file");
    // let metadata = fd.metadata().expect("Failed to get metadata of file");
    let num_workers = 10;
    let split_special_token = b"<|endoftext|>";
    let boundaries = find_chunk_boundaries(&mut fd, num_workers * 10000, split_special_token)
        .expect("Failed to find boundaries");
    let start_end: Vec<(u64, u64)> = boundaries.windows(2).map(|w| (w[0], w[1])).collect();
    println!("{}", start_end.len());

    let index_chunks: Vec<Vec<(u64, u64)>> = (0usize..(num_workers as usize))
        .map(|i| {
            start_end
                .iter()
                .cloned()
                .skip(i)
                .step_by(num_workers as usize)
                .collect()
        })
        .collect();
    let results: Vec<HashMap<String, u64>> = index_chunks
        .par_iter()
        .map(|ranges| count_pretokens(ranges, special_tokens, input_path))
        .collect();

    println!("Done");
    let mut total: HashMap<String, u64> = HashMap::new();
    for partial in results {
        for (k, v) in partial {
            *total.entry(k).or_insert(0) += v;
        }
    }

    total
}
