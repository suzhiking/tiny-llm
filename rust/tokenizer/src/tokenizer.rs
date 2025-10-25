use std::{
    collections::HashMap,
    env,
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

use fancy_regex::Regex;

use crate::commons::{PAT, Pair, Token, Vocab, get_progress_bar, gpt2_bytes_to_unicode};

pub struct Tokenizer {
    vocab: Vocab,
    rev_vocab: HashMap<Token, u64>,
    merges: Vec<Pair>,
    special_tokens: Option<Vec<String>>,
}

use rayon::{
    iter::{IntoParallelIterator, ParallelIterator},
    slice::ParallelSlice,
};

impl Tokenizer {
    pub fn new(vocab: Vocab, merges: Vec<Pair>, special_tokens: Option<Vec<String>>) -> Self {
        let rev_vocab: HashMap<Token, u64> =
            vocab.clone().into_iter().map(|(x, tok)| (tok, x)).collect();
        Self {
            vocab,
            rev_vocab,
            merges,
            special_tokens,
        }
    }

    pub fn from_files(
        vocab_filepath: &Path,
        merges_filepath: &Path,
        special_tokens: Option<Vec<String>>,
    ) -> Self {
        let gpt2_byte_encoder: HashMap<u8, char> = gpt2_bytes_to_unicode();
        let gpt2_byte_decoder: HashMap<char, u8> = gpt2_byte_encoder
            .clone()
            .into_iter()
            .map(|(x, c)| (c, x))
            .collect();

        let vocab: Vocab = {
            let vocab_file = File::open(vocab_filepath).expect("Cannot open vocab file");
            let reader = BufReader::new(vocab_file);

            let raw_vocab: HashMap<String, u64> =
                serde_json::from_reader(reader).expect("Cannot parse vocab file");

            raw_vocab
                .into_iter()
                .map(|(s, x)| {
                    let bytes: Vec<u8> = s
                        .chars()
                        .map(|c| *gpt2_byte_decoder.get(&c).expect("Invalid GPT-2 byte char"))
                        .collect();
                    (x, bytes)
                })
                .collect()
        };

        let merges: Vec<Pair> = {
            let merges_file = File::open(merges_filepath).expect("Cannot open merge file");
            let reader = BufReader::new(merges_file);

            reader
                .lines()
                .map(|line_result| {
                    let line = line_result.expect("Merges file is not correctly formatted");
                    let (str1, str2) = line
                        .split_once(' ')
                        .expect("Merges file line not formatted correctly");
                    let tok1: Vec<u8> = str1
                        .chars()
                        .map(|c| *gpt2_byte_decoder.get(&c).expect("Invalid GPT-2 byte char"))
                        .collect();
                    let tok2: Vec<u8> = str2
                        .chars()
                        .map(|c| *gpt2_byte_decoder.get(&c).expect("Invalid GPT-2 byte char"))
                        .collect();

                    Pair {
                        left: tok1,
                        right: tok2,
                    }
                })
                .collect()
        };

        let rev_vocab: HashMap<Token, u64> =
            vocab.clone().into_iter().map(|(x, tok)| (tok, x)).collect();
        Self {
            vocab,
            rev_vocab,
            merges,
            special_tokens,
        }
    }

    pub fn merge(&self, token: String) -> Vec<u64> {
        let mut bytes: Vec<Vec<u8>> = token.into_bytes().into_iter().map(|b| vec![b]).collect();

        for pair in &self.merges {
            let mut new_bytes: Vec<Vec<u8>> = Vec::with_capacity(bytes.len());
            let mut i = 0;
            while i < bytes.len() {
                if i + 1 < bytes.len() && bytes[i] == pair.left && bytes[i + 1] == pair.right {
                    let mut merged = Vec::with_capacity(bytes[i].len() + bytes[i + 1].len());
                    merged.extend_from_slice(&bytes[i]);
                    merged.extend_from_slice(&bytes[i + 1]);
                    new_bytes.push(merged);
                    i += 2;
                } else {
                    new_bytes.push(bytes[i].clone());
                    i += 1;
                }
            }
            bytes = new_bytes;
        }

        bytes
            .into_iter()
            .map(|tok| {
                *self
                    .rev_vocab
                    .get(&tok)
                    .expect(format!("Invalid token: {:?}", tok).as_ref())
            })
            .collect()
    }

    pub fn encode(&self, text: &str) -> Vec<u64> {
        let special_pat = if let Some(special_tokens) = &self.special_tokens {
            Some(
                Regex::new(
                    &special_tokens
                        .iter()
                        .map(|t| regex::escape(t))
                        .collect::<Vec<_>>()
                        .join("|"),
                )
                .expect("Invalid special token regex"),
            )
        } else {
            None
        };

        let pat = Regex::new(PAT).expect("Invalid regex pattern");

        let mut pretokens: Vec<&str> = Vec::new();

        if let Some(special_pat) = &special_pat {
            let mut last_end = 0;

            for m in special_pat.find_iter(text) {
                let m = m.expect("Invalid match");
                if m.start() > last_end {
                    let normal_segment = &text[last_end..m.start()];
                    pretokens.extend(
                        pat.find_iter(normal_segment)
                            .map(|m| m.expect("Invalid match").as_str()),
                    )
                }
                last_end = m.end();
            }

            if last_end < text.len() {
                let normal_segment = &text[last_end..];
                pretokens.extend(
                    pat.find_iter(normal_segment)
                        .map(|m| m.expect("Invalid match").as_str()),
                )
            }
        } else {
            pretokens.extend(
                pat.find_iter(text)
                    .map(|m| m.expect("Invalid match").as_str()),
            )
        }

        let pbar = get_progress_bar(pretokens.len() as u64, "Merging pretokens");
        pretokens
            .into_par_iter()
            .flat_map(move |s| {
                let res = {
                    let out = self.merge(s.to_string());
                    pbar.inc(1);
                    out
                };
                res
            })
            .collect()
    }

    pub fn encode_iterable(&self, iterable: impl Iterator<Item = impl AsRef<str>>) -> Vec<u64> {
        iterable
            .into_iter()
            .flat_map(|s| self.encode(s.as_ref()))
            .collect()
    }
}
