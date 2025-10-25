mod pair_graph;
mod pair_priority;
mod utils;

use std::{
    collections::HashMap,
    fs::File,
    io::{BufWriter, Write},
    path::Path,
};

use indexmap::IndexMap;
use itertools::Itertools;
use serde::Serialize;
use serde_json::{Map, Serializer, Value, ser::PrettyFormatter};

use crate::{
    bpe::{
        pair_graph::{GraphError, OccId, PairGraph},
        pair_priority::PairPriority,
        utils::pretokenize,
    },
    commons::{Pair, Token, Vocab, get_progress_bar, gpt2_bytes_to_unicode},
};

#[derive(thiserror::Error, Debug)]
pub enum BpeError {
    #[error("invalid configuration: {0}")]
    InvalidConfig(&'static str),
    #[error("heap underflow")]
    HeapEmpty,
    #[error("pair not found")]
    PairNotFound,
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("graph error")]
    Graph(GraphError),
}

impl From<GraphError> for BpeError {
    fn from(value: GraphError) -> Self {
        BpeError::Graph(value)
    }
}

pub struct BPETrainer {
    graph: PairGraph,
    priority: PairPriority,
    vocab: Vocab,
    merges: Vec<Pair>,
    occ_cnt: HashMap<OccId, u64>, // Fixed count
    vocab_size: u64,
    next_vocab_id: u64,
}

impl BPETrainer {
    pub fn new(input_path: &Path, vocab_size: u64, special_tokens: Vec<&str>) -> Self {
        assert!(256 + special_tokens.len() < vocab_size as usize);
        let mut pair_graph = PairGraph::new();
        let mut pair_priority = PairPriority::new();
        let mut occ_cnt = HashMap::new();
        let mut vocab = HashMap::new();

        // Intialize vocab
        for i in 0u64..256 {
            vocab.insert(i, vec![i as u8]);
        }
        let mut next_vocab_id = 256;
        for &tok in &special_tokens {
            vocab.insert(next_vocab_id, tok.to_string().into_bytes());
            next_vocab_id += 1;
        }

        let pretokens = pretokenize(input_path, &special_tokens);
        let mut pair_cnt: HashMap<Pair, u64> = HashMap::new();

        for (s, cnt) in pretokens {
            let mut prev_pair = None;
            let bs = s.as_bytes();
            for w in bs.windows(2) {
                let pair_to_add = Pair {
                    left: vec![w[0]],
                    right: vec![w[1]],
                };
                let occ_id = pair_graph.add_occurence(pair_to_add.clone(), cnt, prev_pair, None);
                *pair_cnt.entry(pair_to_add).or_default() += cnt;
                occ_cnt.insert(occ_id, cnt);

                if let Some(prev_occ_id) = prev_pair {
                    if let Ok(prev_occ) = pair_graph.get_mut(prev_occ_id) {
                        prev_occ.next = Some(occ_id)
                    }
                }

                prev_pair = Some(occ_id);
            }
        }

        for (pair, cnt) in pair_cnt {
            pair_priority.increment(&pair, cnt);
        }

        Self {
            graph: pair_graph,
            priority: pair_priority,
            vocab,
            merges: vec![],
            occ_cnt,
            vocab_size,
            next_vocab_id,
        }
    }

    /// One merge step: picks best pair, applies it across occurrences, updates counts/heap/graph/vocab.
    /// Returns the merged Pair and the new merged Token bytes.
    pub fn step(&mut self) -> Result<(Pair, Token), BpeError> {
        let heap_item = self.priority.pop_valid()?;
        let pair_to_merge = heap_item.pair.clone();
        let mut merged_tok =
            Vec::with_capacity(pair_to_merge.left.len() + pair_to_merge.right.len());
        if self.next_vocab_id == 262 {
            for (pair, cnt) in self.priority.counts.clone() {
                let occs = self.graph.occurrence_of(&pair)?;
                let mut res = 0;
                for occ_id in occs {
                    res += *self.occ_cnt.get(occ_id).expect("");
                }
                assert!(res == cnt, "{:?}: {} != {}, with {:?}", pair, cnt, res, occs);
            }
            let mut h = self.priority.clone();
            // println!("{:?}", self.priority.counts);
            println!("{:?}", heap_item);
            println!("{:?}", h.pop_valid());
            // println!("{:?}", self.priority.heap(occ_id));
        }
        merged_tok.extend_from_slice(&pair_to_merge.left);
        merged_tok.extend_from_slice(&pair_to_merge.right);

        // iterate occurrence ids deterministically
        let occs = self.graph.occurrence_of(&pair_to_merge)?.clone();

        for occ_id in occs {
            if self.graph.get(occ_id)?.pair != pair_to_merge {
                continue;
            }
            let delta = self.graph.splice_and_retarget(occ_id, &mut merged_tok)?;

            let mut apply = |side: &Option<(OccId, Pair, Pair, u64)>| {
                if let Some((id, old_pair, new_pair, by)) = side {
                    let &rep_cnt = self.occ_cnt.get(id).expect("Cannot get pair rep count");
                    self.priority.decrement(old_pair, *by);
                    self.priority.increment(new_pair, *by);
                    // if self.priority.current_count(old_pair) == 0 {
                    //     self.priority.delete(old_pair);
                    //     // self.graph.delete_pair_index(old_pair);
                    // }
                }
            };

            apply(&delta.left);
            apply(&delta.right);
        }

        // All occurrences of the merged pair have been removed from its index above; drop the index entirely.
        self.graph.delete_pair_index(&pair_to_merge);
        self.priority.delete(&pair_to_merge);
        self.vocab.insert(self.next_vocab_id, merged_tok.clone());
        self.next_vocab_id += 1;
        self.merges.push(pair_to_merge.clone());

        Ok((pair_to_merge, merged_tok))
    }

    /// Runs until vocab reaches the limit (or no valid pairs remain).
    pub fn run_to_limit(&mut self) {
        let pbar = get_progress_bar(self.vocab_size - self.next_vocab_id, "Merging tokens");
        for _ in self.next_vocab_id..self.vocab_size {
            let (pair, bs) = self.step().expect("Invalid step");
            pbar.inc(1);
        }
        pbar.finish();
    }

    pub fn save(&self, output_path: &Path) {
        let vocab_output_path = output_path.join("vocab.json");
        let merges_output_path = output_path.join("merges.txt");

        let gpt2_byte_encoder: HashMap<u8, char> = gpt2_bytes_to_unicode();

        let mut raw_vocab: Vec<(String, u64)> = self
            .vocab
            .iter()
            .map(|(&x, bs)| {
                let s: String = bs
                    .iter()
                    .map(|b| gpt2_byte_encoder.get(b).expect("Invalid byte"))
                    .collect();

                (s, x)
            })
            .collect();
        raw_vocab.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        // println!("{:?}", raw_vocab);

        let mut ordered = IndexMap::new();
        for (k, v) in raw_vocab {
            ordered.insert(k, v);
        }

        let vocab_file = File::create(vocab_output_path).expect("Cannot create the file");
        let vocab_writer = BufWriter::new(vocab_file);
        let formatter = PrettyFormatter::with_indent(b"    ");
        let mut serializer = Serializer::with_formatter(vocab_writer, formatter);
        let _ = ordered.serialize(&mut serializer);

        let merges_file = File::create(merges_output_path).expect("Cannot create the file");
        let mut merges_writer = BufWriter::new(merges_file);
        for pair in &self.merges {
            let tok1: String = pair
                .left
                .iter()
                .map(|b| gpt2_byte_encoder.get(b).expect("Invalid byte"))
                .collect();
            let tok2: String = pair
                .right
                .iter()
                .map(|b| gpt2_byte_encoder.get(b).expect("Invalid byte"))
                .collect();

            writeln!(merges_writer, "{} {}", tok1, tok2).expect("Cannot save merges");
        }
    }

    pub fn vocab(&self) -> &Vocab {
        &self.vocab
    }
    pub fn merges(&self) -> &Vec<Pair> {
        &self.merges
    }
    pub fn graph(&self) -> &PairGraph {
        &self.graph
    }
    pub fn priority(&self) -> &PairPriority {
        &self.priority
    }
}
