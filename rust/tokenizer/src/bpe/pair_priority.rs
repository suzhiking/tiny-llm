use indexmap::IndexMap;

use crate::bpe::BpeError;
use crate::commons::Pair;
use std::collections::BinaryHeap;
use std::collections::HashMap;

#[derive(Debug, Eq, Clone)]
pub struct HeapItem {
    pub count: u64,
    pub pair: Pair,
}

impl HeapItem {
    fn new(count: u64, pair: Pair) -> Self {
        Self { count, pair }
    }
}

impl PartialEq for HeapItem {
    fn eq(&self, other: &Self) -> bool {
        self.count == other.count && self.pair == other.pair
    }
}

impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.count
            .cmp(&other.count)
            .then_with(|| self.pair.left.cmp(&other.pair.left))
            .then_with(|| self.pair.right.cmp(&other.pair.right))
    }
}

#[derive(Clone, Debug)]
pub struct PairPriority {
    pub counts: HashMap<Pair, u64>,
    pub heap: BinaryHeap<HeapItem>,
}

impl PairPriority {
    pub fn new() -> Self {
        Self {
            counts: HashMap::new(),
            heap: BinaryHeap::new(),
        }
    }

    pub fn current_count(&self, pair: &Pair) -> u64 {
        *self.counts.get(pair).unwrap_or(&0)
    }

    pub fn set(&mut self, pair: Pair, count: u64) {
        self.counts.insert(pair.clone(), count);
        self.heap.push(HeapItem { count, pair });
    }

    pub fn increment(&mut self, pair: &Pair, by: u64) {
        let c = self.current_count(pair) + by;
        self.set(pair.clone(), c);
    }

    pub fn decrement(&mut self, pair: &Pair, by: u64) {
        let cur = self.current_count(pair);
        let c = cur.saturating_sub(by);
        if c == 0 {
            self.delete(pair);
        } else {
            self.set(pair.clone(), c);
        }
    }

    pub fn delete(&mut self, pair: &Pair) {
        self.counts.remove(pair);
        // lazy removal from heap; pop_valid will skip stale tops
    }

    /// Pops until the top matches the authoritative count.
    pub fn pop_valid(&mut self) -> Result<HeapItem, BpeError> {
        while let Some(HeapItem { count, pair }) = self.heap.pop() {
            if self.current_count(&pair) == count && count > 0 {
                return Ok(HeapItem { count, pair });
            }
        }
        Err(BpeError::HeapEmpty)
    }
}
