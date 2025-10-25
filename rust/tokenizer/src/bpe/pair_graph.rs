use std::collections::{BTreeSet, HashMap};

use indexmap::IndexMap;

use crate::commons::{Pair, Token};

pub type OccId = u64;

#[derive(Debug)]
pub struct Occurrence {
    pub pair: Pair,
    pub count: u64,
    pub prev: Option<OccId>,
    pub next: Option<OccId>,
}

#[derive(Debug)]
pub struct PairGraph {
    occs: HashMap<OccId, Occurrence>,
    pair_to_occs: HashMap<Pair, BTreeSet<OccId>>,
    next_occ_id: OccId,
}

#[derive(Debug)]
pub struct SpliceDelta {
    pub left: Option<(OccId, Pair /*old*/, Pair /*new*/, u64 /*by*/)>,
    pub right: Option<(OccId, Pair /*old*/, Pair /*new*/, u64 /*by*/)>,
}

#[derive(thiserror::Error, Debug)]
pub enum GraphError {
    #[error("pair {0:?} not found")]
    PairNotFound(Pair),
    #[error("occurence {0:?} not found")]
    OccNotFound(OccId),
}

impl PairGraph {
    pub fn new() -> Self {
        Self {
            occs: HashMap::new(),
            pair_to_occs: HashMap::new(),
            next_occ_id: 1,
        }
    }

    pub fn add_occurence(
        &mut self,
        pair: Pair,
        count: u64,
        prev: Option<OccId>,
        next: Option<OccId>,
    ) -> OccId {
        let occ_id = self.next_occ_id;
        self.next_occ_id += 1;
        self.pair_to_occs
            .entry(pair.clone())
            .or_default()
            .insert(occ_id);
        self.occs.insert(
            occ_id,
            Occurrence {
                pair,
                count,
                prev,
                next,
            },
        );

        occ_id
    }

    pub fn get(&self, id: OccId) -> Result<&Occurrence, GraphError> {
        if let Some(occ) = self.occs.get(&id) {
            return Ok(occ);
        }

        Err(GraphError::OccNotFound(id))
    }

    pub fn get_mut(&mut self, id: OccId) -> Result<&mut Occurrence, GraphError> {
        if let Some(occ) = self.occs.get_mut(&id) {
            return Ok(occ);
        }

        Err(GraphError::OccNotFound(id))
    }

    pub fn occurrence_of(&self, pair: &Pair) -> Result<&BTreeSet<u64>, GraphError> {
        if let Some(occs) = self.pair_to_occs.get(pair) {
            return Ok(occs);
        }

        Err(GraphError::PairNotFound(pair.clone()))
    }

    pub fn mut_occurrence_of(&mut self, pair: &Pair) -> &mut BTreeSet<u64> {
        self.pair_to_occs.entry(pair.clone()).or_default()
    }

    pub fn neighbors(&self, id: OccId) -> Result<(Option<OccId>, Option<OccId>), GraphError> {
        let occ = self.get(id)?;

        Ok((occ.prev, occ.next))
    }

    pub fn unlink_middle(
        &mut self,
        left_id: OccId,
        mid_id: OccId,
        right_id: OccId,
    ) -> Option<(Pair, Pair)> {
        todo!()
    }

    pub fn retarget_left(&mut self, left_id: OccId, merged: &Token) -> Result<Pair, GraphError> {
        let occ = self.get_mut(left_id)?;
        occ.pair.right = merged.clone();

        Ok(occ.pair.clone())
    }

    pub fn retarget_right(&mut self, right_id: OccId, merged: &Token) -> Result<Pair, GraphError> {
        let occ = self.get_mut(right_id)?;
        occ.pair.left = merged.clone();

        Ok(occ.pair.clone())
    }

    pub fn splice_and_retarget(
        &mut self,
        mid: OccId,
        merged_token: &Token,
    ) -> Result<SpliceDelta, GraphError> {
        // Remove this occurrence from its pair's index (we iterate over a cloned set upstream)
        let mid_pair = self.get(mid)?.pair.clone();
        if let Some(set) = self.pair_to_occs.get_mut(&mid_pair) {
            set.remove(&mid);
        }

        let (left_id, right_id) = self.neighbors(mid)?;

        if let (Some(l), Some(r)) = (left_id, right_id) {
            self.occs.get_mut(&l).unwrap().next = Some(r);
            self.occs.get_mut(&r).unwrap().prev = Some(l);

            let left_old = self.occs.get(&l).unwrap().pair.clone();
            let right_old = self.occs.get(&r).unwrap().pair.clone();

            let left_pair_occs = self.mut_occurrence_of(&left_old);
            left_pair_occs.remove(&l);
            let right_pair_occs = self.mut_occurrence_of(&right_old);
            right_pair_occs.remove(&r);

            // retarge neighbors to use merged token
            let left_new = self.retarget_left(l, merged_token)?;
            let right_new = self.retarget_right(r, merged_token)?;
            let left_pair_occs = self.mut_occurrence_of(&left_new);
            left_pair_occs.insert(l);
            let right_pair_occs = self.mut_occurrence_of(&right_new);
            right_pair_occs.insert(r);

            let by_left = self.occs.get(&l).unwrap().count;
            let by_right = self.occs.get(&r).unwrap().count;

            Ok(SpliceDelta {
                left: Some((l, left_old, left_new, by_left)),
                right: Some((r, right_old, right_new, by_right)),
            })
        } else if let Some(l) = left_id {
            self.occs.get_mut(&l).unwrap().next = None;

            let left_old = self.occs.get(&l).unwrap().pair.clone();
            let left_pair_occs = self.mut_occurrence_of(&left_old);
            left_pair_occs.remove(&l);

            let left_new = self.retarget_left(l, merged_token)?;
            let left_pair_occs = self.mut_occurrence_of(&left_new);
            left_pair_occs.insert(l);

            let by_left = self.occs.get(&l).unwrap().count;

            Ok(SpliceDelta {
                left: Some((l, left_old, left_new, by_left)),
                right: None,
            })
        } else if let Some(r) = right_id {
            self.occs.get_mut(&r).unwrap().prev = None;

            let right_old = self.occs.get(&r).unwrap().pair.clone();
            let right_pair_occs = self.mut_occurrence_of(&right_old);
            right_pair_occs.remove(&r);

            let right_new = self.retarget_right(r, merged_token)?;
            let right_pair_occs = self.mut_occurrence_of(&right_new);
            right_pair_occs.insert(r);

            let by_right = self.occs.get(&r).unwrap().count;

            Ok(SpliceDelta {
                left: None,
                right: Some((r, right_old, right_new, by_right)),
            })
        } else {
            Ok(SpliceDelta {
                left: None,
                right: None,
            })
        }
    }

    pub fn delete_pair_index(&mut self, pair: &Pair) {
        if let Some(ids) = self.pair_to_occs.remove(pair) {
            debug_assert!(
                ids.is_empty(),
                "delete_pair_index: expected no remaining occurrences for {:?}, found {ids:?}",
                pair
            );
        }
    }
}
