use std::{borrow::Cow, collections::HashMap, fmt::Debug};

use indexmap::IndexMap;
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};

pub const PAT: &str = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+\z|\s+";

pub type Token = Vec<u8>;
#[derive(Clone, Eq, PartialEq, PartialOrd, Ord, Hash)]
pub struct Pair {
    pub left: Token,
    pub right: Token,
}

impl Debug for Pair {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let encoder = gpt2_bytes_to_unicode();
        let ref left_str: String = self.left.clone().iter().map(|b| encoder.get(b).unwrap()).collect();
        let ref right_str: String = self.right.clone().iter().map(|b| encoder.get(b).unwrap()).collect();
        f.write_str(&format!("({}, {})", left_str, right_str))
    }
}
pub type Vocab = HashMap<u64, Token>;

/*  Returns a mapping between every possible byte (an integer from 0 to 255) to a
*    printable unicode string character representation. This function is taken
*
*   As a result, this function returns a dictionary `d` where `d[0]` returns `Ā`.
*   The bytes that are visually printable keep their original string representation [1].
*   For example, `chr(33)` returns `!`, and so accordingly `d[33]` returns `!`.
*   Note in particular that the space character `chr(32)` becomes `d[32]`, which
*   returns 'Ġ'.
*
*   For unprintable characters, the function shifts takes the integer representing
*   the Unicode code point of that character (returned by the Python `ord`) function
*   and shifts it by 256. For example, `ord(" ")` returns `32`, so the the space character
*   ' ' is shifted to `256 + 32`. Since `chr(256 + 32)` returns `Ġ`, we use that as the
*   string representation of the space.
*
*   This function can simplify the BPE implementation and makes it slightly easier to
*   manually inspect the generated merges after they're serialized to a file.
* */
pub fn gpt2_bytes_to_unicode() -> HashMap<u8, char> {
    let mut bs = {
        let mut v1: Vec<u32> = (('!' as u32)..('~' as u32) + 1).collect();
        let mut v2: Vec<u32> = (('¡' as u32)..('¬' as u32) + 1).collect();
        let mut v3: Vec<u32> = (('®' as u32)..('ÿ' as u32) + 1).collect();

        v1.append(&mut v2);
        v1.append(&mut v3);

        v1
    };
    let mut cs = bs.clone();

    let mut n: u32 = 0;
    for b in 0..(1 << 8) {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push((1 << 8) + n);
            n += 1
        }
    }

    let dict: HashMap<u8, char> = bs
        .into_iter()
        .zip(cs)
        .map(|(x1, x2)| (x1 as u8, char::from_u32(x2).unwrap()))
        .collect();

    dict
}

pub fn get_progress_bar(n: u64, msg: impl Into<Cow<'static, str>>) -> ProgressBar {
    let pbar = ProgressBar::new(n);
    pbar.set_style(ProgressStyle::with_template("{spinner:.green} {msg} [{wide_bar:.cyan/blue}] {pos}/{len} ({percent}%) \
         [elapsed: {elapsed_precise} | remaining: {eta_precise}]").expect("Invalid progress bar template"));
    pbar.set_draw_target(ProgressDrawTarget::stderr_with_hz(20));
    pbar.set_message(msg);

    pbar
}
