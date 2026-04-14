//! Byte-Pair Encoding tokenizer built from scratch using `tokenizer.json`.
//!
//! # Pipeline (encode)
//! 1. NFC-normalise the input text.
//! 2. Scan for special tokens (e.g. `<|im_start|>`) and split around them.
//! 3. For each non-special chunk: split with the GPT-2 pre-tokenizer regex.
//! 4. Map every byte of each pre-token to its GPT-2 "byte-level" Unicode char.
//! 5. Apply BPE merges (greedy, highest-priority pair first).
//! 6. Look up each merged symbol in the vocabulary to get token IDs.
//!
//! # Pipeline (decode)
//! Reverse-vocab lookup → concatenate unicode strings → map each char back to
//! the original byte → interpret the byte stream as UTF-8.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use regex::Regex;
use serde::Deserialize;
use unicode_normalization::UnicodeNormalization;

// ---------------------------------------------------------------------------
// GPT-2 byte ↔ unicode mapping
// ---------------------------------------------------------------------------

/// Maps each byte value (0-255) to its GPT-2 "byte-level" Unicode codepoint.
///
/// The mapping keeps printable ASCII + Latin-1 supplement in-place and sends
/// control / high bytes to the Private-Use-adjacent block starting at U+0100.
/// Space (0x20) maps to Ġ (U+0120 = decimal 288).
const BYTE_TO_UNICODE: [u32; 256] = [
    256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, // 0-15
    272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, // 16-31
    288, // 32  space → Ġ
     33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47, // 33-47
     48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63, // 48-63
     64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79, // 64-79
     80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95, // 80-95
     96,  97,  98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, // 96-111
    112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, // 112-126
    289, // 127
    290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, // 128-143
    306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, // 144-159
    322, // 160
    161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, // 161-172
    323, // 173
    174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, // 174-189
    190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, // 190-205
    206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, // 206-221
    222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, // 222-237
    238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, // 238-253
    254, 255, // 254-255
];

/// Build the reverse map: GPT-2 unicode codepoint → byte.
fn build_unicode_to_byte() -> HashMap<char, u8> {
    let mut m = HashMap::with_capacity(256);
    for (b, &cp) in BYTE_TO_UNICODE.iter().enumerate() {
        m.insert(char::from_u32(cp).unwrap(), b as u8);
    }
    m
}

// ---------------------------------------------------------------------------
// JSON schema for tokenizer.json
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct TokenizerJson {
    model: BpeModel,
    #[serde(default)]
    added_tokens: Vec<AddedToken>,
}

#[derive(Deserialize)]
struct BpeModel {
    vocab: HashMap<String, u32>,
    merges: Vec<String>,
}

#[derive(Deserialize)]
struct AddedToken {
    id: u32,
    content: String,
}

#[derive(Deserialize)]
struct TokenizerConfigJson {
    bos_token_id: Option<u32>,
    eos_token_id: Option<u32>,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// A Byte-Pair Encoding tokenizer loaded from `tokenizer.json`.
pub struct Tokenizer {
    /// Token string → token id.
    vocab: HashMap<String, u32>,
    /// Token id → token string.
    reverse_vocab: Vec<String>,
    /// BPE merge rules: (left, right) → merge priority (0 = highest priority).
    merges: HashMap<(String, String), usize>,
    /// Special token strings sorted longest-first (for greedy matching).
    special_tokens: Vec<(String, u32)>,
    /// Reverse map: GPT-2 unicode char → byte.
    unicode_to_byte: HashMap<char, u8>,
    /// Pre-tokenizer regex (GPT-2 / Qwen pattern).
    pretok_re: Regex,
    /// Beginning-of-sequence token id.
    pub bos_token_id: u32,
    /// End-of-sequence token id.
    pub eos_token_id: u32,
}

impl Tokenizer {
    /// Load the tokenizer from `tokenizer.json` (and `tokenizer_config.json`
    /// for bos/eos ids) inside `dir`.
    pub fn from_dir(dir: &Path) -> Result<Self> {
        // --- tokenizer.json ---
        let tok_path = dir.join("tokenizer.json");
        let tok_text = std::fs::read_to_string(&tok_path)
            .with_context(|| format!("reading {}", tok_path.display()))?;
        let tok_json: TokenizerJson =
            serde_json::from_str(&tok_text).context("parsing tokenizer.json")?;

        // --- tokenizer_config.json ---
        let cfg_path = dir.join("tokenizer_config.json");
        let cfg_text = std::fs::read_to_string(&cfg_path)
            .with_context(|| format!("reading {}", cfg_path.display()))?;
        let cfg_json: TokenizerConfigJson =
            serde_json::from_str(&cfg_text).context("parsing tokenizer_config.json")?;

        let vocab = tok_json.model.vocab;
        let vocab_size = vocab.values().copied().max().unwrap_or(0) as usize + 1;

        // Build reverse vocab (id → string).
        let mut reverse_vocab = vec![String::new(); vocab_size];
        for (token, &id) in &vocab {
            if (id as usize) < vocab_size {
                reverse_vocab[id as usize] = token.clone();
            }
        }

        // Build merge priority map:  (left, right) → index.
        // The merge list is "left right" per entry separated by a single space.
        let mut merges = HashMap::with_capacity(tok_json.model.merges.len());
        for (priority, merge_str) in tok_json.model.merges.iter().enumerate() {
            if let Some((left, right)) = merge_str.split_once(' ') {
                merges.insert((left.to_owned(), right.to_owned()), priority);
            }
        }

        // Special tokens: sort longest-first for greedy scanning.
        let mut special_tokens: Vec<(String, u32)> = tok_json
            .added_tokens
            .iter()
            .map(|t| (t.content.clone(), t.id))
            .collect();
        // Also add any special tokens that ended up in vocab but not added_tokens.
        special_tokens.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

        let bos_token_id = cfg_json.bos_token_id.unwrap_or(0);
        let eos_token_id = cfg_json.eos_token_id.unwrap_or(0);

        // GPT-2 / Qwen pre-tokenizer regex (Unicode-aware).
        let pretok_re = Regex::new(
            r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
        ).context("compiling pre-tokenizer regex")?;

        Ok(Self {
            vocab,
            reverse_vocab,
            merges,
            special_tokens,
            unicode_to_byte: build_unicode_to_byte(),
            pretok_re,
            bos_token_id,
            eos_token_id,
        })
    }

    /// Encode `text` into a sequence of token IDs.
    ///
    /// Special tokens (e.g. `<|im_start|>`) are matched first and returned as
    /// single IDs.  The rest of the text goes through NFC normalisation,
    /// pre-tokenisation, byte-level encoding, and BPE.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        // NFC-normalise.
        let text: String = text.nfc().collect();

        let mut ids = Vec::new();
        // Split around special tokens (greedy, longest match wins).
        for chunk in self.split_special(&text) {
            match chunk {
                Chunk::Special(id) => ids.push(id),
                Chunk::Normal(s) => {
                    for m in self.pretok_re.find_iter(&s) {
                        let pre_tok = m.as_str();
                        let bpe_tokens = self.bpe(pre_tok)?;
                        for tok in bpe_tokens {
                            let id = self.vocab.get(&tok).copied().with_context(|| {
                                format!("token not in vocab: {:?}", tok)
                            })?;
                            ids.push(id);
                        }
                    }
                }
            }
        }
        Ok(ids)
    }

    /// Decode a sequence of token IDs back to a UTF-8 string.
    ///
    /// Unknown IDs are skipped silently.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            let token = match self.reverse_vocab.get(id as usize) {
                Some(t) if !t.is_empty() => t,
                _ => continue,
            };
            // Special tokens: emit their content as UTF-8 directly.
            if self.special_tokens.iter().any(|(s, sid)| *sid == id && s == token) {
                bytes.extend_from_slice(token.as_bytes());
                continue;
            }
            // Normal tokens: map each unicode char back through the byte table.
            for ch in token.chars() {
                if let Some(&b) = self.unicode_to_byte.get(&ch) {
                    bytes.push(b);
                }
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    // ---- private helpers ---------------------------------------------------

    /// Apply BPE merges to a single pre-token string.
    ///
    /// The string is first byte-level encoded (each byte → unicode char), giving
    /// an initial sequence of single-char symbols.  Merges are applied greedily:
    /// at each step find the adjacent pair with the lowest priority index (most
    /// important merge) and join them.
    fn bpe(&self, pre_token: &str) -> Result<Vec<String>> {
        // Byte-level encode: each byte → one unicode char.
        let mut symbols: Vec<String> = pre_token
            .bytes()
            .map(|b| {
                let cp = BYTE_TO_UNICODE[b as usize];
                char::from_u32(cp).unwrap().to_string()
            })
            .collect();

        if symbols.len() <= 1 {
            return Ok(symbols);
        }

        loop {
            // Find the pair with the highest priority (lowest index in merges).
            let best = symbols
                .windows(2)
                .enumerate()
                .filter_map(|(i, pair)| {
                    let key = (pair[0].clone(), pair[1].clone());
                    self.merges.get(&key).map(|&prio| (i, prio))
                })
                .min_by_key(|&(_, prio)| prio);

            let (idx, _) = match best {
                Some(b) => b,
                None => break,
            };

            // Merge symbols[idx] and symbols[idx+1].
            let merged = format!("{}{}", symbols[idx], symbols[idx + 1]);
            symbols[idx] = merged;
            symbols.remove(idx + 1);
        }

        Ok(symbols)
    }

    /// Split `text` into alternating [`Chunk::Special`] and [`Chunk::Normal`]
    /// segments by scanning for special tokens longest-first.
    fn split_special<'a>(&self, text: &'a str) -> Vec<Chunk<'a>> {
        let mut chunks = Vec::new();
        let mut remaining = text;

        'outer: while !remaining.is_empty() {
            // Try each special token (sorted longest-first).
            for (special, id) in &self.special_tokens {
                if remaining.starts_with(special.as_str()) {
                    chunks.push(Chunk::Special(*id));
                    remaining = &remaining[special.len()..];
                    continue 'outer;
                }
            }
            // No special token at the start — find where the next one begins.
            let next_special = self
                .special_tokens
                .iter()
                .filter_map(|(s, _)| remaining.find(s.as_str()))
                .min();
            match next_special {
                Some(pos) => {
                    chunks.push(Chunk::Normal(&remaining[..pos]));
                    remaining = &remaining[pos..];
                }
                None => {
                    chunks.push(Chunk::Normal(remaining));
                    break;
                }
            }
        }
        chunks
    }
}

enum Chunk<'a> {
    Normal(&'a str),
    Special(u32),
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn real_tokenizer() -> Option<Tokenizer> {
        let home = std::env::var("HOME").ok()?;
        let dir = Path::new(&home).join(".cache/myllm/Qwen/Qwen2.5-0.5B-Instruct");
        if !dir.join("tokenizer.json").exists() {
            return None;
        }
        Tokenizer::from_dir(&dir).ok()
    }

    #[test]
    fn byte_to_unicode_length() {
        assert_eq!(BYTE_TO_UNICODE.len(), 256);
    }

    #[test]
    fn space_maps_to_gstroke() {
        // Space (0x20 = 32) → U+0120 = 288 decimal = Ġ
        assert_eq!(BYTE_TO_UNICODE[32], 288);
        assert_eq!(char::from_u32(288).unwrap(), 'Ġ');
    }

    #[test]
    fn printable_ascii_maps_to_itself() {
        for b in 33u8..=126 {
            assert_eq!(BYTE_TO_UNICODE[b as usize], b as u32, "byte {b}");
        }
    }

    #[test]
    fn byte_unicode_roundtrip() {
        let u2b = build_unicode_to_byte();
        for (b, &cp) in BYTE_TO_UNICODE.iter().enumerate() {
            let ch = char::from_u32(cp).expect("valid codepoint");
            assert_eq!(*u2b.get(&ch).expect("char in reverse map"), b as u8);
        }
    }

    #[test]
    fn encode_decode_simple() {
        let tok = match real_tokenizer() {
            Some(t) => t,
            None => return,
        };
        let text = "Hello, world!";
        let ids = tok.encode(text).expect("encode");
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, text, "roundtrip failed: ids={ids:?}");
    }

    #[test]
    fn special_token_ids() {
        let tok = match real_tokenizer() {
            Some(t) => t,
            None => return,
        };
        // <|endoftext|> should be id 151643 for Qwen2.5
        assert_eq!(tok.eos_token_id, 151645);
        assert_eq!(tok.bos_token_id, 151643);
    }

    #[test]
    fn encode_special_tokens() {
        let tok = match real_tokenizer() {
            Some(t) => t,
            None => return,
        };
        let ids = tok.encode("<|im_start|>user\n").expect("encode");
        // First token must be the <|im_start|> special token (id 151644).
        assert_eq!(ids[0], 151644, "im_start id wrong, got ids={ids:?}");
    }

    #[test]
    fn encode_decode_unicode() {
        let tok = match real_tokenizer() {
            Some(t) => t,
            None => return,
        };
        let text = "こんにちは世界";
        let ids = tok.encode(text).expect("encode");
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, text);
    }

    #[test]
    fn encode_decode_numbers() {
        let tok = match real_tokenizer() {
            Some(t) => t,
            None => return,
        };
        let text = "3.14159265358979";
        let ids = tok.encode(text).expect("encode");
        let out = tok.decode(&ids);
        assert_eq!(out, text);
    }

    #[test]
    fn empty_string() {
        let tok = match real_tokenizer() {
            Some(t) => t,
            None => return,
        };
        let ids = tok.encode("").expect("encode empty");
        assert!(ids.is_empty());
        assert_eq!(tok.decode(&[]), "");
    }

    /// Build the full Qwen chat prompt and verify the leading special tokens.
    #[test]
    fn chat_template_prefix() {
        let tok = match real_tokenizer() {
            Some(t) => t,
            None => return,
        };
        let prompt =
            "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n";
        let ids = tok.encode(prompt).expect("encode");
        assert_eq!(ids[0], 151644, "<|im_start|>");
        // The last token in the prompt should be another <|im_start|> (151644).
        assert_eq!(*ids.last().unwrap(), 151644, "last im_start");
    }
}
