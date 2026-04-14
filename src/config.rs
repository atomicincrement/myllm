//! Deserialise `config.json` into a [`ModelConfig`] struct.

use std::path::Path;

use anyhow::{Context, Result};
use serde::Deserialize;

/// All fields from the Qwen2 `config.json` that the model code needs.
#[derive(Debug, Deserialize)]
pub struct ModelConfig {
    /// Total number of distinct token ids.
    pub vocab_size: usize,
    /// Size of the hidden state / embedding dimension (d_model).
    pub hidden_size: usize,
    /// Inner size of the SwiGLU MLP (gate/up projection output size).
    pub intermediate_size: usize,
    /// Number of transformer decoder layers.
    pub num_hidden_layers: usize,
    /// Number of query attention heads.
    pub num_attention_heads: usize,
    /// Number of key/value attention heads (< num_attention_heads for GQA).
    pub num_key_value_heads: usize,
    /// Maximum sequence length the model supports.
    pub max_position_embeddings: usize,
    /// Epsilon used inside RMS layer normalisation.
    pub rms_norm_eps: f64,
    /// Base frequency for rotary positional embeddings.
    pub rope_theta: f64,
    /// When true the embedding matrix is re-used as the LM head weight.
    #[serde(default)]
    pub tie_word_embeddings: bool,
    /// Token id used as beginning-of-sequence marker.
    pub bos_token_id: u32,
    /// Token id that signals end of generation.
    pub eos_token_id: u32,
    /// On-disk dtype ("bfloat16", "float16", "float32", …).
    pub torch_dtype: String,
}

impl ModelConfig {
    /// Load and parse `config.json` from `dir`.
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let path = dir.join("config.json");
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        let cfg: ModelConfig =
            serde_json::from_str(&text).context("parsing config.json")?;
        Ok(cfg)
    }

    /// Per-head dimension for query / key / value vectors.
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}
