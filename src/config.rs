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

#[cfg(test)]
mod tests {
    use super::*;

    // Minimal JSON that matches exactly what Qwen2.5-0.5B-Instruct ships.
    const QWEN_CONFIG_JSON: &str = r#"{
        "architectures": ["Qwen2ForCausalLM"],
        "attention_dropout": 0.0,
        "bos_token_id": 151643,
        "eos_token_id": 151645,
        "hidden_act": "silu",
        "hidden_size": 896,
        "initializer_range": 0.02,
        "intermediate_size": 4864,
        "max_position_embeddings": 32768,
        "max_window_layers": 21,
        "model_type": "qwen2",
        "num_attention_heads": 14,
        "num_hidden_layers": 24,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-06,
        "rope_theta": 1000000.0,
        "sliding_window": 32768,
        "tie_word_embeddings": true,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.43.1",
        "use_cache": true,
        "use_sliding_window": false,
        "vocab_size": 151936
    }"#;

    fn qwen_config() -> ModelConfig {
        serde_json::from_str(QWEN_CONFIG_JSON).expect("embedded JSON is valid")
    }

    #[test]
    fn parses_vocab_size() {
        assert_eq!(qwen_config().vocab_size, 151936);
    }

    #[test]
    fn parses_hidden_size() {
        assert_eq!(qwen_config().hidden_size, 896);
    }

    #[test]
    fn parses_intermediate_size() {
        assert_eq!(qwen_config().intermediate_size, 4864);
    }

    #[test]
    fn parses_num_layers() {
        assert_eq!(qwen_config().num_hidden_layers, 24);
    }

    #[test]
    fn parses_attention_heads() {
        let cfg = qwen_config();
        assert_eq!(cfg.num_attention_heads, 14);
        assert_eq!(cfg.num_key_value_heads, 2);
    }

    #[test]
    fn parses_rope_params() {
        let cfg = qwen_config();
        assert_eq!(cfg.max_position_embeddings, 32768);
        assert!((cfg.rope_theta - 1_000_000.0).abs() < 1.0);
        assert!((cfg.rms_norm_eps - 1e-6).abs() < 1e-12);
    }

    #[test]
    fn parses_token_ids() {
        let cfg = qwen_config();
        assert_eq!(cfg.bos_token_id, 151643);
        assert_eq!(cfg.eos_token_id, 151645);
    }

    #[test]
    fn parses_tie_word_embeddings() {
        assert!(qwen_config().tie_word_embeddings);
    }

    #[test]
    fn tie_word_embeddings_defaults_to_false() {
        // When the field is absent, serde should default to false.
        let json = r#"{
            "vocab_size": 1, "hidden_size": 64, "intermediate_size": 128,
            "num_hidden_layers": 1, "num_attention_heads": 2,
            "num_key_value_heads": 2, "max_position_embeddings": 512,
            "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
            "bos_token_id": 0, "eos_token_id": 1, "torch_dtype": "float32"
        }"#;
        let cfg: ModelConfig = serde_json::from_str(json).unwrap();
        assert!(!cfg.tie_word_embeddings);
    }

    #[test]
    fn head_dim_is_hidden_over_heads() {
        // 896 / 14 = 64
        assert_eq!(qwen_config().head_dim(), 64);
    }

    #[test]
    fn unknown_fields_are_ignored() {
        // Extra keys in config.json (model_type, sliding_window, etc.) must not
        // cause a parse error.
        qwen_config(); // would panic if serde rejected unknown fields
    }

    /// Integration test: parse the real file from the local cache.
    /// Skipped automatically if the file hasn't been downloaded yet.
    #[test]
    fn from_dir_reads_real_cache() {
        let home = match std::env::var("HOME") {
            Ok(h) => h,
            Err(_) => return,
        };
        let dir = std::path::PathBuf::from(home)
            .join(".cache/myllm/Qwen/Qwen2.5-0.5B-Instruct");
        if !dir.join("config.json").exists() {
            return; // not downloaded — skip
        }
        let cfg = ModelConfig::from_dir(&dir).expect("from_dir should succeed");
        assert_eq!(cfg.vocab_size, 151936);
        assert_eq!(cfg.hidden_size, 896);
        assert_eq!(cfg.head_dim(), 64);
    }
}
