//! Parse the `.safetensors` weight file and expose a strongly-typed [`Weights`]
//! struct backed by `ndarray` arrays.
//!
//! # Safetensors binary layout
//! ```text
//! [8 bytes] little-endian u64 — length of the JSON header (N)
//! [N bytes] UTF-8 JSON        — maps tensor name → {dtype, shape, data_offsets}
//! [remaining] raw tensor data — byte ranges addressed by data_offsets
//! ```
//! `data_offsets` are `[start, end)` byte positions **relative to the start of
//! the data region** (i.e. after the 8-byte length prefix and the JSON header).
//!
//! # Dtype handling
//! Qwen2 stores everything as `BF16`.  BF16 is the upper 16 bits of an IEEE
//! 754 f32, so conversion to f32 is a zero-copy bit-shift: place the two BF16
//! bytes in the high half of a u32 and reinterpret as f32.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{bail, Context, Result};
use ndarray::{Array1, Array2};
use serde::Deserialize;

use crate::config::ModelConfig;

// ---------------------------------------------------------------------------
// Public weight structs
// ---------------------------------------------------------------------------

/// Weights for a single transformer decoder layer.
pub struct LayerWeights {
    // --- attention ---
    /// RMS norm applied to the hidden state before attention. Shape: [hidden_size]
    pub input_layernorm: Array1<f32>,
    /// Query projection weight. Shape: [num_attention_heads * head_dim, hidden_size]
    pub q_proj_weight: Array2<f32>,
    /// Query projection bias.   Shape: [num_attention_heads * head_dim]
    pub q_proj_bias: Array1<f32>,
    /// Key projection weight.   Shape: [num_key_value_heads * head_dim, hidden_size]
    pub k_proj_weight: Array2<f32>,
    /// Key projection bias.     Shape: [num_key_value_heads * head_dim]
    pub k_proj_bias: Array1<f32>,
    /// Value projection weight. Shape: [num_key_value_heads * head_dim, hidden_size]
    pub v_proj_weight: Array2<f32>,
    /// Value projection bias.   Shape: [num_key_value_heads * head_dim]
    pub v_proj_bias: Array1<f32>,
    /// Output projection.       Shape: [hidden_size, num_attention_heads * head_dim]
    pub o_proj_weight: Array2<f32>,
    // --- MLP ---
    /// RMS norm applied to the hidden state before the MLP. Shape: [hidden_size]
    pub post_attention_layernorm: Array1<f32>,
    /// SwiGLU gate projection.  Shape: [intermediate_size, hidden_size]
    pub gate_proj_weight: Array2<f32>,
    /// SwiGLU up projection.    Shape: [intermediate_size, hidden_size]
    pub up_proj_weight: Array2<f32>,
    /// SwiGLU down projection.  Shape: [hidden_size, intermediate_size]
    pub down_proj_weight: Array2<f32>,
}

/// All model weights, loaded from one `.safetensors` file.
pub struct Weights {
    /// Token embedding table. Shape: [vocab_size, hidden_size]
    pub embed_tokens: Array2<f32>,
    /// Final RMS normalisation before the LM head. Shape: [hidden_size]
    pub norm: Array1<f32>,
    /// Per-layer weights, in layer order (layer 0 first).
    pub layers: Vec<LayerWeights>,
    // `lm_head.weight` is tied to `embed_tokens` in Qwen2.5 — use embed_tokens
    // directly when computing logits.
}

impl Weights {
    /// Load weights from the single `model.safetensors` file inside `dir`.
    pub fn from_dir(dir: &Path, cfg: &ModelConfig) -> Result<Self> {
        let path = dir.join("model.safetensors");
        eprintln!("Loading weights from {} …", path.display());
        let bytes = std::fs::read(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        load_weights(&bytes, cfg)
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Serde target for a single tensor entry in the JSON header.
#[derive(Deserialize)]
struct TensorMeta {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

/// Parse the entire safetensors byte slice and build a [`Weights`].
fn load_weights(bytes: &[u8], cfg: &ModelConfig) -> Result<Weights> {
    if bytes.len() < 8 {
        bail!("safetensors file too small");
    }

    // 1. Read the header length.
    let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;

    if bytes.len() < 8 + header_len {
        bail!("safetensors header truncated");
    }

    // 2. Parse the JSON header.
    let header_json = std::str::from_utf8(&bytes[8..8 + header_len])
        .context("safetensors header is not valid UTF-8")?;
    let header: HashMap<String, serde_json::Value> =
        serde_json::from_str(header_json).context("parsing safetensors header JSON")?;

    // The data region starts right after the JSON header.
    let data_start = 8 + header_len;
    let data = &bytes[data_start..];

    // Build a lookup: name → (f32 array as flat Vec, shape)
    let mut tensors: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    for (name, val) in &header {
        if name == "__metadata__" {
            continue;
        }
        let meta: TensorMeta = serde_json::from_value(val.clone())
            .with_context(|| format!("parsing tensor meta for {name}"))?;
        let [start, end] = meta.data_offsets;
        let raw = data
            .get(start..end)
            .with_context(|| format!("tensor {name}: data_offsets [{start},{end}) out of range"))?;
        let f32_data = decode_to_f32(raw, &meta.dtype)
            .with_context(|| format!("decoding tensor {name} ({})", meta.dtype))?;
        tensors.insert(name.clone(), (f32_data, meta.shape));
    }

    // Helper: extract a 1-D array.
    let get1 = |key: &str| -> Result<Array1<f32>> {
        let (data, shape) = tensors
            .get(key)
            .with_context(|| format!("tensor not found: {key}"))?;
        if shape.len() != 1 {
            bail!("{key}: expected rank-1 tensor, got shape {:?}", shape);
        }
        Ok(Array1::from_vec(data.clone()))
    };

    // Helper: extract a 2-D array in row-major order.
    let get2 = |key: &str| -> Result<Array2<f32>> {
        let (data, shape) = tensors
            .get(key)
            .with_context(|| format!("tensor not found: {key}"))?;
        if shape.len() != 2 {
            bail!("{key}: expected rank-2 tensor, got shape {:?}", shape);
        }
        Array2::from_shape_vec((shape[0], shape[1]), data.clone())
            .with_context(|| format!("reshaping {key}"))
    };

    // 3. Extract global tensors.
    let embed_tokens = get2("model.embed_tokens.weight")?;
    let norm = get1("model.norm.weight")?;

    // 4. Extract per-layer tensors.
    let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
    for i in 0..cfg.num_hidden_layers {
        let p = format!("model.layers.{i}");
        let layer = LayerWeights {
            input_layernorm:        get1(&format!("{p}.input_layernorm.weight"))?,
            q_proj_weight:          get2(&format!("{p}.self_attn.q_proj.weight"))?,
            q_proj_bias:            get1(&format!("{p}.self_attn.q_proj.bias"))?,
            k_proj_weight:          get2(&format!("{p}.self_attn.k_proj.weight"))?,
            k_proj_bias:            get1(&format!("{p}.self_attn.k_proj.bias"))?,
            v_proj_weight:          get2(&format!("{p}.self_attn.v_proj.weight"))?,
            v_proj_bias:            get1(&format!("{p}.self_attn.v_proj.bias"))?,
            o_proj_weight:          get2(&format!("{p}.self_attn.o_proj.weight"))?,
            post_attention_layernorm: get1(&format!("{p}.post_attention_layernorm.weight"))?,
            gate_proj_weight:       get2(&format!("{p}.mlp.gate_proj.weight"))?,
            up_proj_weight:         get2(&format!("{p}.mlp.up_proj.weight"))?,
            down_proj_weight:       get2(&format!("{p}.mlp.down_proj.weight"))?,
        };
        layers.push(layer);
    }

    eprintln!(
        "Loaded {} layers, embed_tokens {:?}, norm {:?}",
        layers.len(),
        embed_tokens.dim(),
        norm.dim()
    );

    Ok(Weights { embed_tokens, norm, layers })
}

/// Convert raw bytes from the safetensors data region into `Vec<f32>`.
///
/// Supported dtypes: `BF16`, `F16`, `F32`.
fn decode_to_f32(raw: &[u8], dtype: &str) -> Result<Vec<f32>> {
    match dtype {
        "BF16" => {
            if raw.len() % 2 != 0 {
                bail!("BF16 buffer length {} is not a multiple of 2", raw.len());
            }
            Ok(raw
                .chunks_exact(2)
                .map(|b| {
                    // BF16 = upper 16 bits of f32.  Shift into position and cast.
                    let bits = u32::from(u16::from_le_bytes([b[0], b[1]])) << 16;
                    f32::from_bits(bits)
                })
                .collect())
        }
        "F16" => {
            if raw.len() % 2 != 0 {
                bail!("F16 buffer length {} is not a multiple of 2", raw.len());
            }
            Ok(raw
                .chunks_exact(2)
                .map(|b| f16_to_f32(u16::from_le_bytes([b[0], b[1]])))
                .collect())
        }
        "F32" => {
            if raw.len() % 4 != 0 {
                bail!("F32 buffer length {} is not a multiple of 4", raw.len());
            }
            Ok(raw
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect())
        }
        other => bail!("unsupported dtype: {other}"),
    }
}

/// Convert a half-precision (IEEE 754 binary16) value to f32.
fn f16_to_f32(half: u16) -> f32 {
    let sign = u32::from(half >> 15) << 31;
    let exp = u32::from((half >> 10) & 0x1F);
    let mantissa = u32::from(half & 0x3FF);
    let bits = if exp == 0 {
        // Subnormal: re-normalise.
        if mantissa == 0 {
            sign // ±0
        } else {
            let mut m = mantissa;
            let mut e = 112u32; // 127 - 15 = 112
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((m & 0x3FF) << 13)
        }
    } else if exp == 31 {
        // Inf / NaN
        sign | 0x7F80_0000 | (mantissa << 13)
    } else {
        sign | ((exp + 112) << 23) | (mantissa << 13)
    };
    f32::from_bits(bits)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- BF16 / F16 / F32 round-trip ----------------------------------------

    #[test]
    fn bf16_zero() {
        let result = decode_to_f32(&[0x00, 0x00], "BF16").unwrap();
        assert_eq!(result[0], 0.0f32);
    }

    #[test]
    fn bf16_one() {
        // BF16 representation of 1.0 is 0x3F80.
        let result = decode_to_f32(&[0x80, 0x3F], "BF16").unwrap();
        assert!((result[0] - 1.0).abs() < 1e-6, "got {}", result[0]);
    }

    #[test]
    fn bf16_minus_two() {
        // -2.0 in BF16 is 0xC000.
        let result = decode_to_f32(&[0x00, 0xC0], "BF16").unwrap();
        assert!((result[0] - (-2.0)).abs() < 1e-6, "got {}", result[0]);
    }

    #[test]
    fn bf16_multiple_values() {
        // 1.0 (0x3F80) followed by -1.0 (0xBF80).
        let bytes = [0x80u8, 0x3F, 0x80, 0xBF];
        let result = decode_to_f32(&bytes, "BF16").unwrap();
        assert_eq!(result.len(), 2);
        assert!((result[0] - 1.0).abs() < 1e-5);
        assert!((result[1] - (-1.0)).abs() < 1e-5);
    }

    #[test]
    fn f32_roundtrip() {
        let val: f32 = 3.14159;
        let bytes = val.to_le_bytes();
        let result = decode_to_f32(&bytes, "F32").unwrap();
        assert!((result[0] - val).abs() < 1e-6);
    }

    #[test]
    fn f16_one() {
        // 1.0 in F16 is 0x3C00.
        let result = decode_to_f32(&[0x00, 0x3C], "F16").unwrap();
        assert!((result[0] - 1.0).abs() < 1e-3, "got {}", result[0]);
    }

    #[test]
    fn f16_zero() {
        let result = decode_to_f32(&[0x00, 0x00], "F16").unwrap();
        assert_eq!(result[0], 0.0f32);
    }

    #[test]
    fn unsupported_dtype_errors() {
        assert!(decode_to_f32(&[0, 0], "INT8").is_err());
    }

    // --- Header parsing ------------------------------------------------------

    fn make_safetensors(name: &str, values: &[f32]) -> Vec<u8> {
        // Build data bytes (F32 LE).
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let meta = serde_json::json!({
            name: {
                "dtype": "F32",
                "shape": [values.len()],
                "data_offsets": [0, data.len()]
            }
        });
        let header = serde_json::to_vec(&meta).unwrap();
        let mut out = (header.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(&header);
        out.extend_from_slice(&data);
        out
    }

    #[test]
    fn parse_minimal_safetensors() {
        let bytes = make_safetensors("model.norm.weight", &[1.0, 2.0, 3.0]);
        let meta = serde_json::json!({
            "model.norm.weight": {
                "dtype": "F32",
                "shape": [3],
                "data_offsets": [0, 12]
            }
        });
        let json_bytes = serde_json::to_vec(&meta).unwrap();
        // Re-parse via our decode path.
        let header: HashMap<String, serde_json::Value> =
            serde_json::from_slice(&json_bytes).unwrap();
        let entry = &header["model.norm.weight"];
        let tensor_meta: TensorMeta = serde_json::from_value(entry.clone()).unwrap();
        let n = bytes.len();
        // Verify offsets make sense.
        assert_eq!(tensor_meta.shape, vec![3]);
        assert_eq!(tensor_meta.data_offsets[1] - tensor_meta.data_offsets[0], 12);
        let _ = n; // suppress warning
    }

    // --- Integration test ----------------------------------------------------

    /// Loads the real model weights if the cache file is present.
    /// This is slow (~few seconds) and requires the file to exist.
    #[test]
    #[ignore]
    fn load_real_weights() {
        use crate::config::ModelConfig;
        use std::path::PathBuf;

        let home = std::env::var("HOME").expect("HOME not set");
        let dir =
            PathBuf::from(home).join(".cache/myllm/Qwen/Qwen2.5-0.5B-Instruct");
        if !dir.join("model.safetensors").exists() {
            return;
        }
        let cfg = ModelConfig::from_dir(&dir).expect("config");
        let weights = Weights::from_dir(&dir, &cfg).expect("weights");

        assert_eq!(weights.embed_tokens.dim(), (151936, 896));
        assert_eq!(weights.norm.dim(), 896);
        assert_eq!(weights.layers.len(), 24);

        let l0 = &weights.layers[0];
        assert_eq!(l0.input_layernorm.dim(), 896);
        assert_eq!(l0.q_proj_weight.dim(), (896, 896));
        assert_eq!(l0.q_proj_bias.dim(), 896);
        assert_eq!(l0.k_proj_weight.dim(), (128, 896));
        assert_eq!(l0.k_proj_bias.dim(), 128);
        assert_eq!(l0.v_proj_weight.dim(), (128, 896));
        assert_eq!(l0.v_proj_bias.dim(), 128);
        assert_eq!(l0.o_proj_weight.dim(), (896, 896));
        assert_eq!(l0.gate_proj_weight.dim(), (4864, 896));
        assert_eq!(l0.up_proj_weight.dim(), (4864, 896));
        assert_eq!(l0.down_proj_weight.dim(), (896, 4864));
    }
}
