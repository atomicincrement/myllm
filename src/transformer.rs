//! Transformer building blocks for Qwen2.
//!
//! * [`rms_norm`]       — Root-Mean-Square layer normalisation
//! * [`RopeCache`]      — precomputed cosine / sine tables for RoPE
//! * [`attention`]      — grouped-query attention with KV cache
//! * [`mlp`]            — SwiGLU feed-forward network
//! * [`InferenceState`] — per-session KV cache
//! * [`forward`]        — full Qwen2 forward pass

use ndarray::{concatenate, Array1, Array2, Array3, ArrayView1, Axis};

use crate::config::ModelConfig;
use crate::safetensors::{LayerWeights, Weights};

// ---------------------------------------------------------------------------
// RMS layer normalisation
// ---------------------------------------------------------------------------

/// Apply RMS layer normalisation to a single hidden-state vector.
///
/// ```text
/// rms  = sqrt( mean(x²) + eps )
/// out  = (x / rms) * weight
/// ```
///
/// Unlike `LayerNorm`, RMS norm has no bias and does **not** subtract the mean.
///
/// # Arguments
/// * `x`      — input vector of length `hidden_size`
/// * `weight` — learned scale vector of length `hidden_size` (from the model weights)
/// * `eps`    — small constant for numerical stability (typically 1e-6)
pub fn rms_norm(x: ArrayView1<f32>, weight: ArrayView1<f32>, eps: f32) -> Array1<f32> {
    debug_assert_eq!(x.len(), weight.len());
    let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let scale = (mean_sq + eps).sqrt().recip();
    &x * scale * &weight
}

/// Apply RMS norm to every row of a 2-D array `[seq_len, hidden_size]`.
pub fn rms_norm_2d(x: &Array2<f32>, weight: ArrayView1<f32>, eps: f32) -> Array2<f32> {
    let (seq, _) = x.dim();
    let mut out = x.clone();
    for i in 0..seq {
        let row = rms_norm(x.row(i), weight, eps);
        out.row_mut(i).assign(&row);
    }
    out
}

// ---------------------------------------------------------------------------
// Rotary Positional Embeddings (RoPE)
// ---------------------------------------------------------------------------

/// Precomputed cosine and sine tables for RoPE.
///
/// # Layout
/// Both tables have shape `[max_seq_len, head_dim / 2]`.
/// For position `p` and pair index `i`, the rotation angle is:
/// ```text
/// θ_i = p / rope_theta^(2i / head_dim)
/// ```
pub struct RopeCache {
    /// `cos[p, i]` — cosine of the angle for position p, pair i
    pub cos: Array2<f32>,
    /// `sin[p, i]` — sine of the angle for position p, pair i
    pub sin: Array2<f32>,
}

impl RopeCache {
    /// Build the RoPE tables from the model config.
    pub fn new(cfg: &ModelConfig) -> Self {
        let head_dim = cfg.head_dim();
        let half = head_dim / 2;
        let max_len = cfg.max_position_embeddings;
        let theta = cfg.rope_theta as f32;

        // Inverse frequencies: 1 / theta^(2i / head_dim)  for i in 0..half
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32))
            .collect();

        let mut cos_table = Array2::<f32>::zeros((max_len, half));
        let mut sin_table = Array2::<f32>::zeros((max_len, half));

        for pos in 0..max_len {
            for (i, &freq) in inv_freq.iter().enumerate() {
                let angle = pos as f32 * freq;
                cos_table[[pos, i]] = angle.cos();
                sin_table[[pos, i]] = angle.sin();
            }
        }

        RopeCache {
            cos: cos_table,
            sin: sin_table,
        }
    }

    /// Apply RoPE to a query or key tensor.
    ///
    /// # Arguments
    /// * `x`         — shape `[seq_len, num_heads, head_dim]`
    /// * `start_pos` — position offset (> 0 during decode when KV cache is used)
    ///
    /// Each head vector is split into pairs `(x₀, x₁)` and rotated:
    /// ```text
    /// x₀' = x₀·cos - x₁·sin
    /// x₁' = x₀·sin + x₁·cos
    /// ```
    pub fn apply(&self, x: &Array3<f32>, start_pos: usize) -> Array3<f32> {
        let (seq, heads, head_dim) = x.dim();
        let half = head_dim / 2;
        let mut out = x.clone();

        for s in 0..seq {
            let pos = start_pos + s;
            let cos_row = self.cos.row(pos);
            let sin_row = self.sin.row(pos);

            for h in 0..heads {
                for i in 0..half {
                    let x0 = x[[s, h, i]];
                    let x1 = x[[s, h, i + half]];
                    let c = cos_row[i];
                    let sn = sin_row[i];
                    out[[s, h, i]]        = x0 * c - x1 * sn;
                    out[[s, h, i + half]] = x0 * sn + x1 * c;
                }
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// KV cache
// ---------------------------------------------------------------------------

/// Per-layer key/value cache for autoregressive generation.
pub struct KvCache {
    /// Accumulated keys:   `[past_len, kv_heads, head_dim]`
    pub k: Array3<f32>,
    /// Accumulated values: `[past_len, kv_heads, head_dim]`
    pub v: Array3<f32>,
}

impl KvCache {
    pub fn new(kv_heads: usize, head_dim: usize) -> Self {
        KvCache {
            k: Array3::zeros((0, kv_heads, head_dim)),
            v: Array3::zeros((0, kv_heads, head_dim)),
        }
    }
}

/// All per-layer KV caches for one inference session.
pub struct InferenceState {
    pub kv_caches: Vec<KvCache>,
}

impl InferenceState {
    pub fn new(cfg: &ModelConfig) -> Self {
        InferenceState {
            kv_caches: (0..cfg.num_hidden_layers)
                .map(|_| KvCache::new(cfg.num_key_value_heads, cfg.head_dim()))
                .collect(),
        }
    }
}

// ---------------------------------------------------------------------------
// Linear projection helper
// ---------------------------------------------------------------------------

/// `y = x @ weight.T  [+ bias]`
///
/// * `x`      — `[seq, in_features]`
/// * `weight` — `[out_features, in_features]`
/// * `bias`   — optional `[out_features]`
fn linear(x: &Array2<f32>, weight: &Array2<f32>, bias: Option<&Array1<f32>>) -> Array2<f32> {
    let mut out = x.dot(&weight.t()); // [seq, out_features]
    if let Some(b) = bias {
        for mut row in out.rows_mut() {
            row += b;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Activation / softmax helpers
// ---------------------------------------------------------------------------

/// Sigmoid-weighted linear unit: `x * σ(x)`.
#[inline]
fn silu(x: f32) -> f32 {
    x * (1.0 / (1.0 + (-x).exp()))
}

/// In-place row softmax (numerically stable).
fn softmax_1d(mut row: ndarray::ArrayViewMut1<f32>) {
    let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    row.mapv_inplace(|v| (v - max).exp());
    let sum = row.sum();
    row.mapv_inplace(|v| v / sum);
}

// ---------------------------------------------------------------------------
// Grouped-query attention (GQA)
// ---------------------------------------------------------------------------

/// Multi-head grouped-query attention for one transformer layer.
///
/// New K/V entries are appended to `kv_cache` in place so subsequent decode
/// calls can reuse them.
///
/// # Arguments
/// * `x`         — pre-norm hidden states `[seq, hidden_size]`
/// * `layer`     — weight tensors for this layer
/// * `rope`      — precomputed RoPE cosine / sine tables
/// * `kv_cache`  — KV cache; extended in-place
/// * `start_pos` — number of tokens already in the cache (= past length)
/// * `cfg`       — model configuration
pub fn attention(
    x: &Array2<f32>,
    layer: &LayerWeights,
    rope: &RopeCache,
    kv_cache: &mut KvCache,
    start_pos: usize,
    cfg: &ModelConfig,
) -> Array2<f32> {
    let (seq, _hidden) = x.dim();
    let q_heads = cfg.num_attention_heads;
    let kv_heads = cfg.num_key_value_heads;
    let head_dim = cfg.head_dim();
    let groups = q_heads / kv_heads; // 7 for Qwen2.5-0.5B

    // Linear projections ---------------------------------------------------
    let q_flat = linear(x, &layer.q_proj_weight, Some(&layer.q_proj_bias));
    let k_flat = linear(x, &layer.k_proj_weight, Some(&layer.k_proj_bias));
    let v_flat = linear(x, &layer.v_proj_weight, Some(&layer.v_proj_bias));

    // Reshape to [seq, heads, head_dim] ------------------------------------
    let q3 = q_flat
        .into_shape_with_order((seq, q_heads, head_dim))
        .expect("Q reshape");
    let k3 = k_flat
        .into_shape_with_order((seq, kv_heads, head_dim))
        .expect("K reshape");
    let v3 = v_flat
        .into_shape_with_order((seq, kv_heads, head_dim))
        .expect("V reshape");

    // Apply RoPE to Q and K ------------------------------------------------
    let q3 = rope.apply(&q3, start_pos);
    let k3 = rope.apply(&k3, start_pos);

    // Extend KV cache -------------------------------------------------------
    kv_cache.k = concatenate(Axis(0), &[kv_cache.k.view(), k3.view()])
        .expect("K concat");
    kv_cache.v = concatenate(Axis(0), &[kv_cache.v.view(), v3.view()])
        .expect("V concat");

    let total = kv_cache.k.dim().0; // start_pos + seq

    // Scaled dot-product attention with causal mask ------------------------
    let scale = (head_dim as f32).sqrt().recip();
    let mut attn_out = Array3::<f32>::zeros((seq, q_heads, head_dim));

    for h in 0..q_heads {
        let kv_h = h / groups; // which KV head this Q maps to

        // scores[s, t] = Q[s,h,:] · K[t,kv_h,:] / sqrt(head_dim)
        let mut scores = Array2::<f32>::zeros((seq, total));
        for s in 0..seq {
            for t in 0..total {
                let dot: f32 = (0..head_dim)
                    .map(|d| q3[[s, h, d]] * kv_cache.k[[t, kv_h, d]])
                    .sum();
                scores[[s, t]] = dot * scale;
            }
        }

        // Apply causal mask: future positions → −∞
        for s in 0..seq {
            let visible_up_to = start_pos + s; // inclusive
            for t in (visible_up_to + 1)..total {
                scores[[s, t]] = f32::NEG_INFINITY;
            }
            softmax_1d(scores.row_mut(s));
        }

        // Weighted sum over values
        for s in 0..seq {
            for d in 0..head_dim {
                let val: f32 = (0..total)
                    .map(|t| scores[[s, t]] * kv_cache.v[[t, kv_h, d]])
                    .sum();
                attn_out[[s, h, d]] = val;
            }
        }
    }

    // Flatten heads back to [seq, q_heads * head_dim] and project ----------
    let out_flat = attn_out
        .into_shape_with_order((seq, q_heads * head_dim))
        .expect("attn flatten");
    linear(&out_flat, &layer.o_proj_weight, None)
}

// ---------------------------------------------------------------------------
// SwiGLU MLP
// ---------------------------------------------------------------------------

/// SwiGLU feed-forward network:
/// ```text
/// gate = silu(x @ gate_proj.T)
/// up   = x @ up_proj.T
/// out  = (gate * up) @ down_proj.T
/// ```
pub fn mlp(x: &Array2<f32>, layer: &LayerWeights) -> Array2<f32> {
    let gate = linear(x, &layer.gate_proj_weight, None).mapv(silu);
    let up = linear(x, &layer.up_proj_weight, None);
    let hidden = gate * up;
    linear(&hidden, &layer.down_proj_weight, None)
}

// ---------------------------------------------------------------------------
// Full forward pass
// ---------------------------------------------------------------------------

/// Run the full Qwen2 transformer on a batch of new tokens.
///
/// # Arguments
/// * `tokens`    — token IDs to process (full prompt for prefill; a single
///                 new token for each subsequent decode step)
/// * `weights`   — model weights (read-only)
/// * `rope`      — precomputed RoPE tables
/// * `state`     — mutable KV-cache state; updated in-place
/// * `cfg`       — model configuration
///
/// # Returns
/// Logits for the **last** token position, shape `[vocab_size]`.
pub fn forward(
    tokens: &[u32],
    weights: &Weights,
    rope: &RopeCache,
    state: &mut InferenceState,
    cfg: &ModelConfig,
) -> Array1<f32> {
    let seq = tokens.len();
    let start_pos = state.kv_caches[0].k.dim().0; // tokens already cached

    // Embedding lookup -------------------------------------------------
    let mut x = Array2::<f32>::zeros((seq, cfg.hidden_size));
    for (i, &tok) in tokens.iter().enumerate() {
        x.row_mut(i)
            .assign(&weights.embed_tokens.row(tok as usize));
    }

    // Transformer layers -----------------------------------------------
    for (li, layer) in weights.layers.iter().enumerate() {
        let eps = cfg.rms_norm_eps as f32;

        // Pre-attention norm
        let normed = rms_norm_2d(&x, layer.input_layernorm.view(), eps);

        // GQA attention
        let attn_out = attention(
            &normed,
            layer,
            rope,
            &mut state.kv_caches[li],
            start_pos,
            cfg,
        );
        x = x + attn_out;

        // Pre-MLP norm
        let normed2 = rms_norm_2d(&x, layer.post_attention_layernorm.view(), eps);

        // SwiGLU MLP
        let mlp_out = mlp(&normed2, layer);
        x = x + mlp_out;
    }

    // Final norm -------------------------------------------------------
    let x_normed = rms_norm_2d(&x, weights.norm.view(), cfg.rms_norm_eps as f32);

    // LM head (weight tied to embed_tokens): hidden @ embed_tokens.T
    // embed_tokens: [vocab, hidden]; last_row: [hidden]
    // → [vocab, hidden] @ [hidden] = [vocab]
    let last_row = x_normed.row(seq - 1).to_owned();
    weights.embed_tokens.dot(&last_row)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    // ---- RMS norm ----------------------------------------------------------

    #[test]
    fn rms_norm_identity_weight() {
        // With weight = all-ones, the output should be x scaled by 1/rms(x).
        let x = array![3.0_f32, 4.0];
        let w = array![1.0_f32, 1.0];
        let out = rms_norm(x.view(), w.view(), 0.0);
        // rms = sqrt((9+16)/2) = sqrt(12.5)
        let rms = (12.5_f32).sqrt();
        assert!((out[0] - 3.0 / rms).abs() < 1e-6, "{}", out[0]);
        assert!((out[1] - 4.0 / rms).abs() < 1e-6, "{}", out[1]);
    }

    #[test]
    fn rms_norm_unit_rms_unchanged() {
        // x = [1, 1]: rms = sqrt((1+1)/2) = 1, so output = x * weight = x.
        let x = array![1.0_f32, 1.0];
        let w = array![1.0_f32, 1.0];
        let out = rms_norm(x.view(), w.view(), 1e-6);
        assert!((out[0] - 1.0).abs() < 1e-5, "got {}", out[0]);
        assert!((out[1] - 1.0).abs() < 1e-5, "got {}", out[1]);
    }

    #[test]
    fn rms_norm_scales_with_weight() {
        let x = array![1.0_f32, 1.0];
        let w = array![2.0_f32, 3.0];
        let out = rms_norm(x.view(), w.view(), 0.0);
        // rms([1,1]) = 1  → out = x * w
        assert!((out[0] - 2.0).abs() < 1e-6);
        assert!((out[1] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn rms_norm_zero_vector_with_eps() {
        let x = array![0.0_f32, 0.0];
        let w = array![1.0_f32, 1.0];
        let out = rms_norm(x.view(), w.view(), 1e-6);
        // Should not panic; result should be near zero.
        assert!(out[0].abs() < 1e-3);
        assert!(out[1].abs() < 1e-3);
    }

    #[test]
    fn rms_norm_2d_applies_per_row() {
        use ndarray::array;
        let x = array![[3.0_f32, 4.0], [1.0, 0.0]];
        let w = array![1.0_f32, 1.0];
        let out = rms_norm_2d(&x, w.view(), 0.0);
        // Row 0: rms = sqrt(12.5)
        let rms0 = (12.5_f32).sqrt();
        assert!((out[[0, 0]] - 3.0 / rms0).abs() < 1e-6);
        // Row 1: only element 0 is non-zero
        let rms1 = (0.5_f32).sqrt();
        assert!((out[[1, 0]] - 1.0 / rms1).abs() < 1e-6);
        assert!((out[[1, 1]]).abs() < 1e-6);
    }

    // ---- RoPE cache --------------------------------------------------------

    fn mini_cfg(head_dim: usize, max_len: usize, rope_theta: f64) -> ModelConfig {
        use serde_json::json;
        let v = json!({
            "vocab_size": 1, "hidden_size": head_dim * 2,
            "intermediate_size": 64, "num_hidden_layers": 1,
            "num_attention_heads": 2, "num_key_value_heads": 2,
            "max_position_embeddings": max_len,
            "rms_norm_eps": 1e-6, "rope_theta": rope_theta,
            "bos_token_id": 0, "eos_token_id": 1, "torch_dtype": "float32"
        });
        serde_json::from_value(v).unwrap()
    }

    #[test]
    fn rope_cache_shape() {
        let cfg = mini_cfg(64, 128, 10000.0);
        let cache = RopeCache::new(&cfg);
        assert_eq!(cache.cos.dim(), (128, 32));
        assert_eq!(cache.sin.dim(), (128, 32));
    }

    #[test]
    fn rope_pos0_is_identity() {
        // At position 0 all angles are 0, so cos=1, sin=0.
        // Applying RoPE at pos 0 should leave the vector unchanged.
        let cfg = mini_cfg(4, 16, 10000.0);
        let cache = RopeCache::new(&cfg);
        // x: [seq=1, heads=1, head_dim=4]
        let mut x = Array3::<f32>::zeros((1, 1, 4));
        x[[0, 0, 0]] = 1.0;
        x[[0, 0, 1]] = 2.0;
        x[[0, 0, 2]] = 3.0;
        x[[0, 0, 3]] = 4.0;
        let out = cache.apply(&x, 0);
        for i in 0..4 {
            assert!((out[[0, 0, i]] - x[[0, 0, i]]).abs() < 1e-6, "dim {i}");
        }
    }

    #[test]
    fn rope_rotation_is_reversible() {
        // Apply RoPE then apply the inverse rotation (negate the angle)
        // and check we recover the original.
        // Inverse RoPE: x0' = x0*cos + x1*sin,  x1' = -x0*sin + x1*cos
        let cfg = mini_cfg(4, 16, 10000.0);
        let cache = RopeCache::new(&cfg);
        let mut x = Array3::<f32>::zeros((1, 1, 4));
        x[[0, 0, 0]] = 1.5;
        x[[0, 0, 1]] = -0.5;
        x[[0, 0, 2]] = 2.0;
        x[[0, 0, 3]] = 0.3;

        let pos = 7;
        let rotated = cache.apply(&x, pos);

        // Manually invert using the cached cos/sin.
        let half = 2;
        for i in 0..half {
            let c = cache.cos[[pos, i]];
            let sn = cache.sin[[pos, i]];
            let r0 = rotated[[0, 0, i]];
            let r1 = rotated[[0, 0, i + half]];
            let orig0 = r0 * c + r1 * sn;
            let orig1 = -r0 * sn + r1 * c;
            assert!((orig0 - x[[0, 0, i]]).abs() < 1e-5, "i={i} orig0");
            assert!((orig1 - x[[0, 0, i + half]]).abs() < 1e-5, "i={i} orig1");
        }
    }

    #[test]
    fn rope_cos_sin_at_known_angle() {
        // With head_dim=2, half=1 and rope_theta=1.0 the single frequency is 1.
        // At position p, angle = p * 1/1.0^0 = p * 1.0.
        let cfg = mini_cfg(2, 8, 1.0);
        let cache = RopeCache::new(&cfg);
        for pos in 0..4usize {
            let expected_cos = (pos as f32).cos();
            let expected_sin = (pos as f32).sin();
            assert!((cache.cos[[pos, 0]] - expected_cos).abs() < 1e-6, "pos={pos}");
            assert!((cache.sin[[pos, 0]] - expected_sin).abs() < 1e-6, "pos={pos}");
        }
    }

    #[test]
    fn rope_multi_seq_positions() {
        // With a 2-token sequence starting at pos 0, each token gets its own
        // rotation; check that the two output tokens differ.
        let cfg = mini_cfg(4, 16, 10000.0);
        let cache = RopeCache::new(&cfg);
        let mut x = Array3::<f32>::zeros((2, 1, 4));
        x[[0, 0, 0]] = 1.0; x[[0, 0, 2]] = 1.0;
        x[[1, 0, 0]] = 1.0; x[[1, 0, 2]] = 1.0; // same values, different positions
        let out = cache.apply(&x, 0);
        // Position 0 → identity; position 1 → rotated.
        // Token 0 and token 1 should differ (unless the angle is a multiple of 2π).
        assert!(
            (out[[0, 0, 0]] - out[[1, 0, 0]]).abs() > 1e-4,
            "tokens at pos 0 and 1 should differ after RoPE"
        );
    }

    // ---- Real-config smoke test -------------------------------------------
    #[test]
    fn rope_real_config_shape() {
        let home = match std::env::var("HOME") {
            Ok(h) => h,
            Err(_) => return,
        };
        let dir = std::path::PathBuf::from(home)
            .join(".cache/myllm/Qwen/Qwen2.5-0.5B-Instruct");
        if !dir.join("config.json").exists() {
            return;
        }
        let cfg = ModelConfig::from_dir(&dir).unwrap();
        let cache = RopeCache::new(&cfg);
        // head_dim = 896/14 = 64, half = 32, max_len = 32768
        assert_eq!(cache.cos.dim(), (32768, 32));
        assert_eq!(cache.sin.dim(), (32768, 32));
    }
}
