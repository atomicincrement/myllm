# Build an LLM with ndarray

## File layout

```
src/
  main.rs          — entry point: ties everything together, runs the chat loop
  download.rs      — HTTP fetching of model files from Hugging Face
  config.rs        — deserialise config.json into a ModelConfig struct
  safetensors.rs   — load .safetensors weight files into ndarray arrays
  tokenizer.rs     — BPE tokenizer: encode text → token ids, decode ids → text
  transformer.rs   — all neural-network layers and the full forward pass
  sample.rs        — autoregressive sampling (greedy / top-p / temperature)
```

## Steps

1. **Add dependencies to Cargo.toml**
   Add `ndarray` (n-dimensional arrays), `reqwest` (HTTP, blocking + rustls-tls features), `serde`/`serde_json` (JSON deserialisation), `safetensors` (weight file format), and `tokenizers` (HuggingFace fast tokenizer). Also add `indicatif` for download progress bars and `anyhow` for ergonomic error handling.

2. **Fetch the Qwen model files over HTTP** (`download.rs`)
   Download `config.json`, `tokenizer.json`, `tokenizer_config.json`, and `model.safetensors` (or sharded `model-00001-of-NNNNN.safetensors` files) from the Hugging Face Hub using reqwest. Cache files to a local directory so they are not re-downloaded on subsequent runs. Verify file sizes / checksums against the Hub metadata.

3. **Parse the model config** (`config.rs`)
   Deserialise `config.json` into a `ModelConfig` struct using serde. Key fields: `vocab_size`, `hidden_size`, `intermediate_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads` (for GQA), `max_position_embeddings`, `rms_norm_eps`, `rope_theta`. This struct is passed to every other module so dimensions are consistent.

4. **Load model weights from safetensors** (`safetensors.rs`)
   Memory-map the `.safetensors` file(s) and deserialise each tensor into an `ndarray::ArrayD<f32>`. Build a `Weights` struct with named fields (e.g. `embed_tokens`, per-layer `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`, `input_layernorm`, `post_attention_layernorm`, `norm`, `lm_head`) so the transformer code can access them by name rather than by string key.

5. **Implement the tokenizer** (`tokenizer.rs`)
   Wrap the HuggingFace `tokenizers` crate to load `tokenizer.json`. Expose `encode(text) -> Vec<u32>` and `decode(ids) -> String`. Also expose constants for special token ids: `bos_token_id`, `eos_token_id`, `pad_token_id`. The Qwen tokenizer uses a custom BPE vocabulary; the `tokenizers` crate handles this transparently.

6. **Implement transformer building blocks** (`transformer.rs`)
   * **RMS layer normalization** — normalize each vector by its root-mean-square, then scale by a learned weight vector. No bias, no mean subtraction (unlike LayerNorm).
   * **Rotary positional embeddings (RoPE)** — precompute `cos`/`sin` tables up to `max_position_embeddings` for the head dimension. Apply by splitting each query/key head into pairs and rotating them. Qwen uses a specific `rope_theta` (often 1 000 000).
   * **Grouped-query attention (GQA) with KV cache** — project input to Q (num_attention_heads), K and V (num_key_value_heads). Repeat K/V heads to match Q head count. Apply RoPE to Q and K. Compute scaled dot-product attention with a causal mask. Append K/V to a growing cache for each layer so past tokens are not recomputed.
   * **SwiGLU feed-forward MLP** — two parallel linear projections (`gate_proj`, `up_proj`) followed by element-wise `SiLU(gate) * up`, then a third projection (`down_proj`) back to `hidden_size`. `intermediate_size` is typically ~2.67× `hidden_size`.

7. **Implement the full forward pass** (`transformer.rs`)
   Embedding lookup (`embed_tokens`) → for each of the N decoder layers: input RMS norm → GQA attention (with residual) → post-attention RMS norm → MLP (with residual) → final RMS norm → linear `lm_head` projection to `vocab_size` logits. The output is the logit vector for the next token position.

8. **Implement autoregressive sampling** (`sample.rs`)
   Given the logit vector, support: **greedy** (argmax), **temperature scaling** (divide logits by T before softmax), and **top-p (nucleus) sampling** (sort by probability, keep the smallest set whose cumulative probability exceeds p, then sample). Expose a `sample(logits, config) -> u32` function. Stop generation when `eos_token_id` is produced or a maximum token count is reached.

9. **Format the prompt with the Qwen chat template** (`main.rs`)
   Qwen uses a specific chat template: wrap the system message in `<|im_start|>system\n…<|im_end|>` and each user/assistant turn similarly. Encode the formatted prompt with the tokenizer, run the forward pass once over the entire prompt (prefill), then enter the decode loop calling the forward pass one token at a time and feeding the sampled token back in.

10. **Wire everything together and test** (`main.rs`)
    Parse CLI arguments (model directory, temperature, top-p, max new tokens). Instantiate config → weights → tokenizer. Run the chat template + decode loop. Test with the prompt "How big is an elephant?" and verify the model produces a coherent factual answer. Add basic benchmarking (tokens/second).


