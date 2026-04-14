mod config;
mod download;
mod safetensors;
mod tokenizer;
mod transformer;

const MODEL_REPO: &str = "Qwen/Qwen2.5-0.5B-Instruct";
const MAX_NEW_TOKENS: usize = 200;

/// Format a user turn into a Qwen2 chat prompt and return the full prompt
/// string, including the trailing `<|im_start|>assistant\n` so the model
/// continues into the assistant turn.
fn build_prompt(user_message: &str) -> String {
    format!(
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n\
         <|im_start|>user\n{user_message}<|im_end|>\n\
         <|im_start|>assistant\n"
    )
}

/// Greedy-argmax sampling: return the token id with the highest logit.
fn greedy(logits: &ndarray::Array1<f32>) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i as u32)
        .expect("empty logits")
}

fn main() -> anyhow::Result<()> {
    // ── 1. Load model artifacts ───────────────────────────────────────────
    let cache_dir = download::fetch_model(MODEL_REPO)?;
    let cfg = config::ModelConfig::from_dir(&cache_dir)?;
    let weights = safetensors::Weights::from_dir(&cache_dir, &cfg)?;
    let tok = tokenizer::Tokenizer::from_dir(&cache_dir, cfg.bos_token_id, cfg.eos_token_id)?;

    eprintln!(
        "Loaded: embed={:?}  norm={:?}  layers={}",
        weights.embed_tokens.dim(),
        weights.norm.dim(),
        weights.layers.len()
    );

    // ── 2. Build RoPE cache and KV-cache state ────────────────────────────
    let rope = transformer::RopeCache::new(&cfg);
    let mut state = transformer::InferenceState::new(&cfg);

    // ── 3. Format prompt and encode ───────────────────────────────────────
    const QUESTION: &str = "How big is an elephant?";
    let prompt = build_prompt(QUESTION);
    eprintln!("Prompt: {prompt:?}");

    let prompt_ids = tok.encode(&prompt)?;
    eprintln!("Prompt tokens: {} ids", prompt_ids.len());

    // ── 4. Prefill – process the full prompt in one forward pass ──────────
    let logits = transformer::forward(&prompt_ids, &weights, &rope, &mut state, &cfg);
    let first_token = greedy(&logits);

    // ── 5. Autoregressive decode until EOS or token budget ────────────────
    let mut generated: Vec<u32> = vec![first_token];
    let eos = cfg.eos_token_id;

    if first_token != eos {
        for _ in 1..MAX_NEW_TOKENS {
            let last = *generated.last().unwrap();
            let logits = transformer::forward(&[last], &weights, &rope, &mut state, &cfg);
            let next = greedy(&logits);
            generated.push(next);
            if next == eos {
                break;
            }
        }
    }

    // ── 6. Decode and print ───────────────────────────────────────────────
    let answer = tok.decode(&generated);
    println!("\nQ: {QUESTION}");
    println!("A: {answer}");

    Ok(())
}

