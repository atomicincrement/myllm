mod config;
mod download;
mod safetensors;
mod tokenizer;

const MODEL_REPO: &str = "Qwen/Qwen2.5-0.5B-Instruct";

fn main() -> anyhow::Result<()> {
    let cache_dir = download::fetch_model(MODEL_REPO)?;
    let cfg = config::ModelConfig::from_dir(&cache_dir)?;
    let weights = safetensors::Weights::from_dir(&cache_dir, &cfg)?;
    println!(
        "embed_tokens: {:?}  norm: {:?}  layers: {}",
        weights.embed_tokens.dim(),
        weights.norm.dim(),
        weights.layers.len()
    );
    Ok(())
}
