mod download;

const MODEL_REPO: &str = "Qwen/Qwen2.5-0.5B-Instruct";

fn main() -> anyhow::Result<()> {
    let cache_dir = download::fetch_model(MODEL_REPO)?;
    println!("Model files ready at: {}", cache_dir.display());
    Ok(())
}
