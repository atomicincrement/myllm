//! Download model files from the Hugging Face Hub and cache them locally.
//!
//! Files are stored under `~/.cache/myllm/<repo_owner>/<repo_name>/`.
//! A file is not re-downloaded if it already exists at the cached path.
//!
//! # Hugging Face URL format
//! `https://huggingface.co/{repo_id}/resolve/{revision}/{filename}`

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use reqwest::blocking::Client;

const HF_BASE: &str = "https://huggingface.co";
const DEFAULT_REVISION: &str = "main";

/// All the files that must be present for the model to run.
///
/// If the repo uses sharded weights (model.safetensors.index.json exists)
/// the shard file names are discovered at runtime and added alongside these.
pub const REQUIRED_FILES: &[&str] = &[
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
];

/// Name of the single-file weight artifact.
pub const WEIGHTS_FILE: &str = "model.safetensors";

/// Name of the shard index (only present in sharded repos).
pub const SHARD_INDEX_FILE: &str = "model.safetensors.index.json";

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Download every file needed to run `repo_id` and return the local cache
/// directory that contains them.
///
/// `repo_id` is the Hugging Face repository in `"owner/name"` form,
/// e.g. `"Qwen/Qwen2.5-0.5B-Instruct"`.
pub fn fetch_model(repo_id: &str) -> Result<PathBuf> {
    let client = Client::new();
    let cache_dir = cache_dir_for(repo_id)?;
    fs::create_dir_all(&cache_dir)
        .with_context(|| format!("creating cache dir {}", cache_dir.display()))?;

    // 1. Always grab the metadata / config files.
    for name in REQUIRED_FILES {
        fetch_file(&client, repo_id, name, &cache_dir)?;
    }

    // 2. Try to detect whether the weights are sharded.
    let index_path = cache_dir.join(SHARD_INDEX_FILE);
    let maybe_index = try_fetch_file(&client, repo_id, SHARD_INDEX_FILE, &cache_dir)?;

    if maybe_index {
        // Sharded: parse the index and download every unique shard.
        let shard_names = shard_names_from_index(&index_path)?;
        for shard in &shard_names {
            fetch_file(&client, repo_id, shard, &cache_dir)?;
        }
        eprintln!(
            "Downloaded {} shard(s) for {}",
            shard_names.len(),
            repo_id
        );
    } else {
        // Single file.
        fetch_file(&client, repo_id, WEIGHTS_FILE, &cache_dir)?;
        eprintln!("Downloaded weights for {}", repo_id);
    }

    Ok(cache_dir)
}

/// Return the local path for a single named file within the model cache,
/// downloading it only if not already present.
pub fn fetch_file(
    client: &Client,
    repo_id: &str,
    filename: &str,
    cache_dir: &Path,
) -> Result<PathBuf> {
    let dest = cache_dir.join(filename);
    if dest.exists() {
        return Ok(dest);
    }

    let url = hf_url(repo_id, filename);
    eprintln!("Downloading {} …", url);

    let mut response = client
        .get(&url)
        .send()
        .with_context(|| format!("GET {}", url))?
        .error_for_status()
        .with_context(|| format!("HTTP error for {}", url))?;

    // Create any parent directories the filename might include (e.g. shards).
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)?;
    }

    let mut file =
        fs::File::create(&dest).with_context(|| format!("creating {}", dest.display()))?;

    response
        .copy_to(&mut file)
        .with_context(|| format!("writing {}", dest.display()))?;

    file.flush()?;
    Ok(dest)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Like `fetch_file`, but returns `false` (without error) on a 404.
/// Returns `true` if the file was already cached or was downloaded successfully.
fn try_fetch_file(
    client: &Client,
    repo_id: &str,
    filename: &str,
    cache_dir: &Path,
) -> Result<bool> {
    let dest = cache_dir.join(filename);
    if dest.exists() {
        return Ok(true);
    }

    let url = hf_url(repo_id, filename);
    let response = client
        .get(&url)
        .send()
        .with_context(|| format!("GET {}", url))?;

    if response.status() == reqwest::StatusCode::NOT_FOUND {
        return Ok(false);
    }

    let mut response = response
        .error_for_status()
        .with_context(|| format!("HTTP error for {}", url))?;

    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)?;
    }

    let mut file =
        fs::File::create(&dest).with_context(|| format!("creating {}", dest.display()))?;

    response
        .copy_to(&mut file)
        .with_context(|| format!("writing {}", dest.display()))?;

    file.flush()?;
    Ok(true)
}

/// Build the Hugging Face CDN URL for a file in a repo.
fn hf_url(repo_id: &str, filename: &str) -> String {
    format!(
        "{}/{}/resolve/{}/{}",
        HF_BASE, repo_id, DEFAULT_REVISION, filename
    )
}

/// Compute the local cache directory for a repo.
///
/// Layout: `~/.cache/myllm/<owner>/<name>/`
fn cache_dir_for(repo_id: &str) -> Result<PathBuf> {
    let home = std::env::var("HOME").context("$HOME not set")?;
    // repo_id is "owner/name" — replace '/' with a path separator.
    let rel: PathBuf = repo_id.split('/').collect();
    Ok(PathBuf::from(home).join(".cache").join("myllm").join(rel))
}

/// Parse `model.safetensors.index.json` and return a deduplicated, sorted list
/// of shard file names.
///
/// The index JSON has the form:
/// ```json
/// { "metadata": { … }, "weight_map": { "tensor_name": "shard_file.safetensors", … } }
/// ```
fn shard_names_from_index(index_path: &Path) -> Result<Vec<String>> {
    let text = fs::read_to_string(index_path)
        .with_context(|| format!("reading {}", index_path.display()))?;

    let v: serde_json::Value =
        serde_json::from_str(&text).context("parsing safetensors shard index")?;

    let weight_map = v["weight_map"]
        .as_object()
        .context("weight_map missing or not an object in shard index")?;

    let mut names: Vec<String> = weight_map
        .values()
        .filter_map(|v| v.as_str().map(str::to_owned))
        .collect();

    names.sort();
    names.dedup();
    Ok(names)
}
