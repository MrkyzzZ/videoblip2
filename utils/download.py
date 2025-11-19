from huggingface_hub import snapshot_download
path = snapshot_download(repo_id="laion/clap-htsat-fused", cache_dir="./pretrained/clap-htsat-fused", token="hf_xxx")
print("Saved model at", path)