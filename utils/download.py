import os
import time
import sys
import logging
from typing import Optional
from huggingface_hub import snapshot_download


def configure_network(proxy: Optional[str] = None, timeout: int = 30):
    """Configure environment for stable HF downloads."""
    # Respect proxy if provided (e.g., http://127.0.0.1:7890)
    if proxy:
        os.environ["http_proxy"] = proxy
        os.environ["https_proxy"] = proxy
        logging.info(f"Using proxy: {proxy}")

    # HuggingFace hub timeout
    os.environ.setdefault("HF_HUB_TIMEOUT", str(timeout))
    # Disable special transfer layer to avoid HTTP/2/SSL quirks
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    # Use certifi CA bundle to avoid SSL issues
    try:
        import certifi
        ca = certifi.where()
        os.environ["SSL_CERT_FILE"] = ca
        os.environ["REQUESTS_CA_BUNDLE"] = ca
        logging.info(f"Using CA bundle: {ca}")
    except Exception:
        logging.warning("certifi not available; continuing without explicit CA bundle.")


def robust_snapshot_download(repo_id: str, local_dir: str, retries: int = 5, backoff: int = 5):
    """Download with simple retry/backoff to mitigate transient SSL errors."""
    os.makedirs(local_dir, exist_ok=True)

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            logging.info(f"Attempt {attempt}/{retries}: downloading {repo_id} -> {local_dir}")
            snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                resume_download=True,
                max_workers=4,
            )
            print(f"Downloaded to: {local_dir}")
            return True
        except Exception as e:
            last_err = e
            logging.warning(f"Download attempt {attempt} failed: {e}")
            # incremental backoff
            time.sleep(backoff * attempt)

    logging.error(f"All attempts failed. Last error: {last_err}")
    return False


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    target_dir = "/root/autodl-tmp/videoblip2/pretrained/flan-t5-large"
    proxy = os.environ.get("HF_PROXY")  # optionally set HF_PROXY=http://127.0.0.1:7890

    configure_network(proxy=proxy, timeout=45)

    ok = robust_snapshot_download(
        repo_id="google/flan-t5-large",
        local_dir=target_dir,
        retries=6,
        backoff=5,
    )

    if not ok:
        logging.error("Snapshot download failed. Tips: check proxy, try again, or download manually via CLI:")
        print("""
pip install -U huggingface_hub
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("google/flan-t5-large", local_dir="/root/autodl-tmp/videoblip2/pretrained/flan-t5-large", local_dir_use_symlinks=False, resume_download=True)
PY
        """)
        sys.exit(1)


if __name__ == "__main__":
    main()