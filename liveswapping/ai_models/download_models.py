import hashlib
import requests
from pathlib import Path
from tqdm import tqdm
import os

MODELS = {
    "reswapper128": {
        "url": "https://github.com/NeuroDonu/LiveSwapping/releases/download/models/reswapper-1019500.pth",
        "sha256": "212092b199452b736e8f80c16e2b578f233ab3f592334ca15f22d351f27461a7",
    },
    "reswapper256": {
        "url": "https://github.com/NeuroDonu/LiveSwapping/releases/download/models/reswapper_256-1567500.pth",
        "sha256": "db059a5cbc9d1c4c98320f15b524492fbd4747caefe20929796338c9f4ee5bd4",
    },
    "inswapper128": {
        "url": "https://github.com/NeuroDonu/LiveSwapping/releases/download/models/inswapper_128.onnx",
        "sha256": "e4a3f08c753cb72d04e10aa0f7dbe3deebbf39567d4ead6dce08e98aa49e16af",
    },
    "gfpgan": {
        "url": "https://github.com/NeuroDonu/LiveSwapping/releases/download/models/GFPGANv1.3.pth",
        "sha256": "c953a88f2727c85c3d9ae72e2bd4846bbaf59fe6972ad94130e23e7017524a70",
    },
}

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def download_with_resume(url: str, dest: Path, chunk_size: int = 8 * 1024 * 1024):
    """Download file with resume capability and progress bar (8MB chunks)."""
    headers = {}
    resume_pos = 0
    
    # Check if partial file exists
    if dest.exists():
        resume_pos = dest.stat().st_size
        headers['Range'] = f'bytes={resume_pos}-'
        print(f"[RESUME] Resuming download from {resume_pos} bytes")
    
    try:
        response = requests.get(url, headers=headers, stream=True, timeout=30)
        response.raise_for_status()
        
        # Get total file size
        if 'content-range' in response.headers:
            total_size = int(response.headers['content-range'].split('/')[-1])
        elif 'content-length' in response.headers:
            total_size = int(response.headers['content-length']) + resume_pos
        else:
            total_size = None
            
        mode = 'ab' if resume_pos > 0 else 'wb'
        
        with dest.open(mode) as f, tqdm(
            desc=f"[DOWNLOAD] {dest.name}",
            total=total_size,
            initial=resume_pos,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            miniters=1
        ) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
                    
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Download error: {e}")
        raise


def ensure_model(name: str):
    info = MODELS.get(name)
    if not info:
        raise ValueError(f"Unknown model: {name}")
    dest = MODELS_DIR / Path(info["url"]).name
    
    # Check if file already exists and is valid
    if dest.exists() and _sha256(dest) == info["sha256"]:
        #print(f"[SUCCESS] Model {name} already downloaded.")
        return dest
    
    print(f"[DOWNLOAD] Downloading model {name}...")
    
    # Ensure directory exists
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    # Download with resume capability
    try:
        download_with_resume(info["url"], dest)
        
        # Verify checksum
        print(f"[VERIFY] Verifying checksum for {name}...")
        if _sha256(dest) != info["sha256"]:
            dest.unlink(missing_ok=True)
            raise RuntimeError(f"[ERROR] Checksum mismatch for {name}!")
            
        print(f"[SUCCESS] Model {name} downloaded successfully!")
        return dest
        
    except Exception as e:
        # Clean up partial download on failure
        if dest.exists():
            dest.unlink(missing_ok=True)
        raise RuntimeError(f"[ERROR] Failed to download {name}: {e}")


def main():
    for name in MODELS:
        ensure_model(name) 