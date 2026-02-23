import os
import requests
from tqdm import tqdm

# --- Configuration ---
MODELS_DIR = "models"
MODELS = {
    "llama-3-8b": {
        "url": "https://huggingface.co/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
        "path": os.path.join(MODELS_DIR, "llama-3-8b-instruct.Q4_K_M.gguf")
    },
    "alexa-wakeword": {
        "url": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/alexa_v0.1.onnx",
        "path": os.path.join(MODELS_DIR, "alexa_v0.1.onnx")
    }
}

# --- Core Logic ---

def download_model(url: str, save_path: str):
    if os.path.exists(save_path):
        print(f"✔ Model already exists at: {save_path}")
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"\n⬇ Downloading from:\n{url}")
    response = requests.get(url, stream=True)

    if response.status_code != 200:
        print(f"❌ Failed to download. Status code: {response.status_code}")
        return

    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1KB

    progress_bar = tqdm(total=total_size, unit="iB", unit_scale=True)

    with open(save_path, "wb") as f:
        for chunk in response.iter_content(block_size):
            if chunk:
                f.write(chunk)
                progress_bar.update(len(chunk))

    progress_bar.close()

    if total_size != 0 and progress_bar.n != total_size:
        print("❌ ERROR: Download incomplete.")
    else:
        print(f"✅ Download complete: {save_path}")

# --- Execution ---

print("🚀 Starting Aira Model Downloader...")
os.makedirs(MODELS_DIR, exist_ok=True)

for name, config in MODELS.items():
    print(f"\n--- Processing {name} ---")
    download_model(config["url"], config["path"])

print("\n🎯 All model checks complete.")
