from huggingface_hub import hf_hub_download
import os

def check_model():
    repo_id = "Qwen/Qwen2-1.5B-Instruct-GGUF"
    filename = "qwen2-1_5b-instruct-q4_k_m.gguf" # Better Q4 version
    local_filename = "qwen2-1_5b-instruct-q4.gguf" # Matching app.py path
    dest_path = os.path.join("models", local_filename)

    if os.path.exists(dest_path):
        print(f"✅ Model already exists at {dest_path}")
        return dest_path

    print(f"Downloading {filename} from {repo_id}...")

    try:
        path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir="models", local_dir_use_symlinks=False)
        # Rename to match expected name in app.py if needed
        if os.path.exists(path) and path != dest_path:
            if os.path.exists(dest_path):
                os.remove(dest_path) # Remove existing to allow rename or just skip
            os.rename(path, dest_path)
        print(f"✅ Model ready at {dest_path}")
        return dest_path
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        print("Please download it manually from: https://huggingface.co/Qwen/Qwen2-1.5B-Instruct-GGUF/blob/main/qwen2-1_5b-instruct-q4_k_m.gguf")
        return None

if __name__ == "__main__":
    check_model()
