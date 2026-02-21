from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel
from llama_cpp import Llama
import torchaudio
import tempfile
import os
import torch
import numpy as np
import threading
from download_model import check_model
from wakeword import run_wakeword

app = FastAPI()

# --- INITIALIZATION ---

# Ensure model is downloaded before initializing LLM
check_model()

# Load Whisper Model (CPU Optimized)
# tiny.en is recommended for speed on CPU
whisper_model = WhisperModel("tiny.en", device="cpu", compute_type="int8")

# Load LLM (Qwen2-1.5B GGUF)
MODEL_PATH = os.path.join("models", "qwen2-1_5b-instruct-q4.gguf")
# Initialize LLM only if file exists, to avoid startup failure
llm = None
if os.path.exists(MODEL_PATH):
    llm = Llama(model_path=MODEL_PATH, n_ctx=2048, n_threads=8)
else:
    print(f"WARNING: LLM model not found at {MODEL_PATH}. Chat features will be limited.")

# --- BACKGROUND TASKS ---

@app.on_event("startup")
async def startup_event():
    # Start Wake Word Listener in a background thread
    print("Starting Wake Word Listener in background...")
    threading.Thread(target=run_wakeword, daemon=True).start()

# --- UTILITIES ---

def detect_intent(text: str) -> str:
    """Rule-based intent classifier."""
    text = text.lower()
    
    # OS / System commands
    system_words = ["shutdown", "restart", "open", "close", "launch", "exit"]
    if any(word in text for word in system_words):
        return "system_command"
    
    # Simple greetings or very short chat
    if len(text.split()) < 4:
        return "fast_chat"
    
    return "smart_llm"

def run_llm(prompt: str, max_tokens: int = 150) -> str:
    """Helper to run the local LLM."""
    if not llm:
        return "I'm sorry, my brain (LLM) is currently offline. Please check the model file."
    
    output = llm(
        f"<|im_start|>system\nYou are AIRA, a helpful and intelligent AI assistant. Keep your responses concise and natural.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
        max_tokens=max_tokens,
        stop=["<|im_end|>", "</s>"],
        echo=False
    )
    return output["choices"][0]["text"].strip()

def route_engine(text: str):
    """Router Engine that directs traffic to specialized executors."""
    intent = detect_intent(text)
    
    if intent == "system_command":
        # Placeholder for real Command Executor logic
        return {"response": f"Acknowledged. Executing system command for: '{text}'", "type": "command"}
    
    elif intent == "fast_chat":
        # Faster response for simple queries
        response = run_llm(f"Reply briefly to: {text}", max_tokens=50)
        return {"response": response, "type": "fast_chat"}
    
    else:
        # Detailed thinking for complex queries
        response = run_llm(text, max_tokens=300)
        return {"response": response, "type": "smart_llm"}

# --- ENDPOINTS ---

# Root and Health Check
@app.get("/")
def root():
    return {"message": "AIRA is LIVE! 🚀"}

@app.get("/healthz")
async def health_check():
    return {"status": "ok", "llm_loaded": llm is not None}

@app.post("/listen")
async def listen(file: UploadFile = File(...)):
    try:
        # 1. Speech-to-Text (ASR)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Transcribe using faster-whisper
        segments, info = whisper_model.transcribe(tmp_path, beam_size=5)
        text = " ".join([segment.text for segment in segments]).strip()
        
        os.remove(tmp_path)

        if not text:
            return {"error": "No speech detected"}

        # 2. Intelligence Flow (Classifier -> Router)
        result = route_engine(text)

        return {
            "input_text": text,
            "response": result["response"],
            "type": result["type"],
            "language": info.language
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
