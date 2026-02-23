import os
import asyncio
import threading
import wave
import queue
import sounddevice as sd
import numpy as np
import time
import webbrowser
import pyautogui
import edge_tts
import pygame
from fastapi import FastAPI
from pydantic import BaseModel
from faster_whisper import WhisperModel
from llama_cpp import Llama
from openwakeword.model import Model
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
MODEL_PATH = "models/llama-3-8b-instruct.Q4_K_M.gguf"
WAKEWORD_MODEL = "alexa"
STT_MODEL_SIZE = "base"
VOICE = "en-US-AriaNeural"

app = FastAPI(title="Aira Voice Assistant")

# --- Core Aira Engine ---

class AiraEngine:
    def __init__(self):
        print("Initializing Aira (Consolidated Edition)...")

        # STT
        self.stt_model = WhisperModel(
            STT_MODEL_SIZE,
            device="cpu",
            compute_type="int8"
        )

        # LLM
        self.llm = None
        if os.path.exists(MODEL_PATH):
            print(f"Loading local LLM from {MODEL_PATH}...")
            self.llm = Llama(
                model_path=MODEL_PATH,
                n_ctx=2048,
                n_threads=4,
                verbose=False
            )
        else:
            print("⚠ LLM model not found.")

        # TTS
        pygame.mixer.init()

        # Wake word
        wakeword_path = "models/alexa_v0.1.onnx"
        if os.path.exists(wakeword_path):
            self.oww_model = Model(
                wakeword_models=[wakeword_path],
                inference_framework="onnx"
            )
        else:
            print("⚠ Wake word model not found.")
            self.oww_model = Model(
                wakeword_models=[WAKEWORD_MODEL],
                inference_framework="onnx"
            )

    # ---------------- STT ----------------
    def transcribe(self, audio_path):
        segments, _ = self.stt_model.transcribe(audio_path, beam_size=5)
        return " ".join([segment.text for segment in segments]).strip()

    # ---------------- LLM ----------------
    def get_nlp_response(self, text):
        if not self.llm:
            return "My local model is not loaded."

        prompt = (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{text}"
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )

        output = self.llm(
            prompt,
            max_tokens=150,
            stop=["<|eot_id|>"],
            echo=False
        )

        return output["choices"][0]["text"].strip()

    # ---------------- TTS ----------------
    async def speak(self, text):
        print(f"Aira: {text}")

        output_file = "response.mp3"
        communicate = edge_tts.Communicate(text, VOICE)
        await communicate.save(output_file)

        pygame.mixer.music.load(output_file)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

    # ---------------- RECORD COMMAND ----------------
    def record_command(self, filename="command.wav"):
        print("Listening for command...")

        CHUNK = 1024
        CHANNELS = 1
        RATE = 16000

        audio_q = queue.Queue()

        def callback(indata, frames, time_info, status):
            audio_q.put(indata.copy())

        frames = []
        last_voice_time = time.time()
        recording_started = False

        with sd.InputStream(
            samplerate=RATE,
            channels=CHANNELS,
            dtype='int16',
            blocksize=CHUNK,
            device=self.mic_index,
            callback=callback
        ):
            while True:
                chunk = audio_q.get()
                frames.append(chunk)

                amplitude = np.abs(chunk).max()

                if amplitude > 500:
                    last_voice_time = time.time()
                    recording_started = True

                if recording_started and (time.time() - last_voice_time) > 1.5:
                    break

                if not recording_started and len(frames) > int(RATE / CHUNK * 5):
                    break

        audio_data = np.concatenate(frames, axis=0)

        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(RATE)
            wf.writeframes(audio_data.tobytes())

        return filename

    # ---------------- LOGIC ----------------
    async def run_logic(self):
        audio_file = self.record_command()
        text = self.transcribe(audio_file)

        if not text:
            return

        print(f"User: {text}")
        low_text = text.lower()

        if "open" in low_text and ("browser" in low_text or "google" in low_text):
            webbrowser.open("https://www.google.com")
            await self.speak("Opening browser.")

        elif "close" in low_text:
            pyautogui.hotkey('alt', 'f4')
            await self.speak("Closing application.")

        elif "restart" in low_text:
            await self.speak("Restarting in five seconds.")
            os.system("shutdown /r /t 5")

        elif "shutdown" in low_text:
            await self.speak("Shutting down in five seconds.")
            os.system("shutdown /s /t 5")

        else:
            response = self.get_nlp_response(text)
            await self.speak(response)

    # ---------------- LISTEN LOOP ----------------
    def listen_loop(self):
        print(f"Aira is active. Say '{WAKEWORD_MODEL}' to start.")

        try:
            devices = sd.query_devices()
            self.mic_index = None

            for i, d in enumerate(devices):
                if "Microphone" in d["name"] and d["max_input_channels"] > 0:
                    self.mic_index = i
                    break

            if self.mic_index is None:
                print("❌ No microphone found.")
                return

            device_info = devices[self.mic_index]
            samplerate = int(device_info["default_samplerate"])

            print(f"[listen_loop] Using: {device_info['name']} (index {self.mic_index})")
            print(f"[listen_loop] Native samplerate: {samplerate}")

        except Exception as e:
            print(f"Audio device error: {e}")
            return

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(status)

        # Convert to mono if needed
        if indata.ndim > 1:
            audio_frame = indata[:, 0]
        else:
            audio_frame = indata

        audio_frame = audio_frame.astype(np.int16)

        self.oww_model.predict(audio_frame)

        for mdl in self.oww_model.prediction_buffer.keys():
            if list(self.oww_model.prediction_buffer[mdl])[-1] > 0.6:
                print("Wake word detected!")
                asyncio.run(self.run_logic())

    try:
        with sd.InputStream(
            device=self.mic_index,
            samplerate=samplerate,   # use native rate
            channels=device_info["max_input_channels"],  # use native channels
            callback=audio_callback
        ):
            print("[listen_loop] Microphone stream open. Listening...")
            threading.Event().wait()

    except Exception as e:
        print(f"Stream error: {e}")

# Instantiate
engine = AiraEngine()


# ----- FASTAPI setup -----
@app.on_event("startup")
async def startup_event():
    threading.Thread(
        target=engine.listen_loop,
        daemon=True
    ).start()

# Root and Health Check
@app.get("/")
def root():
    return {"message": "Aira is LIVE 🚀"}

@app.get("/healthz")
async def health_check():
    return {"status": "ok"}

# Manual command execution
class ManualCommand(BaseModel):
    text: str

@app.post("/execute")
async def manual_execute(cmd: ManualCommand):
    # This allows manual triggering via the API
    response = engine.get_nlp_response(cmd.text)
    await engine.speak(response)
    return {"response": response}
