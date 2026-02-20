from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from transformers import pipeline
import torchaudio
import tempfile
import os
import torch

app = FastAPI()

# Root and Health Check
@app.get("/")
def root():
    return {"message": "Voia is LIVE!🚀"}

@app.get("/healthz")
async def health_check():
    return {"status": "ok"}


# Initialize Hugging Face pipeline
asr = pipeline("automatic-speech-recognition", model="openai/whisper-tiny.en")
# asr = pipeline("automatic-speech-recognition", model="openai/whisper-medium.en")


# Voice to Text (Listen)
@app.post("/listen")
async def listen(file: UploadFile = File(...)):
    try:
        # Save the uploaded audio to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Load waveform with torchaudio (ffmpeg not needed)
        waveform, sample_rate = torchaudio.load(tmp_path)

        # ✅ Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # ✅ Resample to 16k if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=16000
            )
            waveform = resampler(waveform)
            sample_rate = 16000

        result = asr(
            {
                "array": waveform.squeeze().numpy(),
                "sampling_rate": sample_rate
            },
            return_timestamps=True   # required for >30s
        )

        # Clean up temp file
        os.remove(tmp_path)

        return {
            "text": result["text"],
            "timestamps": result.get("chunks", [])
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
        