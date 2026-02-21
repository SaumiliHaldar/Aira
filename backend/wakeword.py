import sounddevice as sd
import numpy as np
from openwakeword.model import Model
import argparse
import time

# --- CONFIGURATION ---
WAKE_WORD = "aira" # We can use custom or pre-trained models
CHANNELS = 1
RATE = 16000
CHUNK_SIZE = 1280 # Required for openwakeword

def get_input_device():
    """Finds a valid input device, prioritizing microphones."""
    devices = sd.query_devices()
    default_input = sd.default.device[0]
    
    # Priority 1: Check if default is valid
    if default_input != -1:
        return default_input
        
    # Priority 2: Look for 'mic' or 'microphone'
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0 and 'mic' in dev['name'].lower():
            print(f"Using preferred microphone: {dev['name']} (ID: {i})")
            return i

    # Priority 3: Fallback to any input device
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            print(f"Using fallback input device: {dev['name']} (ID: {i})")
            return i
            
    return None

def run_wakeword(threshold=0.5):
    """
    Continously listen for the 'Hey Aira' wake word.
    """
    device_id = get_input_device()
    if device_id is None:
        print("Error: No audio input device found.")
        return

    # Load openwakeword model
    model = Model(wakeword_models=["alexa"], inference_framework="onnx")
    
    print(f"AIRA is listening for 'Hey Aira'...")
    
    def callback(indata, frames, time_info, status):
        if status:
            print(f"Status: {status}")
        
        # Openwakeword expects int16 audio
        audio_int16 = (indata * 32768).astype(np.int16).flatten()
        
        # Get predictions
        prediction = model.predict(audio_int16)
        
        for key, score in prediction.items():
            if score > threshold:
                print(f"\n[!] WAKE WORD DETECTED: {key} (Score: {score:.2f})")
                print(">>> Activating AIRA Intelligence Flow...")
                # Here we could trigger a beep or notify the backend
                # For now, we'll just log it.
                time.sleep(1) # Cooldown

    try:
        with sd.InputStream(samplerate=RATE, 
                            channels=CHANNELS, 
                            callback=callback, 
                            blocksize=CHUNK_SIZE,
                            device=device_id):
            while True:
                time.sleep(0.1)
    except Exception as e:
        print(f"Wake Word Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AIRA Wake Word Listener")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection threshold (0.0 to 1.0)")
    args = parser.parse_args()
    
    run_wakeword(threshold=args.threshold)
