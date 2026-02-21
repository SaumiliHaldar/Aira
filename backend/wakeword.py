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

def listen_for_wake_word(threshold=0.5):
    """
    Continously listen for the 'Hey Aira' wake word.
    """
    # Load openwakeword model
    # Note: openwakeword has built-in models for common words.
    # We will use the default 'alexa' or similar and map it to 'Aira' for now, 
    # or look for a custom one if available.
    model = Model(wakeword_models=["alexa"], inference_framework="onnx")
    
    print(f"AIRA is listening for 'Hey Aira'...")
    
    def callback(indata, frames, time_info, status):
        if status:
            print(status)
        
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
        with sd.InputStream(samplerate=RATE, channels=CHANNELS, callback=callback, blocksize=CHUNK_SIZE):
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping AIRA Wake Word listener...")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AIRA Wake Word Listener")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection threshold (0.0 to 1.0)")
    args = parser.parse_args()
    
    listen_for_wake_word(threshold=args.threshold)
