import numpy as np
import sounddevice as sd
import tensorflow as tf
import librosa
import json
import time

# --- KONFIGURACJA ---
MODEL_PATH = "voice_command_model.h5"
CLASSES_PATH = "classes.npy"
CONFIG_PATH = "config.json"

# Parametry audio (muszą być identyczne jak przy treningu)
TARGET_SR = 16000
DURATION = 1.0
N_MELS = 80
N_FFT = 400
HOP_LENGTH = 160
THRESHOLD = 0.85
STEP_SIZE = 0.2  # Przesunięcie okna o 200ms

# --- ŁADOWANIE ZASOBÓW ---
print("Wczytywanie modelu...")
model = tf.keras.models.load_model(MODEL_PATH)
classes = np.load(CLASSES_PATH)
config = json.load(open(CONFIG_PATH))


def preprocess_audio(audio_data):
    """Przetwarzanie surowego dźwięku na spektrogram."""
    mel = librosa.feature.melspectrogram(
        y=audio_data, sr=TARGET_SR, n_fft=N_FFT,
        hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)

    # Normalizacja
    mean = np.mean(log_mel)
    std = np.std(log_mel) + 1e-9
    log_mel = (log_mel - mean) / std

    # Przygotowanie kształtu pod sieć CNN (1, Szerokość, Wysokość, 1)
    input_data = log_mel.T
    input_data = np.expand_dims(input_data, axis=(0, -1))
    return input_data


def start_live_recognition():
    window_samples = int(TARGET_SR * DURATION)
    step_samples = int(TARGET_SR * STEP_SIZE)
    audio_buffer = np.zeros(window_samples, dtype=np.float32)

    print(f"\nSystem gotowy. Nasłuchiwanie komend: {list(classes)}")

    try:
        with sd.InputStream(samplerate=TARGET_SR, channels=1, dtype='float32') as stream:
            while True:
                # 1. Pobranie fragmentu dźwięku
                new_chunk, _ = stream.read(step_samples)
                new_chunk = new_chunk.flatten()

                # 2. Przesunięcie okna (Sliding Window)
                audio_buffer = np.roll(audio_buffer, -step_samples)
                audio_buffer[-step_samples:] = new_chunk

                # 3. VAD - Sprawdzenie czy nie ma ciszy
                if np.max(np.abs(audio_buffer)) > 0.03:

                    # 4. Predykcja
                    input_tensor = preprocess_audio(audio_buffer)
                    prediction = model.predict(input_tensor, verbose=0)

                    idx = np.argmax(prediction)
                    prob = prediction[0][idx]
                    command = classes[idx]

                    # 5. Wykrycie komendy
                    if prob > THRESHOLD:
                        print(f"Rozpoznano: {command} ({prob:.2f})")

                        # --- MIEJSCE NA MQTT ---
                        # Tutaj dodamy funkcję wysyłającą wiadomość

                        # Reset bufora, aby nie wykryć tego samego słowa w kolejnym przesunięciu
                        audio_buffer = np.zeros(window_samples, dtype=np.float32)
                        time.sleep(0.3)

    except KeyboardInterrupt:
        print("\nZatrzymano nasłuchiwanie.")


if __name__ == "__main__":
    start_live_recognition()