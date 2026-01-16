import numpy as np
import sounddevice as sd
import tensorflow as tf
import librosa
import json
import time
import paho.mqtt.client as mqtt

# --- KONFIGURACJA MQTT ---
MQTT_BROKER = "broker.emqx.io" # Adres
MQTT_PORT = 1883
MQTT_TOPIC = "AI/voice" # Temat, na którym nasłuchuje urządzenie

# --- INICJALIZACJA MQTT ---
client = mqtt.Client()
try:
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    print(f"Połączono z brokerem MQTT: {MQTT_BROKER}")
except Exception as e:
    print(f"Błąd połączenia MQTT: {e}")

# --- KONFIGURACJA Modelu---
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

# --- LOGIKA STEROWANIA ---
last_digit = None
last_digit_time = 0
TIMEOUT = 5.0  # Masz 5 sekund na powiedzenie on/off po numerze


def process_command(cmd):
    global last_digit, last_digit_time

    # Mapowanie komend (jeśli model zwraca np. 'one' zamiast '1')
    digits = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
              "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9"}

    # 1. Sprawdź czy to cyfra
    if cmd in digits:
        last_digit = digits[cmd]
        last_digit_time = time.time()
        print(f"Wybrano pokój: {last_digit}. Czekam na komendę on/off...")

    # 2. Sprawdź czy to on/off
    elif cmd in ["on", "off"]:
        if last_digit and (time.time() - last_digit_time < TIMEOUT):
            payload = f"{last_digit} {cmd}"
            client.publish(MQTT_TOPIC, payload)
            print(f"Wysłano MQTT: [{MQTT_TOPIC}] -> {payload}")
            last_digit = None  # Reset po wysłaniu
        else:
            print("Najpierw podaj numer(0-9)!")

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
                        # Reset bufora, aby nie wykryć tego samego słowa w kolejnym przesunięciu
                        audio_buffer = np.zeros(window_samples, dtype=np.float32)
                        time.sleep(0.3)

if __name__ == "__main__":
    start_live_recognition()