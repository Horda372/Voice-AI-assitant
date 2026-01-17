import numpy as np
import sounddevice as sd
import tensorflow as tf
import librosa
import json
import time
import paho.mqtt.client as mqtt

# --- KONFIGURACJA MQTT ---
MQTT_BROKER = "broker.emqx.io"
MQTT_PORT = 1883
MQTT_TOPIC = "AI/voice"

# --- INICJALIZACJA MQTT ---
client = mqtt.Client()
try:
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    print(f"Połączono z brokerem MQTT: {MQTT_BROKER}")
except Exception as e:
    print(f"Błąd połączenia MQTT: {e}")

# --- KONFIGURACJA MODELU ---
MODEL_PATH = "voice_command_model.h5"
CLASSES_PATH = "classes.npy"
CONFIG_PATH = "config.json"

# Parametry audio (muszą być identyczne jak w create_melspectograms.py)
TARGET_SR = 16000
DURATION = 1.0
N_MELS = 80
N_FFT = 400
HOP_LENGTH = 160

# Parametry detekcji
THRESHOLD = 0.85  # Minimalna pewność, by uznać komendę
SILENCE_THRESHOLD = 0.02  # Minimalna głośność (amplituda), by w ogóle analizować (zmniejszono z 0.1)
COOLDOWN = 1.0  # Czas blokady po wykryciu (s)

# --- ŁADOWANIE ZASOBÓW ---
print("Wczytywanie modelu...")
model = tf.keras.models.load_model(MODEL_PATH)
classes = np.load(CLASSES_PATH)
# config = json.load(open(CONFIG_PATH)) # Opcjonalne

# --- ZMIENNE STANU ---
last_digit = None
last_digit_time = 0
TIMEOUT = 5.0
last_recognition_time = 0


def select_audio_device():
    """Wybór mikrofonu."""
    print("\n--- DOSTĘPNE URZĄDZENIA AUDIO ---")
    devices = sd.query_devices()
    input_devices = []
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            input_devices.append(i)
            print(f"ID {i}: {dev['name']}")

    default = sd.default.device[0]
    choice = input(f"\nPodaj ID (Enter = domyślne {default}): ")
    return int(choice) if choice.strip() else default


def process_command(cmd, probability):
    """Logika sterowania."""
    global last_digit, last_digit_time

    digits_map = {
        "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
        "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9"
    }

    current_time = time.time()

    # 1. Wykryto cyfrę
    if cmd in digits_map:
        last_digit = digits_map[cmd]
        last_digit_time = current_time
        print(f"   [!] Zapamiętano: {last_digit}. Czekam na 'on'/'off'...")

    # 2. Wykryto komendę ON/OFF
    elif cmd in ["on", "off"]:
        if last_digit is not None:
            if current_time - last_digit_time < TIMEOUT:
                payload = f"{last_digit} {cmd}"
                client.publish(MQTT_TOPIC, payload)
                print(f"   >>> WYSŁANO MQTT: {payload} <<<")
                last_digit = None
            else:
                print("   [Timeout] Minął czas na komendę. Powtórz cyfrę.")
                last_digit = None
        else:
            print("   [Błąd] Najpierw podaj cyfrę.")


def preprocess_audio(audio_data):
    """
    Kluczowa funkcja z zabezpieczeniem przed ciszą.
    Zwraca None, jeśli sygnał to tylko szum tła.
    """
    # 1. Tworzenie mel-spektrogramu
    mel = librosa.feature.melspectrogram(
        y=audio_data, sr=TARGET_SR, n_fft=N_FFT,
        hop_length=HOP_LENGTH, n_mels=N_MELS, power=2.0
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)

    # 2. ZABEZPIECZENIE: Sprawdzenie "płaskości" sygnału (szumu)
    # Cisza ma bardzo małe odchylenie standardowe.
    # Jeśli znormalizujemy ciszę, wyjdą losowe wzory ("six"/"eight").
    current_std = np.std(log_mel)

    # Jeśli odchylenie jest poniżej 3.0 dB, uznajemy to za szum tła i odrzucamy
    if current_std < 3.0:
        return None

    # 3. Normalizacja (tylko jeśli sygnał jest znaczący)
    mean = np.mean(log_mel)
    std = current_std + 1e-9
    log_mel = (log_mel - mean) / std

    # 4. Dopasowanie kształtu (Batch, Height, Width, Channels)
    input_data = log_mel[np.newaxis, ..., np.newaxis]
    return input_data


def start_live_recognition():
    global last_recognition_time
    device_id = select_audio_device()

    # Bufor na 1 sekundę
    window_samples = int(TARGET_SR * DURATION)
    # Przesunięcie o 0.2 sekundy
    step_samples = int(TARGET_SR * 0.2)

    audio_buffer = np.zeros(window_samples, dtype=np.float32)

    print(f"\nNasłuchuję... (Klasy: {classes})")
    print("Mów do mikrofonu.\n")

    with sd.InputStream(device=device_id, channels=1, samplerate=TARGET_SR,
                        blocksize=step_samples, dtype='float32') as stream:
        while True:
            # Pobierz nowe próbki
            new_data, overflow = stream.read(step_samples)
            if overflow: print("!", end="", flush=True)

            # Aktualizacja bufora (FIFO)
            audio_buffer = np.roll(audio_buffer, -step_samples)
            audio_buffer[-step_samples:] = new_data.flatten()

            # --- WSTĘPNA FILTRACJA (Bramka szumów - amplituda) ---
            # Jeśli jest absolutna cisza, nawet nie próbuj liczyć spektrogramu
            if np.max(np.abs(audio_buffer)) < SILENCE_THRESHOLD:
                continue

            # --- PRZETWARZANIE I ZABEZPIECZENIE PRZED SZUMEM ---
            input_tensor = preprocess_audio(audio_buffer)

            # Jeśli preprocess_audio zwróciło None (bo wykryło płaski szum), pomiń
            if input_tensor is None:
                continue

            # --- PREDYKCJA ---
            prediction = model.predict(input_tensor, verbose=0)
            idx = np.argmax(prediction)
            prob = prediction[0][idx]
            command = classes[idx]

            # Wyświetlanie wszystkiego co ma sensowną pewność (dla debugowania)
            if prob > 0.5:
                # print(f"Debug: {command} ({prob:.2f})") # Odkomentuj by widzieć co model "myśli"
                pass

            # --- DECYZJA ---
            current_time = time.time()
            if (prob > THRESHOLD and
                    command not in ["silence", "unknown"] and
                    (current_time - last_recognition_time > COOLDOWN)):
                print(f"\n🎙️ Wykryto: '{command}' ({prob:.2f})")
                process_command(command, prob)
                last_recognition_time = current_time


if __name__ == "__main__":
    try:
        start_live_recognition()
    except KeyboardInterrupt:
        print("\nZatrzymano.")
        client.disconnect()