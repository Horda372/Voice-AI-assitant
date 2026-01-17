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

# --- KONFIGURACJA Modelu ---
MODEL_PATH = "voice_command_model.h5"
CLASSES_PATH = "classes.npy"
CONFIG_PATH = "config.json"

# Parametry audio (muszą być identyczne jak przy treningu)
TARGET_SR = 16000
DURATION = 1.0
N_MELS = 80
N_FFT = 400
HOP_LENGTH = 160

# Parametry detekcji
THRESHOLD = 0.85  # Pewność modelu (0.0 - 1.0)
SILENCE_THRESHOLD = 0.1  # Ignorowanie cichych dźwięków (szum tła)
STEP_SIZE = 0.2  # Przesunięcie okna co 200ms
COOLDOWN = 1.0  # Czas blokady po wykryciu komendy (żeby nie wykrywał 5 razy tego samego)

# --- ŁADOWANIE ZASOBÓW ---
print("Wczytywanie modelu...")
model = tf.keras.models.load_model(MODEL_PATH)
classes = np.load(CLASSES_PATH)
config = json.load(open(CONFIG_PATH))

# --- ZMIENNE GLOBALNE STANU ---
last_digit = None  # Ostatnia usłyszana cyfra
last_digit_time = 0  # Czas usłyszenia ostatniej cyfry
TIMEOUT = 5.0  # Czas na powiedzenie on/off po cyfrze
last_recognition_time = 0  # Do cooldownu


def select_audio_device():
    """Wyświetla listę urządzeń audio i pozwala użytkownikowi wybrać."""
    print("\n--- DOSTĘPNE URZĄDZENIA AUDIO ---")
    devices = sd.query_devices()
    input_devices = []

    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            input_devices.append(i)
            print(f"ID {i}: {dev['name']} (Kanały: {dev['max_input_channels']})")

    default_device = sd.default.device[0]
    print(f"\nDomyślne urządzenie ID: {default_device}")

    choice = input("Podaj ID urządzenia (lub wciśnij ENTER dla domyślnego): ")

    if choice.strip() == "":
        return default_device

    try:
        device_id = int(choice)
        if device_id in input_devices:
            return device_id
        else:
            print("Nieprawidłowe ID. Używam domyślnego.")
            return default_device
    except ValueError:
        print("Błąd wprowadzania. Używam domyślnego.")
        return default_device


def process_command(cmd, probability):
    """Logika sterowania: Cyfra -> On/Off"""
    global last_digit, last_digit_time

    # Mapowanie słów na cyfry
    digits_map = {
        "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
        "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9"
    }

    current_time = time.time()

    # SCENARIUSZ 1: Wykryto cyfrę
    if cmd in digits_map:
        last_digit = digits_map[cmd]
        last_digit_time = current_time
        print(f"2[AKTYWACJA] Wybrano numer: {last_digit}. Czekam na komendę 'on' lub 'off'...")

    # SCENARIUSZ 2: Wykryto komendę ON/OFF
    elif cmd in ["on", "off"]:
        # Sprawdzamy, czy mamy zapamiętaną cyfrę i czy nie minął czas (5s)
        if last_digit is not None:
            time_diff = current_time - last_digit_time

            if time_diff < TIMEOUT:
                # Sukces - wysyłamy MQTT
                payload = f"{last_digit} {cmd}"
                client.publish(MQTT_TOPIC, payload)
                print(f"[SUKCES] Wysłano komendę: {payload}")
                print(f"   (Opóźnienie od cyfry: {time_diff:.2f}s)")

                last_digit = None  # Resetujemy stan po wykonaniu
            else:
                print(f"[TIMEOUT] Upłynął czas na komendę (minęło {time_diff:.1f}s). Powiedz cyfrę ponownie.")
                last_digit = None
        else:
            print(f"[IGNOROWANE] Słyszę '{cmd}', ale najpierw musisz podać numer (0-9).")

    # SCENARIUSZ 3: Inne słowa (np. bed, house) - ignorujemy w logice sterowania
    else:
        print(f"   (Słyszano: {cmd}, ale to nie jest komenda sterująca)")


def preprocess_audio(audio_data):
    """Przetwarzanie surowego dźwięku na spektrogram (identycznie jak w treningu)."""
    mel = librosa.feature.melspectrogram(
        y=audio_data, sr=TARGET_SR, n_fft=N_FFT,
        hop_length=HOP_LENGTH, n_mels=N_MELS, power=2.0
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)

    # Normalizacja standardowa (taka sama jak w create_melspectograms.py)
    mean = np.mean(log_mel)
    std = np.std(log_mel) + 1e-9
    log_mel = (log_mel - mean) / std

    input_data = log_mel[np.newaxis, ..., np.newaxis]
    return input_data


def start_live_recognition():
    global last_recognition_time

    # 1. Wybór mikrofonu
    device_id = select_audio_device()
    print(f"\nUruchamianie nasłuchu na urządzeniu ID: {device_id}...")

    # Bufor kołowy
    window_samples = int(TARGET_SR * DURATION)
    step_samples = int(TARGET_SR * STEP_SIZE)
    audio_buffer = np.zeros(window_samples, dtype=np.float32)

    print(f"Gotowy! Dostępne klasy: {list(classes)}")
    print(f"Powiedz cyfrę (0-9), a następnie 'on' lub 'off'.")

    with sd.InputStream(device=device_id, samplerate=TARGET_SR, channels=1, dtype='float32') as stream:
        while True:
            # 1. Pobranie fragmentu dźwięku
            new_chunk, overflow = stream.read(step_samples)
            if overflow:
                print("Warning: Audio overflow")

            new_chunk = new_chunk.flatten()

            # 2. Aktualizacja bufora (przesuwanie okna)
            audio_buffer = np.roll(audio_buffer, -step_samples)
            audio_buffer[-step_samples:] = new_chunk

            # 3. Bramka szumów (VAD - Voice Activity Detection)
            # Jeśli dźwięk jest zbyt cichy, w ogóle nie uruchamiamy modelu
            current_volume = np.max(np.abs(audio_buffer))
            if current_volume > SILENCE_THRESHOLD:

                # 4. Predykcja
                input_tensor = preprocess_audio(audio_buffer)
                prediction = model.predict(input_tensor, verbose=0)

                idx = np.argmax(prediction)
                prob = prediction[0][idx]
                command = classes[idx]

                # 5. Interpretacja wyniku
                current_time = time.time()

                # Sprawdzamy THRESHOLD, ignorujemy 'silence'/'unknown' i sprawdzamy COOLDOWN
                if (prob > THRESHOLD and
                        command not in ["silence", "unknown"] and
                        (current_time - last_recognition_time > COOLDOWN)):
                    print(f"\n🎙️ Rozpoznano: '{command}' (Pewność: {prob:.2f})")
                    process_command(command, prob)

                    # Ustawiamy czas ostatniej detekcji, żeby nie wykrywać tego samego słowa 5 razy pod rząd
                    last_recognition_time = current_time

                    # Opcjonalnie: czyścimy bufor, żeby nie łapać końcówki słowa w następnym oknie
                    # audio_buffer = np.zeros(window_samples, dtype=np.float32)


if __name__ == "__main__":
    try:
        start_live_recognition()
    except KeyboardInterrupt:
        print("\nZatrzymano program.")
        client.disconnect()