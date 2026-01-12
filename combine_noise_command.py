import os
import json
import numpy as np
import librosa
import soundfile as sf
import random
from pathlib import Path

# ---------------- CONFIG ----------------

# bezpieczne wczytanie JSON w relacji do skryptu
CONFIG_PATH = Path(__file__).parent / "config.json"
config = json.load(open(CONFIG_PATH, "r", encoding="utf-8"))
config = json.load(open("config.json", "r", encoding="utf-8"))

print("Zawartość config:", config)
print("Klucze config:", list(config.keys()))

print("Sprawdzam config przed użyciem combined_path:")
print(config)
print("combined_path" in config)
print(config.get("combined_path", "BRAK"))


# foldery
BASE_DIR = Path(config["prepared_dataset_path"])
COMMANDS_DIR = Path(config["commands_path"])
NOISE_DIR = Path(config["noise_path"])
OUTPUT_DIR = Path(config["combined_path"])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# parametry
TARGET_SR = config["sample_rate"]  # 16000 Hz
SNR_MIN = 5
SNR_MAX = 20
# ----------------------------------------

# lista plików
command_files = [f for f in os.listdir(COMMANDS_DIR) if f.endswith(".wav")]
noise_files = [f for f in os.listdir(NOISE_DIR) if f.endswith(".wav")]

assert len(command_files) > 0, "Brak plików w commands/"
assert len(noise_files) > 0, "Brak plików w noise/"

# funkcja miksowania
def add_noise(clean, noise, snr_db):
    clean_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)
    snr_linear = 10 ** (snr_db / 10)
    scale = np.sqrt(clean_power / (snr_linear * noise_power))
    return clean + scale * noise

# przetwarzanie
for cmd_file in command_files:
    cmd_path = COMMANDS_DIR / cmd_file

    # losowy szum
    noise_file = random.choice(noise_files)
    noise_path = NOISE_DIR / noise_file

    # wczytanie + resampling
    clean, _ = librosa.load(cmd_path, sr=TARGET_SR, mono=True)
    noise, _ = librosa.load(noise_path, sr=TARGET_SR, mono=True)

    # zabezpieczenie długości (1 sekunda)
    min_len = min(len(clean), len(noise))
    clean = clean[:min_len]
    noise = noise[:min_len]

    # losowy SNR
    snr = random.uniform(SNR_MIN, SNR_MAX)

    # miks
    mixed = add_noise(clean, noise, snr)

    # ochrona przed clippingiem
    mixed = mixed / max(1.0, np.max(np.abs(mixed)))

    # zapis
    out_name = cmd_file.replace(".wav", "_noisy.wav")
    out_path = OUTPUT_DIR / out_name

    sf.write(out_path, mixed, TARGET_SR)

print(f"✔ Zapisano {len(command_files)} plików w {OUTPUT_DIR}")
