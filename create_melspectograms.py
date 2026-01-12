import os
import numpy as np
import librosa
from pathlib import Path

# ---------------- CONFIG ----------------
TARGET_SR = 16000       # Hz
N_MELS = 80             # liczba pasm mel
N_FFT = 400             # 25 ms okno
HOP_LENGTH = 160        # 10 ms przesunięcie
FMIN = 0
FMAX = 8000

COMBINED_DIR = Path("data/prepared_dataset/combined")
OUTPUT_DIR = Path("data/prepared_dataset/mel_specs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# ----------------------------------------

audio_files = [f for f in COMBINED_DIR.glob("*.wav")]
print(f"Znaleziono {len(audio_files)} plików audio w {COMBINED_DIR}")

def compute_mel_spectrogram(audio, sr=TARGET_SR):
    # mel-spektrogram (moc)
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=N_FFT,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0
    )
    # log-scale
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel

def normalize_mel(mel):
    mean = np.mean(mel)
    std = np.std(mel) + 1e-9
    return (mel - mean) / std

for file_path in audio_files:
    audio, _ = librosa.load(file_path, sr=TARGET_SR, mono=True)

    mel = compute_mel_spectrogram(audio)
    mel = normalize_mel(mel)

    out_name = file_path.stem + ".npy"
    np.save(OUTPUT_DIR / out_name, mel)

print(f"✔ Zapisano {len(audio_files)} mel-spektrogramów w {OUTPUT_DIR}")
