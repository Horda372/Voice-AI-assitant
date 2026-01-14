import librosa
import numpy as np
import tensorflow as tf
from pathlib import Path

# Parametry zgodne z create_melspectograms.py
TARGET_SR = 16000
N_MELS = 80
N_FFT = 400
HOP_LENGTH = 160

def preprocess_audio(path):
    # Wczytanie i standaryzacja długości do 1s
    audio, _ = librosa.load(path, sr=TARGET_SR)
    if len(audio) > TARGET_SR:
        audio = audio[:TARGET_SR]
    else:
        audio = np.pad(audio, (0, max(0, TARGET_SR - len(audio))))
    
    # Generowanie spektrogramu
    mel = librosa.feature.melspectrogram(
        y=audio, sr=TARGET_SR, n_fft=N_FFT, 
        hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    
    # Normalizacja (zgodna z create_melspectograms.py)
    mean = np.mean(log_mel)
    std = np.std(log_mel) + 1e-9
    log_mel = (log_mel - mean) / std
    
    # Zwracamy czysty spektrogram 2D (80, 101)
    return log_mel

# Załadowanie modelu i klas
model = tf.keras.models.load_model("voice_command_model.h5")
classes = np.load("classes.npy")

def predict_command(file_path):
    processed_audio = preprocess_audio(file_path)
    # Zmiana wymiarów na (1, 80, 101, 1) - Batch size 1
    input_data = processed_audio[np.newaxis, ..., np.newaxis]
    
    prediction = model.predict(input_data, verbose=0)
    class_idx = np.argmax(prediction)
    return classes[class_idx], prediction[0][class_idx]

# Przykład użycia
file_to_test = "data/dataset/yes/0a7c2a8d_nohash_0.wav" # Podaj ścieżkę do pliku
cmd, prob = predict_command(file_to_test)
print(f"Rozpoznana komenda: {cmd} (Prawdopodobieństwo: {prob:.2f})")