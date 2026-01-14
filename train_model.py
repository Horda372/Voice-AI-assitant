import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import json

# 1. Ładowanie konfiguracji
config = json.load(open("config.json"))
SPECS_DIR = Path("data/prepared_dataset/mel_specs")
COMMANDS = config["commands"]

def load_dataset(specs_dir):
    X, y = [], []
    files = list(specs_dir.glob("*.npy"))
    
    for f in files:
        # Wyciąganie etykiety z nazwy pliku (np. "yes__filename.npy")
        label = f.stem.split("__")[0]
        if label in COMMANDS:
            spec = np.load(f)
            X.append(spec)
            y.append(label)
    
    return np.array(X), np.array(y)

# 2. Przygotowanie danych
print("Wczytywanie danych...")
X, y = load_dataset(SPECS_DIR)

# Dodanie wymiaru kanału (wymagane przez CNN: Batch, Height, Width, Channels)
X = X[..., np.newaxis]

# Kodowanie etykiet (tekst -> liczby)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(COMMANDS)

X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 3. Architektura Modelu CNN
input_shape = X_train.shape[1:] # np. (80, 101, 1)

model = models.Sequential([
    layers.Input(shape=input_shape),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 4. Trenowanie
print("Rozpoczynanie trenowania...")
history = model.fit(X_train, y_train, epochs=15, 
                    validation_data=(X_val, y_val), batch_size=32)

# 5. Zapis modelu
model.save("voice_command_model.h5")
np.save("classes.npy", label_encoder.classes_)
print("✔ Model zapisany jako voice_command_model.h5")