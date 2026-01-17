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

# KLUCZOWA ZMIANA: Dodajemy klasy pomocnicze do listy komend
# 'silence' (szum tła) oraz 'unknown' (słowa spoza listy komend)
COMMANDS = config["commands"]
if "silence" not in COMMANDS: COMMANDS.append("silence")
if "unknown" not in COMMANDS: COMMANDS.append("unknown")


def load_dataset(specs_dir):
    X, y = [], []
    files = list(specs_dir.glob("*.npy"))

    print(f"Znaleziono {len(files)} plików. Rozpoczynam wczytywanie...")

    for f in files:
        # Wyciąganie etykiety (np. "yes__filename.npy")
        label = f.stem.split("__")[0]

        # Jeśli etykieta jest na liście komend, używamy jej bezpośrednio
        # Jeśli nie, przypisujemy ją do klasy 'unknown'
        if label in COMMANDS:
            target_label = label
        else:
            target_label = "unknown"

        spec = np.load(f)


        X.append(spec)
        y.append(target_label)

    return np.array(X), np.array(y)


# 2. Przygotowanie danych
X, y = load_dataset(SPECS_DIR)
X = X[..., np.newaxis]  # Dodanie kanału dla CNN

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)

X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 3. Architektura Modelu CNN (Ulepszona o BatchNormalization)
input_shape = X_train.shape[1:]

model = models.Sequential([
    layers.Input(shape=input_shape),

    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),  # Zwiększony dropout dla lepszej generalizacji
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 4. Trenowanie
print(f"Trenowanie dla klas: {label_encoder.classes_}")
# Zwiększona liczba epok, aby model lepiej nauczył się rozróżniać szum
history = model.fit(X_train, y_train, epochs=20,
                    validation_data=(X_val, y_val), batch_size=32)

# 5. Zapis
model.save("voice_command_model.h5")
np.save("classes.npy", label_encoder.classes_)

print("✔ Model z klasami 'silence' i 'unknown' zapisany.")

# Opcjonalnie: Rysowanie wykresu (jeśli masz matplotlib)
try:
    import matplotlib.pyplot as plt

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig('training_history.png')
    print("✔ Wykres zapisany jako training_history.png")
except ImportError:
    pass