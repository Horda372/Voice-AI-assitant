import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight  # <--- WAŻNY NOWY IMPORT
from pathlib import Path
import json

# 1. Ładowanie konfiguracji
try:
    config = json.load(open("config.json"))
    COMMANDS = config["commands"]
except:
    # Fallback jeśli config nie istnieje
    COMMANDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]

SPECS_DIR = Path("data/prepared_dataset/mel_specs")

# Dodajemy klasy specjalne
if "silence" not in COMMANDS: COMMANDS.append("silence")
if "unknown" not in COMMANDS: COMMANDS.append("unknown")


def load_dataset(specs_dir):
    X, y = [], []
    files = list(specs_dir.glob("*.npy"))

    print(f"Znaleziono {len(files)} plików. Rozpoczynam wczytywanie...")

    # Liczniki dla statystyki
    counts = {}

    for f in files:
        # Wyciąganie etykiety
        label = f.stem.split("__")[0]

        # --- NAPRAWA ETYKIET ---
        # Jeśli plik nazywa się 'noise', przypisz go do 'silence'
        if "noise" in label:
            label = "silence"
        # -----------------------

        # Jeśli etykieta jest na liście komend, używamy jej
        if label in COMMANDS:
            target_label = label
        else:
            target_label = "unknown"

        # Statystyka
        counts[target_label] = counts.get(target_label, 0) + 1

        spec = np.load(f)
        X.append(spec)
        y.append(target_label)

    print("\n--- LICZEBNOŚĆ KLAS ---")
    for cls, count in counts.items():
        print(f"  {cls}: {count}")
    print("-----------------------\n")

    return np.array(X), np.array(y)


# 2. Przygotowanie danych
X, y = load_dataset(SPECS_DIR)

# Dodanie kanału dla CNN (wymagane: [samples, height, width, 1])
X = X[..., np.newaxis]

# Kodowanie etykiet (tekst -> liczby)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)

# Podział na trening i walidację
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# --- KLUCZOWE: OBLICZANIE WAG KLAS ---
# To rozwiązuje problem małej ilości szumu (380 plików).
# Model dostanie "karę" za ignorowanie mało licznych klas.
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
# Konwersja na słownik wymagany przez Keras {0: 1.5, 1: 0.8, ...}
class_weights_dict = dict(enumerate(class_weights))

print("Wagi dla poszczególnych klas (wyższe = ważniejsza klasa):")
for i, weight in class_weights_dict.items():
    print(f"  {label_encoder.classes_[i]}: {weight:.2f}")

# 3. Architektura Modelu CNN
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
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 4. Trenowanie z uwzględnieniem wag
print(f"\nTrenowanie dla klas: {label_encoder.classes_}")

history = model.fit(
    X_train, y_train,
    epochs=25,  # Trochę więcej epok, bo trudniej się uczy z wagami
    validation_data=(X_val, y_val),
    batch_size=32,
    class_weight=class_weights_dict  # <--- TUTAJ DZIEJE SIĘ MAGIA
)

# 5. Zapis
model.save("voice_command_model.h5")
np.save("classes.npy", label_encoder.classes_)

print("✔ Model zapisany. Klasa 'silence' powinna być teraz poprawnie rozpoznawana.")

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