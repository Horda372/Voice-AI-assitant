import numpy as np
from pathlib import Path
import json
import random
from collections import Counter

# CONFIG
SPECS_DIR = Path("data/prepared_dataset/mel_specs")
config = json.load(open("config.json"))
COMMANDS = config["commands"]

def get_speaker_id(filename_stem):
    parts = filename_stem.split("__")
    if len(parts) >= 2:
        # Example: "yes__0a9f9af7_nohash_0__0000" -> "0a9f9af7"
        return parts[1].split("_")[0]
    return "unknown"

print("--- DIAGNOSIS START ---")

# 1. Load Filenames & Speakers
files = list(SPECS_DIR.glob("*.npy"))
data = []
for f in files:
    label = f.stem.split("__")[0]
    if label in COMMANDS:
        spk = get_speaker_id(f.stem)
        data.append({"file": f.name, "label": label, "speaker": spk})

print(f"Total files loaded: {len(data)}")

if len(data) == 0:
    print("❌ ERROR: No files found! Check your paths.")
    exit()

# 2. Analyze Speaker Extraction
speakers = [d["speaker"] for d in data]
unique_speakers = set(speakers)
print(f"Unique Speakers found: {len(unique_speakers)}")

# Check if parsing failed (e.g., everyone is 'unknown')
if len(unique_speakers) < 5:
    print("❌ CRITICAL WARNING: Very few speakers found!")
    print(f"Sample Speaker IDs: {list(unique_speakers)}")
    print("Check if filename format matches: command__speaker_nohash_num__index.npy")
else:
    print(f"✅ Speaker extraction seems okay. Sample IDs: {list(unique_speakers)[:5]}")

# 3. Simulate the Split
val_split = 0.2
unique_speakers_list = list(unique_speakers)
random.shuffle(unique_speakers_list)
n_val = int(len(unique_speakers_list) * val_split)
val_speakers_set = set(unique_speakers_list[:n_val])

train_data = [d for d in data if d["speaker"] not in val_speakers_set]
val_data = [d for d in data if d["speaker"] in val_speakers_set]

print(f"\n--- SIMULATED SPLIT ---")
print(f"Train Samples: {len(train_data)}")
print(f"Val Samples:   {len(val_data)}")

# 4. Check for Class Imbalance in Validation
print("\n--- CLASS DISTRIBUTION IN VALIDATION ---")
val_labels = [d["label"] for d in val_data]
val_counts = Counter(val_labels)
print(val_counts)

if len(val_counts) < len(COMMANDS):
    print("❌ WARNING: Some classes are missing from Validation set!")
    missing = set(COMMANDS) - set(val_counts.keys())
    print(f"Missing: {missing}")
elif any(v < 5 for v in val_counts.values()):
    print("❌ WARNING: Very few samples for some classes in Validation!")

# 5. Check for Data Leakage (Intersection)
train_speakers = set(d["speaker"] for d in train_data)
val_speakers_actual = set(d["speaker"] for d in val_data)
intersection = train_speakers.intersection(val_speakers_actual)

if len(intersection) > 0:
    print(f"❌ CRITICAL FAILURE: {len(intersection)} speakers exist in BOTH sets!")
else:
    print("✅ SUCCESS: No speaker overlap between Train and Validation.")