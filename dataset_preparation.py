# %%
"""
Dataset Preparation Script
Prepares audio recordings for training a voice command recognition model.
"""
import json
import random
from collections import Counter
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from utils.logger import log_header

# %%
log_header("Load config file")
config = json.load(open("config.json"))

TARGET_DURATION = config["duration"]  # 1 second
TARGET_SR = config["sample_rate"]  # 16000 Hz
TARGET_PER_CLASS = config["target_per_class"]  # 3000
COMMANDS = config["commands"]
DATASET_PATH = Path(config["original_dataset_path"])
OUTPUT_COMMANDS = Path(config["commands_path"])
OUTPUT_NOISE = Path(config["noise_path"])
METADATA_FILE = config["metadata_file"]

# Create output folders
OUTPUT_COMMANDS.mkdir(parents=True, exist_ok=True)
OUTPUT_NOISE.mkdir(parents=True, exist_ok=True)


# %%
log_header("Create helping functions")


def pad_or_trim(audio, target_length):
    if len(audio) == target_length:
        return audio
    elif len(audio) < target_length:
        pad_total = target_length - len(audio)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        return np.pad(audio, (pad_left, pad_right))
    else:
        trim_total = len(audio) - target_length
        trim_left = trim_total // 2
        return audio[trim_left : trim_left + target_length]


def normalize(audio, peak=0.9):
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        return audio / max_val * peak
    return audio


# %%
log_header("Loading background noise")

noise_folder = DATASET_PATH / "_background_noise_"
noise_files = list(noise_folder.glob("*.wav"))

noises = {}
for f in noise_files:
    audio, _ = librosa.load(str(f), sr=TARGET_SR)
    noises[f.stem] = audio
    print(f"  Loaded: {f.stem} ({len(audio)/TARGET_SR:.1f}s)")

noise_names = list(noises.keys())
print(f"Total: {len(noises)} noise files")


# %%
log_header("Finding recordings")

all_recordings = {}
for cmd in COMMANDS:
    folder = DATASET_PATH / cmd
    if folder.exists():
        files = list(folder.glob("*.wav"))
        all_recordings[cmd] = files
        print(f"  {cmd}: {len(files)} files")

# %%
log_header("Filtering outliers")

durations = []
for cmd, files in all_recordings.items():
    for f in files:
        info = sf.info(str(f))
        durations.append(info.duration)

mean_dur = np.mean(durations)
std_dur = np.std(durations)
min_dur = mean_dur - 2 * std_dur
max_dur = mean_dur + 2 * std_dur

print(f"Mean duration: {mean_dur:.3f}s")
print(f"Acceptable range: {min_dur:.3f}s - {max_dur:.3f}s")

# Keep only files with acceptable duration
filtered_recordings = {}
total_before = 0
total_after = 0

for cmd, files in all_recordings.items():
    good_files = []
    for f in files:
        info = sf.info(str(f))
        if min_dur <= info.duration <= max_dur:
            good_files.append(f)
    filtered_recordings[cmd] = good_files
    total_before += len(files)
    total_after += len(good_files)

print(f"Kept {total_after}/{total_before} recordings")


# %%
log_header("Processing commands")

target_samples = int(TARGET_DURATION * TARGET_SR)
metadata = []

for cmd in sorted(filtered_recordings.keys()):
    files = filtered_recordings[cmd]

    # If too many, randomly select
    if len(files) > TARGET_PER_CLASS:
        files_to_process = random.sample(files, TARGET_PER_CLASS)
        print(f"\n{cmd}: {len(filtered_recordings[cmd])} -> {TARGET_PER_CLASS}")
    else:
        files_to_process = files
        print(f"\n{cmd}: {len(filtered_recordings[cmd])}")

    for i, f in enumerate(files_to_process):
        # Load audio
        audio, _ = librosa.load(str(f), sr=TARGET_SR)

        # Make exactly 1 second
        audio = pad_or_trim(audio, target_samples)

        # Normalize volume
        audio = normalize(audio)

        # Save
        out_name = f"{cmd}__{f.stem}__{i:04d}.wav"
        out_path = OUTPUT_COMMANDS / out_name
        sf.write(str(out_path), audio, TARGET_SR)

        # Save metadata
        metadata.append(
            {
                "path": str(out_path),
                "original": str(f),
                "command": cmd,
            }
        )

    print("  Done!")

print(f"\nTotal processed: {len(metadata)}")


# %%
log_header("Processing noise segments")

noise_metadata = []

for name, audio in noises.items():
    num_segments = len(audio) // target_samples
    print(f"  {name}: {num_segments} segments")

    for i in range(num_segments):
        segment = audio[i * target_samples : (i + 1) * target_samples]
        segment = normalize(segment)

        out_name = f"noise__{name}_{i:04d}.wav"
        out_path = OUTPUT_NOISE / out_name
        sf.write(str(out_path), segment, TARGET_SR)

        noise_metadata.append({"path": str(out_path), "type": name, "segment": i})

print(f"Total noise segments: {len(noise_metadata)}")


# %%
log_header("Saving metadata")

all_metadata = {
    "commands": metadata,
    "noise": noise_metadata,
    "info": {
        "duration": TARGET_DURATION,
        "sample_rate": TARGET_SR,
        "per_class": TARGET_PER_CLASS,
    },
}

with open(METADATA_FILE, "w") as f:
    json.dump(all_metadata, f, indent=2)

print(f"Saved to: {METADATA_FILE}")


# %%
log_header("SUMMARY")

counts = Counter([m["command"] for m in metadata])
print("\nRecordings per command:")
for cmd, count in sorted(counts.items()):
    print(f"  {cmd}: {count}")

print(f"\nTotal commands: {len(metadata)}")
print(f"Total noise: {len(noise_metadata)}")
print("\nDone!")
