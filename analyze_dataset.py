# %%
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf

from utils.logger import log_header

# %%
log_header("Load config file")
config = json.load(open("config.json"))

COMMANDS = config["commands"]
DATASET_PATH = Path(config["original_dataset_path"])

# %%
all_folders = [
    f for f in DATASET_PATH.iterdir() if f.is_dir() and f.name != "_background_noise_"
]
command_folders = [f for f in all_folders if f.name in COMMANDS]


print(f"Command amount: {len(command_folders)}")
print(f"\nCommands: {sorted([f.name for f in command_folders])}")

# %%
data = []
for folder in command_folders:
    wav_files = list(folder.glob("*.wav"))
    data.append({"class": folder.name, "audio_amount": len(wav_files)})

df = pd.DataFrame(data).sort_values("class")

log_header("Amount number for each command:")
print(df.to_string(index=False))


# %%
log_header("Dataset description")
dataset_describe = df.describe()
dataset_describe.loc["sum"] = df["audio_amount"].sum()
dataset_describe.loc["median"] = df["audio_amount"].median()
dataset_describe.loc["std"] = df["audio_amount"].std()
dataset_describe = dataset_describe.round(2)
dataset_describe = dataset_describe.loc[
    ["count", "sum", "mean", "median", "std", "min", "25%", "50%", "75%", "max"]
]
print(dataset_describe)

# %%
plt.figure(figsize=(14, 6))
plt.bar(df["class"], df["audio_amount"])
plt.xlabel("Command")
plt.ylabel("Audio amount")
plt.title("Audio amount for each command")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.grid(axis="y", alpha=0.3)
plt.show()

# %%
log_header("Analyze audio duration")

durations = []
durations_per_class = {folder.name: [] for folder in command_folders}

for folder in command_folders:
    wav_files = list(folder.glob("*.wav"))
    for wav_file in wav_files:
        audio_info = sf.info(str(wav_file))
        duration = audio_info.duration
        durations.append(
            {
                "class": folder.name,
                "file": wav_file.name,
                "duration_s": duration,
                "sample_rate": audio_info.samplerate,
                "channels": audio_info.channels,
            }
        )
        durations_per_class[folder.name].append(duration)

df_durations = pd.DataFrame(durations)
print(df_durations.head())

# %%
log_header("Audio duration statistics:")

print(f"Total recordings analyzed: {len(df_durations)}")
print(f"Mean duration: {df_durations['duration_s'].mean():.3f}s")
print(f"Median duration: {df_durations['duration_s'].median():.3f}s")
print(f"Minimum duration: {df_durations['duration_s'].min():.3f}s")
print(f"Maximum duration: {df_durations['duration_s'].max():.3f}s")
print(f"Standard deviation: {df_durations['duration_s'].std():.3f}s")

# %%
log_header("Sample rate information (sample_rate: audio_amount):")
print(df_durations["sample_rate"].value_counts().to_dict())

print("Channels information (channels_amount: audio_amount):")
print(df_durations["channels"].value_counts().to_dict())

# %%
class_duration_stats = []
for class_name, class_durations in durations_per_class.items():
    if class_durations:
        class_duration_stats.append(
            {
                "class": class_name,
                "mean_duration_s": np.mean(class_durations),
                "median_duration_s": np.median(class_durations),
                "min_duration_s": np.min(class_durations),
                "max_duration_s": np.max(class_durations),
                "std_duration_s": np.std(class_durations),
            }
        )

df_class_durations = pd.DataFrame(class_duration_stats).sort_values("class")
log_header("Duration statistics for each class:")
print(df_class_durations.to_string(index=False))

# %%
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Duration histogram
axes[0].hist(df_durations["duration_s"], bins=50, edgecolor="black", alpha=0.7)
axes[0].set_xlabel("Audio duration (s)")
axes[0].set_ylabel("Audio Count")
axes[0].set_title("Distribution of all recording durations")
axes[0].grid(axis="y", alpha=0.3)
axes[0].axvline(
    df_durations["duration_s"].mean(),
    color="red",
    linestyle="--",
    label=f'Mean: {df_durations["duration_s"].mean():.3f}s',
)
axes[0].axvline(
    df_durations["duration_s"].median(),
    color="green",
    linestyle="--",
    label=f'Median: {df_durations["duration_s"].median():.3f}s',
)
axes[0].legend()

# Box plot for each class
class_names = sorted(durations_per_class.keys())
class_durations_list = [durations_per_class[name] for name in class_names]
axes[1].boxplot(class_durations_list, labels=class_names, vert=True)
axes[1].set_xlabel("Class")
axes[1].set_ylabel("Audio duration (s)")
axes[1].set_title("Distribution of recording durations per class")
axes[1].tick_params(axis="x", rotation=45)
axes[1].grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()

# %%
log_header("Recordings with unusual duration:")

mean_duration = df_durations["duration_s"].mean()
std_duration = df_durations["duration_s"].std()
threshold = 2  # 2 standard deviations

outliers = df_durations[
    (df_durations["duration_s"] < mean_duration - threshold * std_duration)
    | (df_durations["duration_s"] > mean_duration + threshold * std_duration)
]

if len(outliers) > 0:
    print(f"Found {len(outliers)} recordings with unusual duration")
    print("Unusual duration in each class:")
    print(outliers.groupby("class").count()["file"])
    print("List of recordings:")
    print(outliers[["class", "file", "duration_s"]].to_string(index=False))
else:
    print("No recordings with unusual duration found.")
