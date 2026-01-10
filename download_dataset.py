import tarfile
from pathlib import Path

import requests

URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
OUTPUT_DIR = Path("data/dataset")
ARCHIVE_PATH = Path("data/speech_commands_v0.02.tar.gz")


def download_file(url, output_path):
    print("Downloading Dataset...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    print("Finished")


def extract_tar_gz(archive_path, output_dir):
    print("Extracting ...")
    output_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=output_dir)
    print("Extracted to:", output_dir.resolve())


Path("data").mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

download_file(URL, ARCHIVE_PATH)
extract_tar_gz(ARCHIVE_PATH, OUTPUT_DIR)
