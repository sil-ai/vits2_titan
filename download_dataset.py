import os
import sys
import tarfile
import requests
from tqdm import tqdm
import shutil
from pathlib import Path

class DatasetDownloader:
    def __init__(self):
        self.dataset_url = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
        self.dataset_name = "downloaded_datasets/LJSpeech-1.1"
        self.dataset_file = f"{self.dataset_name}.tar.bz2"

    def download_file(self, url, filename):
        """Download file with progress bar"""
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        with open(filename, 'wb') as file, tqdm(
            desc=filename,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)

    def extract_tar(self, filename):
        """Extract tar.bz2 file"""
        print(f"Extracting {filename}...")
        with tarfile.open(filename, 'r:bz2') as tar:
            tar.extractall(path='downloaded_datasets')
        print("Extraction completed!")

    def prepare_dataset(self):
        """Main function to download and prepare the dataset"""
        # Create necessary directories
        os.makedirs("downloaded_datasets", exist_ok=True)

        # Download dataset if not exists
        if not os.path.exists(self.dataset_file):
            print(f"Downloading {self.dataset_name} dataset...")
            self.download_file(self.dataset_url, self.dataset_file)

        # Extract if not already extracted
        # if not os.path.exists(self.dataset_name):
        self.extract_tar(self.dataset_file)
        # else:
        #     print(f"Dataset {self.dataset_name} already exists")

        # Create symbolic link
        link_path = os.path.join("downloaded_datasets", "DUMMY1")
        if not os.path.exists(link_path):
            print("Creating symbolic link...")
            os.symlink(os.path.abspath(self.dataset_name), link_path)
            print(f"Symbolic link created at {link_path}")

        print("\nDataset preparation completed!")
        print(f"Dataset location: {os.path.abspath(self.dataset_name)}")
        print(f"Symbolic link: {os.path.abspath(link_path)}")
        metadata_path = os.path.join("downloaded_datasets", "LJSpeech-1.1", "metadata.csv")
        metadata_copy_path = os.path.join("downloaded_datasets", "LJSpeech-1.1", "metadata_copy.csv")
        if os.path.exists(metadata_path):
            print("Copying metadata.csv to metadata_copy.csv...")
            shutil.copy(metadata_path, metadata_copy_path)
            print("Copy completed!")
        else:
            print("metadata.csv not found. Copy operation aborted.")

if __name__ == "__main__":
    downloader = DatasetDownloader()
    downloader.prepare_dataset()