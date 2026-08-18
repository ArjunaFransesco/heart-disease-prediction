import os
import csv
import requests

DATA_URL = "https://raw.githubusercontent.com/mrdbourke/zero-to-mastery-ml/master/data/heart-disease.csv"
OUTPUT_FILE = os.path.join("data", "raw", "heart.csv")

def download_data(url: str = DATA_URL, output_path: str = OUTPUT_FILE):
    """Downloads clinical heart disease dataset from remote URL."""
    print(f"Downloading clinical dataset from {url}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    res = requests.get(url)
    if res.status_code == 200:
        with open(output_path, "wb") as f:
            f.write(res.content)
        print(f"Dataset successfully saved to {output_path}")
    else:
        raise Exception(f"Failed to download data. HTTP Status: {res.status_code}")

def load_data(file_path: str = OUTPUT_FILE):
    """Loads CSV into pandas DataFrame if pandas is installed, otherwise list of dicts."""
    if not os.path.exists(file_path):
        download_data(output_path=file_path)
    try:
        import pandas as pd
        return pd.read_csv(file_path)
    except ImportError:
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return list(reader)

if __name__ == "__main__":
    data = load_data()
    print("Dataset loaded successfully!")
    if hasattr(data, 'shape'):
        print(f"Shape: {data.shape}")
    else:
        print(f"Total records: {len(data)}, Sample row: {data[0]}")
