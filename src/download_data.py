import os
import requests
from huggingface_hub import hf_hub_download

def download_datasets():
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    datasets = [
        {
            "repo_id": "xiaowu0162/longmemeval-cleaned",
            "filename": "longmemeval_s_cleaned.json",
            "target": "data/longmemeval_s_cleaned.json"
        },
        {
            "repo_id": "xiaowu0162/longmemeval-cleaned",
            "filename": "longmemeval_m_cleaned.json",
            "target": "data/longmemeval_m_cleaned.json"
        }
    ]
    
    for ds in datasets:
        if not os.path.exists(ds["target"]):
            print(f"Downloading {ds['filename']} from {ds['repo_id']}...")
            try:
                hf_hub_download(
                    repo_id=ds["repo_id"],
                    filename=ds["filename"],
                    local_dir=data_dir,
                    repo_type="dataset"
                )
                print(f"Successfully downloaded {ds['filename']}.")
            except Exception as e:
                print(f"Error downloading {ds['filename']}: {e}")
                print(f"Attempting direct HTTP download...")
                url = f"https://huggingface.co/datasets/{ds['repo_id']}/resolve/main/{ds['filename']}"
                response = requests.get(url)
                if response.status_code == 200:
                    with open(ds["target"], "wb") as f:
                        f.write(response.content)
                    print(f"Successfully downloaded {ds['filename']} via HTTP.")
                else:
                    print(f"HTTP download failed with status {response.status_code}")
        else:
            print(f"File {ds['target']} already exists. Skipping.")

if __name__ == "__main__":
    download_datasets()
