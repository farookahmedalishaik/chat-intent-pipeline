# 9) push_evaluation_artifacts.py

import os
from huggingface_hub import HfApi, login

# Import settings from the central config file
from config import (
    HF_TOKEN,
    HF_DATASET_REPO_ID,
    TEST_METRICS_FILE,
    TEST_CONFUSION_FILE,
    TEST_PREDS_FILE
)

def main():
    """Main function to log in and upload evaluation artifacts."""
    if not HF_TOKEN:
        print("Hugging Face token not found. Set HF_TOKEN in your .env file.")
        return

    if not HF_DATASET_REPO_ID:
        print("Hugging Face dataset repo ID is not configured. HF_DATASET_REPO_ID in .env")
        return

    print(f"Logging in to Hugging Face Hub...")
    login(token=HF_TOKEN)

    api = HfApi()

    # Define the files to upload
    files_to_upload = {
        TEST_METRICS_FILE: "test_metrics_bert.csv",
        TEST_CONFUSION_FILE: "test_confusion_bert.csv",
        TEST_PREDS_FILE: "test_preds.npy"
    }

    print(f"\nStarting upload of evaluation artifacts to '{HF_DATASET_REPO_ID}'...")
    for local_path, repo_path in files_to_upload.items():
        if os.path.exists(local_path):
            print(f"  -> Uploading {repo_path}...")
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=repo_path,
                repo_id=HF_DATASET_REPO_ID,
                repo_type="dataset",
            )
        else:
            print(f"  -> Skipping {repo_path} (not found locally at {local_path}).")

    print(f"\n Successfully uploaded evaluation artifacts.")


if __name__ == "__main__":
    main()

    