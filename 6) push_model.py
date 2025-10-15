# 6) push_model.py

import os
from huggingface_hub import HfApi, login

# Import required settings from the central config file
from config import (
    HF_TOKEN,
    HF_USER,
    HF_REPO_ID,
    MODEL_DIR_PATH,
    CLASS_THRESHOLDS_FILE,
    VAL_METRICS_FILE,
    VAL_CONFUSION_FILE
)

def main():
    if not HF_TOKEN:
        print("Hugging Face token missing in environment. Set HF_TOKEN in .env")
        return

    if not HF_USER:
        print("Hugging Face username missing in environment. Set HF_USER in .env")
        return

    login(token=HF_TOKEN)
    api = HfApi()

    api.create_repo(repo_id=HF_REPO_ID, private=False, exist_ok=True)

    if not os.path.isdir(MODEL_DIR_PATH):
        print(f"Local model directory '{MODEL_DIR_PATH}' not found")
        return

    print(f"Uploading model files from '{MODEL_DIR_PATH}' to '{HF_REPO_ID}' on Hugging Face Hub")
    api.upload_folder(
        folder_path=MODEL_DIR_PATH,
        repo_id=HF_REPO_ID,
        repo_type="model",
        commit_message="Add fine-tuned BERT intent model",
    )
    print(f"Model files are pushed successfully to {HF_REPO_ID}")

    def upload_if_exists(path, dest_name, msg):
        if os.path.exists(path):
            api.upload_file(
                path_or_fileobj=path,
                path_in_repo=dest_name,
                repo_id=HF_REPO_ID,
                repo_type="model",
                commit_message=msg,
            )
            print(f"Uploaded {path}")
        else:
            print(f"{path} not found (upload skip)")

    # Upload evaluation results that are specific to this model version.

    upload_if_exists(CLASS_THRESHOLDS_FILE, "class_thresholds.json", "Add per-class confidence thresholds")
    upload_if_exists(VAL_METRICS_FILE, "val_classification_report.csv", "Add validation metrics")
    upload_if_exists(VAL_CONFUSION_FILE, "val_confusion_matrix.csv", "Add validation confusion matrix")


    print(f"All available files pushed to {HF_REPO_ID}")


if __name__ == "__main__":
    main()
