from huggingface_hub import HfApi
import os

api = HfApi()

# IMPORTANT: Replace 'your-username' with your actual Hugging Face username
hf_username = "farookahmedalishaik" # <--- YOUR ACTUAL USERNAME
hf_repo_id = f"{hf_username}/intent-bert"

# Create the repository on Hugging Face Hub if it doesn't exist
# This will be done via HTTP, not Git clone
api.create_repo(repo_id=hf_repo_id, private=False, exist_ok=True)
# Setting private=False means it will be a public model, which is usually fine.
# Change to private=True if you want to keep it private (requires Pro plan or paid features for private space on some tiers)

local_model_dir = "artifacts/bert_intent_model"
local_metrics_file = "artifacts/test_metrics_bert.csv"
local_confusion_file = "artifacts/test_confusion_bert.csv"
local_label_map_file = "artifacts/label_mapping.csv"


# Check if the local model directory exists
if not os.path.isdir(local_model_dir):
    print(f"Error: Local model directory '{local_model_dir}' not found. Please ensure your model files are there.")
    exit()

print(f"Uploading model files from '{local_model_dir}' to '{hf_repo_id}' on Hugging Face Hub...")
api.upload_folder(
    folder_path=local_model_dir,
    repo_id=hf_repo_id,
    repo_type="model",
    commit_message="Add fine-tuned BERT intent model",
)
print(f"Model files pushed successfully to {hf_repo_id}")

# --- NEW: Upload additional artifacts ---
print("Uploading additional artifacts (metrics, confusion matrix, label map)...")

# Upload test_metrics_bert.csv
if os.path.exists(local_metrics_file):
    api.upload_file(
        path_or_fileobj=local_metrics_file,
        path_in_repo="test_metrics_bert.csv", # Name it directly in the root of the HF repo
        repo_id=hf_repo_id,
        repo_type="model",
        commit_message="Add test metrics for BERT model",
    )
    print(f"Uploaded {local_metrics_file}")
else:
    print(f"Warning: {local_metrics_file} not found. Skipping upload.")

# Upload test_confusion_bert.csv
if os.path.exists(local_confusion_file):
    api.upload_file(
        path_or_fileobj=local_confusion_file,
        path_in_repo="test_confusion_bert.csv",
        repo_id=hf_repo_id,
        repo_type="model",
        commit_message="Add confusion matrix for BERT model",
    )
    print(f"Uploaded {local_confusion_file}")
else:
    print(f"Warning: {local_confusion_file} not found. Skipping upload.")

# Upload label_mapping.csv
if os.path.exists(local_label_map_file):
    api.upload_file(
        path_or_fileobj=local_label_map_file,
        path_in_repo="label_mapping.csv",
        repo_id=hf_repo_id,
        repo_type="model",
        commit_message="Add label mapping for BERT model",
    )
    print(f"Uploaded {local_label_map_file}")
else:
    print(f"Warning: {local_label_map_file} not found. Skipping upload.")

print(f"All specified files pushed successfully to {hf_repo_id}")