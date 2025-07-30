#push_model.py
import os
from huggingface_hub import HfApi, login
from dotenv import load_dotenv

# Load variables from the .env file
load_dotenv()

# Hugging Face token will be loaded from the environment variable & available because of the load_dotenv() call
hf_token = os.environ.get("HF_TOKEN")

# Check if the token was found
if not hf_token:
    print("There's an error, hugging face token not present found check your .env file for the token.")
    exit()

# Log in to Hugging Face using the token
login(token =hf_token)
api = HfApi()


# Hugging face credentials
hf_username = "farookahmedalishaik"
hf_repo_id = f"{hf_username}/intent-bert"

# Create repository on Hugging Face Hub using via HTTP but not Git clone
api.create_repo(repo_id = hf_repo_id, private = False, exist_ok = True) #private =False means it's a public model

local_model_dir = "artifacts/bert_intent_model"
local_metrics_file = "artifacts/test_metrics_bert.csv"
local_confusion_file = "artifacts/test_confusion_bert.csv"
local_label_map_file = "artifacts/label_mapping.csv"


# Check if the local model directory exists
if not os.path.isdir(local_model_dir):
    print(f"Local model directory '{local_model_dir}' not found")
    exit()

print(f"uploading model files from '{local_model_dir}' to '{hf_repo_id}' on Hugging Face Hub")
api.upload_folder(
    folder_path = local_model_dir,
    repo_id = hf_repo_id,
    repo_type = "model",
    commit_message ="Add fine-tuned BERT intent model",
)
print(f"Model files are pushed successfully to {hf_repo_id}")

print("Uploading additional artifacts like (metrics, confusion matrix, label map)")

# Upload test_metrics_bert.csv
if os.path.exists(local_metrics_file):
    api.upload_file(
        path_or_fileobj = local_metrics_file,
        path_in_repo = "test_metrics_bert.csv", # Name it directly in the root of the HF repo
        repo_id = hf_repo_id,
        repo_type = "model",
        commit_message = "Add test metrics for BERT model",
    )
    print(f"Uploaded {local_metrics_file}")
else:
    print(f"{local_metrics_file} not found (upload skip)")

# Upload test_confusion_bert.csv
if os.path.exists(local_confusion_file):
    api.upload_file(
        path_or_fileobj = local_confusion_file,
        path_in_repo = "test_confusion_bert.csv",
        repo_id =hf_repo_id,
        repo_type = "model",
        commit_message = "Add confusion matrix for BERT model",
    )
    print(f"Uploaded {local_confusion_file}")
else:
    print(f"{local_confusion_file} not found (upload skip)")

# Upload label_mapping.csv
if os.path.exists(local_label_map_file):
    api.upload_file(
        path_or_fileobj = local_label_map_file,
        path_in_repo = "label_mapping.csv",
        repo_id = hf_repo_id,
        repo_type = "model",
        commit_message = "Add label mapping for BERT model",
    )
    print(f"Uploaded {local_label_map_file}")
else:
    print(f"{local_label_map_file} not found (upload skip.")

print(f"All files are pushed successfully to {hf_repo_id}")