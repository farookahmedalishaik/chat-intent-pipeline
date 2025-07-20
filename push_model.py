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

# Check if the local model directory exists
if not os.path.isdir(local_model_dir):
    print(f"Error: Local model directory '{local_model_dir}' not found. Please ensure your model files are there.")
    exit()

# Use upload_folder to push the contents of your local directory directly
print(f"Uploading files from '{local_model_dir}' to '{hf_repo_id}' on Hugging Face Hub...")
api.upload_folder(
    folder_path=local_model_dir,
    repo_id=hf_repo_id,
    repo_type="model", # Specify that it's a model repository
    commit_message="Add fine-tuned BERT intent model",
    # Optionally, you can set delete_patterns to remove files on the Hub
    # that are no longer present locally if this is an update.
    # delete_patterns=["*"] # Use with caution: deletes all remote files not in folder_path
)

print(f"Model pushed successfully to {hf_repo_id}")