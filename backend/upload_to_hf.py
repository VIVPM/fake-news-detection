"""One-time script to upload existing artifacts to Hugging Face Hub."""

import os
from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

HF_TOKEN = os.getenv("HF_TOKEN")
HF_REPO_ID = os.getenv("HF_REPO_ID")
ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")

api = HfApi(token=HF_TOKEN)

# Create repo if it doesn't exist
api.create_repo(repo_id=HF_REPO_ID, exist_ok=True)

# Upload files
for filename in ["lstm_model.onnx", "tokenizer.json"]:
    filepath = os.path.join(ARTIFACTS_DIR, filename)
    if os.path.exists(filepath):
        print(f"Uploading {filename}...")
        api.upload_file(
            path_or_fileobj=filepath,
            path_in_repo=filename,
            repo_id=HF_REPO_ID,
        )
        print(f"  [OK] {filename} uploaded")
    else:
        print(f"  [SKIP] {filename} not found")

print(f"\nDone! Files available at https://huggingface.co/{HF_REPO_ID}")
