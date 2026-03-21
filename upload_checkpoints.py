"""Upload all checkpoints from a run to Hugging Face."""
import os
import argparse
from huggingface_hub import HfApi

parser = argparse.ArgumentParser()
parser.add_argument("--repo-id", type=str, required=True, help="e.g. migub/grpo-multigame-self-only")
parser.add_argument("--output-dir", type=str, required=True, help="e.g. ./output/grpo_multigame_self_only")
args = parser.parse_args()

api = HfApi()
for name in sorted(os.listdir(args.output_dir)):
    path = os.path.join(args.output_dir, name)
    if os.path.isdir(path) and (name.startswith("checkpoint-") or name == "final"):
        print(f"Uploading {name}...")
        api.upload_folder(
            repo_id=args.repo_id,
            folder_path=path,
            path_in_repo=name,
            repo_type="model",
        )
        print(f"  Done: {name}")

print("All checkpoints uploaded!")
