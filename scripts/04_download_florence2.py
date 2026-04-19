# scripts/04_download_florence2.py
from huggingface_hub import snapshot_download

print("Downloading Florence-2-base ...")
snapshot_download(
    repo_id="microsoft/Florence-2-base",
    local_dir="/Data1/cse_24203016/construction_site/models/florence2-base"
)
print("Done.")