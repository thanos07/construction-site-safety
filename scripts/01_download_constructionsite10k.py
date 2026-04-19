from huggingface_hub import snapshot_download
import os

BASE = "/Data1/cse_24203016/construction_site/data/constructionsite10k"
os.makedirs(BASE, exist_ok=True)

print("Downloading ConstructionSite10k from LouisChen15/ConstructionSite ...")
snapshot_download(
    repo_id="LouisChen15/ConstructionSite",   # CORRECT link you provided
    repo_type="dataset",
    local_dir=BASE
)
print(f"Done. Check: {BASE}")