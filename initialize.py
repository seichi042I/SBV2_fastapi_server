import json
from huggingface_hub import hf_hub_download
from pathlib import Path
from style_bert_vits2.logging import logger


def download_bert_models():
    with open("bert/bert_models.json", "r", encoding="utf-8") as fp:
        models = json.load(fp)
    for k, v in models.items():
        local_path = Path("bert").joinpath(k)
        for file in v["files"]:
            if not Path(local_path).joinpath(file).exists():
                logger.info(f"Downloading {k} {file}")
                hf_hub_download(v["repo_id"], file, local_dir=local_path)
                
if __name__ == "__main__":
    download_bert_models()