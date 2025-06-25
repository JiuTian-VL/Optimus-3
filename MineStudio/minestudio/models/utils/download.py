'''
Date: 2024-12-14 01:46:36
LastEditors: muzhancun muzhancun@126.com
LastEditTime: 2024-12-14 02:00:17
FilePath: /MineStudio/minestudio/models/utils/download.py
'''
import huggingface_hub
import os
import pathlib

def download_model(model_name: str, local_dir: str = "downloads") -> str:

    assert model_name in ["ROCKET-1", "VPT", "GROOT", "STEVE-1"], f"Unknown model: {model_name}"

    local_dir = os.path.join(pathlib.Path(__file__).parent.parent.absolute(), local_dir, model_name)
    
    download = False
    response = input(f"Detecting missing {model_name}, do you want to download it from huggingface (Y/N)?\n")
    if not os.path.exists(local_dir):
        while True:
            if response == 'Y' or response == 'y':
                download = True
                break
            elif response == 'N' or response == 'n':
                break
            else:
                response = input("Please input Y or N:\n")

    if not download:
        return None

    print(f"Downloading {model_name} to {local_dir}")

    huggingface_hub.hf_hub_download(repo_id=f'CraftJarvis/{model_name}', filename='.', local_dir=local_dir, repo_type='model')
    
    return local_dir