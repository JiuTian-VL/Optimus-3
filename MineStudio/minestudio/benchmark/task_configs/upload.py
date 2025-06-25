"""
Date: 2025-01-06 20:08:00
LastEditors: muzhancun muzhancun@stu.pku.edu.cn
LastEditTime: 2025-02-25 14:00:49
FilePath: /MineStudio/minestudio/benchmark/task_configs/upload.py
"""

from huggingface_hub import HfApi

if __name__ == "__main__":
    local_dir = "./hard"
    repo_id = "CraftJarvis/MineStudio_task_group.hard"
    api = HfApi()
    # api.create_repo(repo_id=repo_id, repo_type='dataset')
    # if not api.repo_exists(repo_id=repo_id, repo_type='dataset'):
    #     api.create_repo(repo_id=repo_id, repo_type='dataset')
    api.upload_folder(folder_path=local_dir, repo_id=repo_id, repo_type="dataset")
