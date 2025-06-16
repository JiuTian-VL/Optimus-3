import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from lightning.pytorch.loggers import WandbLogger  # noqa
from sentence_transformers import SentenceTransformer


class TaskRouterModel(nn.Module, PyTorchModelHubMixin):
    def __init__(self):
        super().__init__()

        self.bert = SentenceTransformer("/data1/Models/sentence-bert-base")
        self.head = nn.Sequential(nn.Linear(768, 768 * 4), nn.ReLU(), nn.Linear(768 * 4, 5))

    def forward(self, x):
        embed = self.bert.encode(x, convert_to_tensor=True, device=self.bert.device, show_progress_bar=False)
        return self.head(embed)

    def router(self, query: str) -> int:
        logits = self.forward(query)
        return logits.argmax(dim=-1).item()
