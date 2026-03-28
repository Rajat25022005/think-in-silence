import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


MODEL_NAME = "BAAI/bge-large-en-v1.5"
OUTPUT_DIM  = 1024


class BGELargeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.model     = AutoModel.from_pretrained(MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.output_dim = OUTPUT_DIM

        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    @staticmethod
    def _mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        expanded = mask.unsqueeze(-1).float()
        summed   = (hidden * expanded).sum(dim=1)
        counts   = expanded.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    @torch.no_grad()
    def forward_sequence(self, input_ids: torch.Tensor,
                         attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state

    @torch.no_grad()
    def forward_pooled(self, input_ids: torch.Tensor,
                       attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.forward_sequence(input_ids, attention_mask)
        return self._mean_pool(hidden, attention_mask)
