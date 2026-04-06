import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = 'google/gemma-3-4b-it'
OUTPUT_DIM  = 2560   # Gemma-3-4B hidden dim

class Gemma3Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=torch.bfloat16,
            device_map='auto',
            output_hidden_states=True
        )
        self.output_dim = OUTPUT_DIM
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    def train(self, mode=True):
        # Backbone is always frozen — never allow training mode
        return super().train(False)

    @staticmethod
    def _last_token_pool(hidden, mask):
        # Last non-padding token — correct for causal LLMs
        last_idx = mask.sum(dim=1) - 1
        return hidden[torch.arange(hidden.size(0)), last_idx]

    @torch.no_grad()
    def forward_sequence(self, input_ids, attention_mask):
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        # Use second-to-last layer — more stable than final
        return out.hidden_states[-2]

    @torch.no_grad()
    def forward_pooled(self, input_ids, attention_mask):
        hidden = self.forward_sequence(input_ids, attention_mask)
        return self._last_token_pool(hidden, attention_mask)
