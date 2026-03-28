import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


MODEL_NAME = "meta-llama/Llama-3-8B"
OUTPUT_DIM  = 4096


class LlamaEncoderBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            output_hidden_states=True
        )
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
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,
                             output_hidden_states=True)
        return outputs.hidden_states[-1]

    @torch.no_grad()
    def forward_pooled(self, input_ids: torch.Tensor,
                       attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.forward_sequence(input_ids, attention_mask)
        return self._mean_pool(hidden, attention_mask)
