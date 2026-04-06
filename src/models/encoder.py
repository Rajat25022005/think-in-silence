import torch
import torch.nn as nn


class FrozenEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        backbone_name = cfg.model.backbone
        proj_dim      = cfg.model.proj_dim

        if backbone_name == "bge_large":
            from src.models.backbones.bge_large import BGELargeBackbone
            self.backbone = BGELargeBackbone()
        elif backbone_name == "distilbert":
            from src.models.backbones.distilbert import DistilBERTBackbone
            self.backbone = DistilBERTBackbone()
        elif backbone_name == "llama":
            from src.models.backbones.llama_encoder import LlamaEncoderBackbone
            self.backbone = LlamaEncoderBackbone()
        elif backbone_name == "gemma3":
            from src.models.backbones.gemma3 import Gemma3Backbone
            self.backbone = Gemma3Backbone()
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")

        backbone_dim = self.backbone.output_dim

        self.question_proj = nn.Sequential(
            nn.Linear(backbone_dim, proj_dim),
            nn.LayerNorm(proj_dim)
        )
        self.answer_proj = nn.Sequential(
            nn.Linear(backbone_dim, proj_dim),
            nn.LayerNorm(proj_dim)
        )

    def encode_question(self, input_ids: torch.Tensor,
                        attention_mask: torch.Tensor) -> torch.Tensor:
        seq = self.backbone.forward_sequence(input_ids, attention_mask)
        return self.question_proj(seq.to(self.question_proj[0].weight.dtype))

    def encode_answer(self, input_ids: torch.Tensor,
                      attention_mask: torch.Tensor) -> torch.Tensor:
        pooled = self.backbone.forward_pooled(input_ids, attention_mask)
        return self.answer_proj(pooled.to(self.answer_proj[0].weight.dtype))

    @property
    def tokenizer(self):
        return self.backbone.tokenizer
