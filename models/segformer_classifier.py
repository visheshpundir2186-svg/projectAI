import torch
import torch.nn as nn

class SegFormerClassifier(nn.Module):
    """
    SegFormer (Mix Transformer - MiT) adapted with a Classification Head.
    Captures non-local self-attention across multi-scale satellite image patches.
    """
    def __init__(self, num_classes=3, pretrained_model="nvidia/mit-b0"):
        super().__init__()
        from transformers import SegformerForImageClassification, SegformerConfig
        
        try:
            self.model = SegformerForImageClassification.from_pretrained(
                pretrained_model,
                num_labels=num_classes,
                ignore_mismatched_sizes=True
            )
        except Exception:
            # Fallback initialization from config if offline
            config = SegformerConfig.from_pretrained(pretrained_model)
            config.num_labels = num_classes
            self.model = SegformerForImageClassification(config)

    def forward(self, x):
        # HuggingFace SegFormer expects pixel_values
        outputs = self.model(pixel_values=x)
        return outputs.logits
