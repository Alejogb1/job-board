---
title: "ViT Models in PyTorch: Quick Tip to Disable Fusing Attention"
date: '2024-11-08'
id: 'vit-models-in-pytorch-quick-tip-to-disable-fusing-attention'
---

```python
import torch
import torchvision

# Load the ViT_B_16 model
retrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT
pretrained_vit_1 = torchvision.models.vit_b_16(weights=retrained_vit_weights).to(device)

# Disable fused attention in the ViT_B_16 model
for block in pretrained_vit_1.blocks:
    block.attn.fused_attn = False
```
