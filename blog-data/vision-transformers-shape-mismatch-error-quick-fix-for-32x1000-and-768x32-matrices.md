---
title: "Vision Transformers: Shape Mismatch Error - Quick Fix for 32x1000 and 768x32 Matrices"
date: '2024-11-08'
id: 'vision-transformers-shape-mismatch-error-quick-fix-for-32x1000-and-768x32-matrices'
---

```python
class RegressionViT(nn.Module):
    def __init__(self, in_features=224 * 224 * 3, num_classes=1, pretrained=True):
        super(RegressionViT, self).__init__()
        self.vit_b_16 = vit_b_16(pretrained=pretrained)
        # Accessing the actual output feature size from vit_b_16
        heads = self.vit_b_16.heads
        heads.head = nn.Linear(heads.head.in_features, num_classes) 

    def forward(self, x):
        x = self.vit_b_16(x)
        return x

# Model
model = RegressionViT(num_classes=1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.MSELoss()  # Use appropriate loss function for regression
optimizer = optim.Adam(model.parameters(), lr=0.0001)
```
