---
title: "How does quantization make AI models more efficient?"
date: '2024-11-14'
id: 'how-does-quantization-make-ai-models-more-efficient'
---

So quantization is basically like squeezing a big model into a smaller space. Imagine your model as a giant picture, and you want to make it fit on your phone screen. You have to reduce the number of colors, or "bits" in the picture to make it smaller. That's what quantization does.

Here's a snippet to give you an idea. This code defines a simple quantizer:

```python
def quantize(x, num_bits):
  """Quantizes the input tensor x to num_bits."""
  min_val = x.min()
  max_val = x.max()
  range_val = max_val - min_val
  quantized_x = (x - min_val) / range_val * (2**num_bits - 1)
  quantized_x = quantized_x.round().astype(np.int)
  return quantized_x
```

The search term is "quantization in machine learning" and it's a powerful technique to make models smaller and faster. It's especially helpful when you want to run your models on devices with limited resources like mobile phones or embedded systems.
