---
title: "How can Fastspeech2 be exported to ONNX with dynamic shapes?"
date: "2025-01-30"
id: "how-can-fastspeech2-be-exported-to-onnx-with"
---
Exporting FastSpeech2 models to ONNX with dynamic shapes requires careful consideration of how tensor dimensions are handled during both the PyTorch model definition and the ONNX conversion process. Specifically, issues often arise from the variance in input sequence lengths, which, if not correctly specified, lead to rigid graph structures within the ONNX model, hindering adaptability to different sequence lengths at runtime.

My past project involved deploying a custom text-to-speech system, and I encountered this precise problem with our FastSpeech2 implementation. The initial exports resulted in ONNX graphs that were only compatible with input sequences exactly the size used during tracing, rendering them practically unusable in a dynamic production environment. I resolved this by explicitly defining dynamic axes within the ONNX export process.

The core issue is that during PyTorch's tracing or scripting for ONNX export, it often infers the tensor shapes at export time based on the example inputs provided. If these inputs have specific, fixed dimensions (e.g., an input sequence of length 100), the resulting ONNX graph will be constrained to those dimensions. To enable dynamic shapes, one must instruct the ONNX exporter about which axes within the input and output tensors can vary.

Firstly, one needs to define `input_names`, `output_names`, and `dynamic_axes` parameters in the `torch.onnx.export` function. These parameters will inform the exporter which tensors and dimensions should be allowed to vary dynamically. `input_names` and `output_names` are lists of strings specifying the names of each input and output tensor as interpreted by the graph. `dynamic_axes` is a dictionary mapping each named tensor to a list of axis indices allowed to vary.

Consider the typical FastSpeech2 inference setup: we input text tokens (`input_tokens`), a length tensor (`input_lengths`), and potentially some speaker embedding data. The core output is the mel-spectrogram. The `input_tokens` and `input_lengths` vary significantly depending on the input text, making them excellent candidates for dynamic shapes.

Let's examine three code examples that illustrate the evolution from a naive, statically shaped export to a fully dynamic export.

**Example 1: Naive Static Export (Incorrect)**

```python
import torch
import torch.nn as nn
from torch.onnx import export

class MockFastSpeech2(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_mel_bins):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.Linear(embed_dim, hidden_size)
        self.decoder = nn.Linear(hidden_size, num_mel_bins)

    def forward(self, input_tokens, input_lengths):
        x = self.embedding(input_tokens)
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Example Input
vocab_size = 100
embed_dim = 128
hidden_size = 256
num_mel_bins = 80
max_sequence_length = 128 # Assumption for tracing

model = MockFastSpeech2(vocab_size, embed_dim, hidden_size, num_mel_bins)
input_tokens_example = torch.randint(0, vocab_size, (1, max_sequence_length), dtype=torch.long)
input_lengths_example = torch.tensor([max_sequence_length], dtype=torch.long)

# Attempt to export without specifying dynamic axes
try:
  export(model,
          (input_tokens_example, input_lengths_example),
          'static_fastspeech2.onnx',
          opset_version=13,
          verbose=False
          )
  print("Static export successful")
except Exception as e:
  print(f"Static export failed: {e}")
```
This code attempts to export the model without dynamic axis specification, resulting in an ONNX graph fixed to the `max_sequence_length` of 128. This limitation becomes apparent during inference when encountering sequences of different length. The export may or may not succeed, but the resulting ONNX model is practically useless for dynamic inputs.

**Example 2: Partially Dynamic Export (Improved, but incomplete)**

```python
import torch
import torch.nn as nn
from torch.onnx import export

class MockFastSpeech2(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_mel_bins):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.Linear(embed_dim, hidden_size)
        self.decoder = nn.Linear(hidden_size, num_mel_bins)

    def forward(self, input_tokens, input_lengths):
      x = self.embedding(input_tokens)
      x = self.encoder(x)
      x = self.decoder(x)
      return x

# Example Input
vocab_size = 100
embed_dim = 128
hidden_size = 256
num_mel_bins = 80
max_sequence_length = 128

model = MockFastSpeech2(vocab_size, embed_dim, hidden_size, num_mel_bins)
input_tokens_example = torch.randint(0, vocab_size, (1, max_sequence_length), dtype=torch.long)
input_lengths_example = torch.tensor([max_sequence_length], dtype=torch.long)

# Partially dynamic export - only input tokens have dynamic shapes
input_names = ['input_tokens', 'input_lengths']
output_names = ['output_mel']
dynamic_axes = {'input_tokens': {1: 'sequence_length'}}

try:
  export(model,
          (input_tokens_example, input_lengths_example),
          'partial_dynamic_fastspeech2.onnx',
          opset_version=13,
          input_names=input_names,
          output_names=output_names,
          dynamic_axes=dynamic_axes,
          verbose=False
          )
  print("Partially dynamic export successful")
except Exception as e:
  print(f"Partially dynamic export failed: {e}")
```

This example improves upon the first by using the `dynamic_axes` parameter. We now specify that the second dimension (axis index 1) of the `input_tokens` tensor can vary, using the variable name `'sequence_length'`. This makes the ONNX graph compatible with variable length sequences for `input_tokens`. However, input\_lengths, while variable during inference, is still fixed at export, which might cause some issues if not handled properly by the inference system.

**Example 3: Fully Dynamic Export (Correct)**

```python
import torch
import torch.nn as nn
from torch.onnx import export

class MockFastSpeech2(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_mel_bins):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.Linear(embed_dim, hidden_size)
        self.decoder = nn.Linear(hidden_size, num_mel_bins)

    def forward(self, input_tokens, input_lengths):
      x = self.embedding(input_tokens)
      x = self.encoder(x)
      x = self.decoder(x)
      return x

# Example Input
vocab_size = 100
embed_dim = 128
hidden_size = 256
num_mel_bins = 80
max_sequence_length = 128

model = MockFastSpeech2(vocab_size, embed_dim, hidden_size, num_mel_bins)
input_tokens_example = torch.randint(0, vocab_size, (1, max_sequence_length), dtype=torch.long)
input_lengths_example = torch.tensor([max_sequence_length], dtype=torch.long)

# Fully dynamic export - both input_tokens and input_lengths have dynamic shapes
input_names = ['input_tokens', 'input_lengths']
output_names = ['output_mel']
dynamic_axes = {'input_tokens': {1: 'sequence_length'},
                'input_lengths':{0: 'batch_size'}}

try:
  export(model,
          (input_tokens_example, input_lengths_example),
          'dynamic_fastspeech2.onnx',
          opset_version=13,
          input_names=input_names,
          output_names=output_names,
          dynamic_axes=dynamic_axes,
          verbose=False
          )
  print("Fully dynamic export successful")
except Exception as e:
  print(f"Fully dynamic export failed: {e}")
```

In this final example, we declare both the sequence length of `input_tokens` and the batch size of `input_lengths` as dynamic axes.  `input_tokens` has its second dimension (axis index 1, as sequence length) set to `'sequence_length'` while `input_lengths` has its first dimension (axis index 0, representing batch size) set to `'batch_size'`. This makes the exported ONNX model fully flexible in terms of batch and sequence lengths.  The ONNX runtime environment can then handle arbitrary sequence lengths and batches during inference.

To thoroughly understand and successfully implement ONNX exports with dynamic shapes for FastSpeech2 or other sequence models, I highly recommend reviewing the official ONNX documentation, paying specific attention to the sections regarding dynamic axes and operator support.  Further examination of the PyTorch export documentation will clarify the precise interaction of `torch.onnx.export` with these dynamic settings. In practical implementations, exploring community forums dedicated to ONNX and PyTorch can provide a wealth of insight on how users have overcome similar hurdles. Finally, carefully verifying the exported ONNX model's functionality using tools such as `onnxruntime` is critical to ensuring it is behaving as intended.
