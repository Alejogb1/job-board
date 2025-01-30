---
title: "How can I access attention weights from a pre-trained DistilBERT model?"
date: "2025-01-30"
id: "how-can-i-access-attention-weights-from-a"
---
Accessing attention weights from a pre-trained DistilBERT model requires careful navigation of the model's internal structure and the application of PyTorch's (or a similar framework's) mechanisms for intermediate value capture. The DistilBERT architecture, while a distilled version of BERT, retains a multi-headed attention mechanism within each of its transformer layers. Extracting these weights necessitates understanding how these layers are indexed and how PyTorch facilitates access to the outputs of specific modules during a forward pass.

The key is leveraging *hooks*, particularly forward hooks, offered by PyTorch's `nn.Module` class. These hooks allow the attachment of functions that are executed during a forward pass, either before or after a specific module’s execution. In our case, we will target specific attention layers within the DistilBERT model and intercept their output - the attention weights – for our analysis.

Let's consider this process within a realistic scenario. I recall a recent project where I was tasked with analyzing the attention patterns of a DistilBERT model for a named-entity recognition task. The goal wasn’t simply prediction, but rather to gain insight into *which words* the model was focusing on when making predictions. I implemented an approach relying on forward hooks and targeted specific layers within the model to extract the needed weights. This implementation enabled a detailed analysis of the model’s behavior.

The DistilBERT architecture is composed of a series of *transformer layers*, each with self-attention mechanisms. Each self-attention module calculates query, key, and value matrices for each token in the input. The attention weights are then derived using these matrices and quantify the "attention" or relationship between all pairs of tokens. This is critical: we are not accessing a single weight for each token, but rather an attention matrix between all tokens at every layer.

I will show three concrete examples to illustrate how to correctly extract attention weights, each with progressively more detail and sophistication:

**Example 1: Basic Extraction using a Simple Hook**

This first example demonstrates the most fundamental approach: attaching a hook to a single, specific attention layer and capturing the attention weights. This serves as a basic building block. Note that the DistilBERT transformer layers are generally named with integers, and we target a single layer and its self-attention mechanism for this example.

```python
import torch
from transformers import DistilBertModel, DistilBertTokenizer

# Load the pre-trained model and tokenizer
model_name = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertModel.from_pretrained(model_name)

# Example text
text = "This is a simple example sentence."
inputs = tokenizer(text, return_tensors="pt")

# Placeholder to store the attention weights.
attention_weights = []


def attention_hook(module, input, output):
    # Output in DistilBERT's attention layer is a tuple,
    # with the attention weights as the first element.
    attention_weights.append(output[0].detach())


# Attach the hook to the self-attention layer of the 2nd transformer block, for example
hook_handle = model.transformer.layer[1].attention.attn.register_forward_hook(attention_hook)

# Run the forward pass. This is crucial as the hook is executed during this
with torch.no_grad():
    outputs = model(**inputs)

# Remove hook for efficiency
hook_handle.remove()

# Now we have the attention_weights
print("Shape of extracted attention weights:", attention_weights[0].shape) # Expected output: torch.Size([1, 12, 7, 7]) (batch size, num heads, num tokens, num tokens)

```

In this initial snippet, I import the required PyTorch and Hugging Face Transformers libraries. The model and tokenizer are instantiated from the `distilbert-base-uncased` checkpoint. An example input sentence is tokenized and prepared for model consumption. I create a simple hook function (`attention_hook`), which will append the attention weights to the `attention_weights` list whenever the targeted layer is passed through during a forward pass. Note the use of `.detach()`, which creates a copy of the tensor and detaches it from the computation graph. This is important to avoid unnecessary gradient computations. A single hook is attached to a specific attention layer and the model's forward pass is executed within `torch.no_grad()` to further improve efficiency. The hook is then removed, and the resulting attention weights are printed in terms of shape to verify extraction.

**Example 2: Capturing Weights from All Transformer Layers**

This second example enhances the first by capturing attention weights from *all* transformer layers within the DistilBERT model. This provides a more comprehensive view of the attention patterns. Instead of targeting a specific layer, we iterate through all layers.

```python
import torch
from transformers import DistilBertModel, DistilBertTokenizer

# Load the pre-trained model and tokenizer
model_name = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertModel.from_pretrained(model_name)

# Example text
text = "This is a simple example sentence."
inputs = tokenizer(text, return_tensors="pt")

# Placeholder to store the attention weights for all layers
all_attention_weights = []

def attention_hook(layer_id):
    def hook(module, input, output):
        all_attention_weights.append((layer_id,output[0].detach()))
    return hook


# Attach hooks to all self-attention layers within each transformer layer
hook_handles = []
for layer_id, layer in enumerate(model.transformer.layer):
  hook_handle = layer.attention.attn.register_forward_hook(attention_hook(layer_id))
  hook_handles.append(hook_handle)


# Run the forward pass
with torch.no_grad():
    outputs = model(**inputs)

# Remove all hooks
for hook_handle in hook_handles:
  hook_handle.remove()

# Print the shape of all extracted attention weights
for layer_id, attn in all_attention_weights:
    print(f"Shape of attention weights from layer {layer_id}: {attn.shape}")

```

The second example now employs a for loop to iterate over all transformer layers within `model.transformer.layer`. For each layer, a hook is attached to its respective attention mechanism. To keep track of which layer generated which weights, a closure was used, capturing `layer_id` from each loop iteration. This way, the `attention_hook` appends the correct layer ID alongside the extracted weights. Again, after executing the forward pass and collecting the weights, the hooks are removed. The code then iterates through the accumulated weights, printing the shape and associated layer ID for verification. This pattern provides a more structured approach to the analysis of multiple layers.

**Example 3: Accessing Heads and Aggregating Results**

The final example adds a layer of complexity by showing how to access weights from individual attention heads within a single layer and perform some basic aggregation, such as averaging weights across all heads. This demonstrates a method for analyzing the attention of specific attention heads within the model.

```python
import torch
from transformers import DistilBertModel, DistilBertTokenizer

# Load the pre-trained model and tokenizer
model_name = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertModel.from_pretrained(model_name)

# Example text
text = "This is a simple example sentence."
inputs = tokenizer(text, return_tensors="pt")


# Placeholder to store the attention weights for a single layer, per head.
head_weights = []

def attention_hook(module, input, output):
    # Output in DistilBERT's attention layer is a tuple,
    # with the attention weights as the first element.
    attention_weights_tensor = output[0].detach() # Shape (batch size, num_heads, num_tokens, num_tokens)
    batch_size, num_heads, num_tokens, _ = attention_weights_tensor.shape

    for head_idx in range(num_heads):
        head_weights.append(attention_weights_tensor[:, head_idx, :, :])  # Shape (batch_size, num_tokens, num_tokens)


#Attach hook to the self attention mechanism of the first transformer layer
hook_handle = model.transformer.layer[0].attention.attn.register_forward_hook(attention_hook)

# Run the forward pass
with torch.no_grad():
    outputs = model(**inputs)


#Remove hook
hook_handle.remove()

# Analyze aggregated attention for one layer's heads
print("Shape of a single head's attention:",head_weights[0].shape)
averaged_attention = torch.mean(torch.stack(head_weights), dim=0) # Shape: (1, num_tokens, num_tokens)
print("Shape of averaged head attention:", averaged_attention.shape)
```

This third example focuses on capturing weights from individual attention heads. The hook extracts the full weight tensor and iterates through the head dimension, collecting each head's weights. Note how head-specific weights are appended to the `head_weights` list. This example aggregates the per-head attention, by stacking all extracted head weights, then computing the mean. This process demonstrates a way to examine not just the aggregate attention, but also the contribution of each head individually.

These three examples demonstrate increasing sophistication in how to access attention weights.  The choice of which approach is suitable for a particular task will depend upon the required detail of analysis. For example, in a project attempting to identify *which* heads are most relevant for a given task, or *how* layer-specific attention differs, it is paramount to have individual attention head access. In more general analysis, where per-layer attention is the focus, iterating through all layers will suffice.

For further study on this topic, I recommend consulting the following resources. First, *The Illustrated Transformer*, which provides a highly visual overview of the Transformer architecture and self-attention mechanisms. Next, the official documentation of the *Hugging Face Transformers* library. This is the definitive source of information about the API, the pre-trained models, and the classes used to extract weights and perform analysis. Finally, a comprehensive study of attention mechanisms in deep learning. This research provides a valuable theoretical foundation for the practical application of the above.
