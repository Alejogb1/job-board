---
title: "How many parameters are in BERT's masked language model?"
date: "2025-01-30"
id: "how-many-parameters-are-in-berts-masked-language"
---
The precise number of parameters in BERT's Masked Language Model (MLM) component isn't a single, universally fixed value. It’s inherently tied to the specific BERT model variant being used, primarily the distinction between BERT-Base and BERT-Large, and influenced by implementation details. However, understanding how these parameters are allocated within the architecture provides a strong foundation to grasp the order of magnitude involved.

The BERT architecture, pre-trained through the MLM and next sentence prediction objectives, is built around multiple transformer encoder layers. These layers are the heart of the model, containing feed-forward networks and multi-head self-attention mechanisms, each contributing significantly to the parameter count. The MLM itself doesn't represent a distinct layer; rather, it’s a training objective applied to the final encoded representation. We use a classification head, typically a linear layer followed by an activation (usually softmax) over the vocabulary size for each token, allowing prediction of the masked words. This is the structure to consider while analyzing parameters.

Specifically, the feed-forward network in each transformer block has two linear transformations. The self-attention mechanism also incorporates multiple linear layers to derive the query, key, and value matrices, and a final linear projection after the scaled dot-product attention is computed. Parameters are also introduced by the embeddings (token, position, and segment) used in the input processing phase. Thus the total number of trainable parameters is a sum from each layer and component. While it is rare to only use the MLM, it's important to understand the parameter impact of the overall model during this process.

In my experience fine-tuning various BERT models for natural language understanding tasks, I've observed that the majority of the parameters are concentrated in the transformer encoder layers. The MLM head, being a final linear classification layer, adds a smaller but crucial portion of the overall parameters. The size of the vocabulary directly impacts the size of this output layer; with typical wordpiece tokenization around 30,000 tokens, this layer alone becomes quite sizable. This is especially true in BERT-Large, which has increased hidden size, number of attention heads, and number of transformer layers, all of which contribute a large number to the total parameters.

To understand how the parameters are calculated at a higher level, let's break down key components with pseudo-code examples. Note this will be a simplified representation and real implementations can have more minor variations.

**Code Example 1: Parameter Count in a Single Transformer Layer**

```python
def calculate_transformer_layer_parameters(hidden_size, num_attention_heads):
    """
    Calculates the parameters in a single transformer layer.
    Assumes a standard feed-forward network with two linear projections.
    """
    # Linear layers for query, key, and value in self-attention.
    attention_param = 3 * hidden_size * hidden_size
    # Projection layer after the multi-head attention calculation.
    attention_output_param = hidden_size * hidden_size
    # Feed-forward network with a hidden size matching the hidden layer size.
    ffn_param = 2 * hidden_size * hidden_size # 2 linear layers with hidden_size each
    # Bias terms for the linear layers.
    bias_param = 4 * hidden_size # 4 linear layers with hidden_size bias each
    # Parameters for the multi-head self-attention mechanism.
    num_attention_param = (
        attention_param + attention_output_param + bias_param
    ) 
    # Calculate total parameters in this layer
    total_params = num_attention_param + ffn_param
    return total_params

hidden_size = 768  # Example from BERT-Base
num_attention_heads = 12 # Example from BERT-Base
params_per_layer = calculate_transformer_layer_parameters(hidden_size, num_attention_heads)
print(f"Parameters in a single transformer layer: {params_per_layer}")
```

This example illustrates how the `hidden_size` and `num_attention_heads` interact to determine the parameters in a single transformer layer. The linear transformations within self-attention and the feed-forward network contribute the most to the total number of parameters. In BERT-Base there are 12 of these layers and in BERT-Large there are 24.

**Code Example 2: Parameter Count in the Output Layer**

```python
def calculate_output_layer_parameters(hidden_size, vocab_size):
    """
    Calculates the parameters in the output layer (MLM head).
    """
    # Linear transformation from the last hidden state to the vocabulary size.
    output_param = hidden_size * vocab_size
    # Bias term for output transformation
    bias_param = vocab_size
    total_params = output_param + bias_param
    return total_params

hidden_size = 768 # Example from BERT-Base
vocab_size = 30522 # Example vocabulary size
params_output_layer = calculate_output_layer_parameters(hidden_size, vocab_size)
print(f"Parameters in the output layer: {params_output_layer}")
```

Here, the number of parameters in the output layer is dominated by the product of the `hidden_size` and the `vocab_size`. This illustrates that the vocabulary size has a major impact on the number of parameters in the MLM classification head. A larger vocabulary requires a significantly larger output layer.

**Code Example 3: Estimating Total Parameters**

```python
def estimate_total_bert_parameters(hidden_size, num_attention_heads, num_layers, vocab_size):
  """
  Estimates total BERT parameters.
  """
  params_per_layer = calculate_transformer_layer_parameters(hidden_size, num_attention_heads)
  params_output_layer = calculate_output_layer_parameters(hidden_size, vocab_size)
  total_parameters_layers = params_per_layer * num_layers
  total_parameters = total_parameters_layers + params_output_layer
  # Adding Embedding layer parameters
  embedding_parameters = 3 * hidden_size * vocab_size # 3 embeddings - token, position, segment
  total_parameters += embedding_parameters
  return total_parameters
    

hidden_size_base = 768
num_attention_heads_base = 12
num_layers_base = 12 # number of transformer layers
vocab_size_base = 30522

hidden_size_large = 1024
num_attention_heads_large = 16
num_layers_large = 24 # number of transformer layers
vocab_size_large = 30522

total_params_base = estimate_total_bert_parameters(hidden_size_base, num_attention_heads_base, num_layers_base, vocab_size_base)
total_params_large = estimate_total_bert_parameters(hidden_size_large, num_attention_heads_large, num_layers_large, vocab_size_large)


print(f"Estimated parameters for BERT-Base: {total_params_base / 10**6:.2f} million")
print(f"Estimated parameters for BERT-Large: {total_params_large / 10**6:.2f} million")
```

This function provides a rough estimate of the total parameters. The token, positional, and segment embedding parameters are added, which significantly impact the overall parameter count. The core of the calculation is based on parameters from the transformer layers and from the final classification head. The output highlights the massive difference in parameters between BERT-Base and BERT-Large, stemming from the increased hidden sizes and layers. Note that this approach is still an approximation, and the true count can vary, even within the same model designation, by subtle implementation choices.

It's crucial to recognize that the provided examples are simplified. Actual implementations might contain additional parameters due to normalization layers, attention dropout, and other minor design considerations. Additionally, model parallelism, parameter sharing, and other optimization techniques might influence the actual count.

For further understanding, I would recommend reviewing research papers detailing the BERT architecture, paying particular attention to the exact dimensions of the linear transformations and the layers within the transformer blocks. The original BERT paper should be a primary reference. Also, exploring documentation on pre-trained model repositories is valuable, since such repositories provide parameter counts and model configurations directly. Finally, examining open-source implementations of BERT using frameworks like TensorFlow or PyTorch is exceptionally beneficial to see how these models are instantiated in real-world usage. These resources, when combined, will solidify understanding of the magnitude of the parameter count in a BERT masked language model and how they are distributed across the layers of the model.
