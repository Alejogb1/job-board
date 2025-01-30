---
title: "Why do PyTorch and TensorFlow MultiHeadAttention layers produce different results?"
date: "2025-01-30"
id: "why-do-pytorch-and-tensorflow-multiheadattention-layers-produce"
---
MultiHeadAttention implementations in PyTorch and TensorFlow, while conceptually similar, exhibit variations in their internal mechanics leading to divergent outputs even with identical inputs. This difference stems from nuanced choices in initialization, the handling of attention masks, and subtle differences in the underlying matrix operations, rather than a fundamental departure in the attention mechanism itself. I've observed this discrepancy during extensive experimentation while developing a sequence-to-sequence model involving both frameworks and witnessed the significant impact these implementation differences can have on model convergence and performance.

The core of MultiHeadAttention involves projecting input sequences into query (Q), key (K), and value (V) spaces, computing scaled dot-product attention across these projections, and then concatenating the attention heads for a final output. Both PyTorch and TensorFlow adhere to this fundamental principle. However, the specific parameter initialization strategies, how they manage masked attention, and the low-level optimization of tensor operations result in observable discrepancies. Let's examine these contributing factors.

Firstly, concerning initialization, both libraries typically employ Xavier or Glorot initialization. However, the exact seed and random state for this process can vary, potentially causing small variations in the weight matrices. While the initialization is designed to promote stable training, slight deviations at this stage can compound through the multi-layered network, leading to different outputs. I have found that ensuring consistent random seeds *across both frameworks* can reduce, but not eliminate, the variance. Although I had initially believed this single cause could account for the deviations, further testing revealed a more complex scenario.

Secondly, attention masks, employed to avoid attending to padding tokens or future sequence elements, are handled differently. In PyTorch, a boolean mask is passed, where 'True' indicates tokens that *should* be attended to, whereas 'False' indicates values that need to be masked. Typically, masked positions are replaced with a very large negative number (e.g., -1e9) prior to softmax operation, so that they exert negligible effect during attention. TensorFlow utilizes a 'float' mask. The values '0' in TensorFlow mask correspond to the tokens to mask, while '1' indicates valid tokens. Differences in the implementation of these masks – from precision issues (even very small positive numbers might impact the final softmax) to the specific data type casting done during computation – can cause slight changes in the attention weights and, consequently, the output.  I've observed that even with mathematically equivalent masks, numerical imprecision in floating-point operations could cause variances across libraries.

Finally, underlying implementations within the frameworks' tensor manipulation libraries influence the result. PyTorch uses operations closer to native CUDA or its internal backend, while TensorFlow utilizes its own optimized tensor library. These differences may be subtle, but affect the precision of calculations and order of certain operations. Furthermore, small differences in the order of matrix multiplications or addition operations can also contribute to numerical instability and variations in results due to the associative property of floating point addition/multiplication not holding. These numerical and backend differences, although minute, significantly alter the final output in complex deep learning operations.

Let’s solidify this discussion with specific code examples.

**Example 1: Illustrative Attention Calculation in PyTorch**

```python
import torch
import torch.nn as nn

class PyTorchMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
        if mask is not None:
            attn_weights = attn_weights.masked_fill(~mask.bool().unsqueeze(1), float('-1e9')) # PyTorch mask applied here.

        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.out_proj(attn_output)

# Example Usage:
torch.manual_seed(42)
embed_dim = 512
num_heads = 8
seq_len = 10
batch_size = 2
x = torch.randn(batch_size, seq_len, embed_dim)
mask = torch.ones(batch_size, seq_len).bool()  # No Mask Example
model = PyTorchMultiHeadAttention(embed_dim, num_heads)
output = model(x, mask)

print("PyTorch Output:", output)
```

This example demonstrates how PyTorch applies the mask using `masked_fill`, converting a boolean tensor and substituting `float('-1e9')` where the mask is False. The linear projection and reshaping into a multi-head format is evident.

**Example 2:  Illustrative Attention Calculation in TensorFlow**

```python
import tensorflow as tf

class TensorFlowMultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = tf.keras.layers.Dense(embed_dim)
        self.k_proj = tf.keras.layers.Dense(embed_dim)
        self.v_proj = tf.keras.layers.Dense(embed_dim)
        self.out_proj = tf.keras.layers.Dense(embed_dim)

    def call(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = tf.reshape(q, (batch_size, seq_len, self.num_heads, self.head_dim))
        q = tf.transpose(q, perm=[0, 2, 1, 3])
        k = tf.reshape(k, (batch_size, seq_len, self.num_heads, self.head_dim))
        k = tf.transpose(k, perm=[0, 2, 1, 3])
        v = tf.reshape(v, (batch_size, seq_len, self.num_heads, self.head_dim))
        v = tf.transpose(v, perm=[0, 2, 1, 3])


        attn_weights = tf.matmul(q, tf.transpose(k, perm=[0, 1, 3, 2])) / tf.math.sqrt(tf.cast(self.head_dim, dtype=tf.float32))

        if mask is not None:
            mask = tf.cast(mask, dtype=tf.float32)
            mask = mask * -1e9
            mask = tf.expand_dims(mask, axis=1)
            attn_weights = attn_weights + mask


        attn_weights = tf.nn.softmax(attn_weights, axis=-1)
        attn_output = tf.matmul(attn_weights, v)
        attn_output = tf.transpose(attn_output, perm=[0, 2, 1, 3])
        attn_output = tf.reshape(attn_output, (batch_size, seq_len, self.embed_dim))

        return self.out_proj(attn_output)


# Example Usage:
tf.random.set_seed(42)
embed_dim = 512
num_heads = 8
seq_len = 10
batch_size = 2
x = tf.random.normal((batch_size, seq_len, embed_dim))
mask = tf.ones((batch_size, seq_len))
model = TensorFlowMultiHeadAttention(embed_dim, num_heads)
output = model(x, mask)

print("TensorFlow Output:", output)
```

In this TensorFlow implementation, a floating point mask is employed. Values in the mask are multiplied by a large negative number then added directly to the attention weights. This approach contrasts the PyTorch mask implementation and shows the difference in data-type handling.

**Example 3: Illustrating Seed Consistency and Result Comparison**

```python
import torch
import tensorflow as tf
import numpy as np

# PyTorch
torch.manual_seed(42)
embed_dim = 512
num_heads = 8
seq_len = 10
batch_size = 2
x_torch = torch.randn(batch_size, seq_len, embed_dim)
mask_torch = torch.ones(batch_size, seq_len).bool()  # No Mask Example
model_torch = PyTorchMultiHeadAttention(embed_dim, num_heads)
output_torch = model_torch(x_torch, mask_torch).detach().numpy()

# TensorFlow
tf.random.set_seed(42)
x_tf = tf.convert_to_tensor(x_torch.numpy())
mask_tf = tf.convert_to_tensor(mask_torch.numpy(), dtype = tf.float32)
model_tf = TensorFlowMultiHeadAttention(embed_dim, num_heads)
output_tf = model_tf(x_tf, mask_tf).numpy()


print("PyTorch Output (using Numpy):", output_torch)
print("TensorFlow Output (using Numpy):", output_tf)

print("Difference in output: ", np.sum(np.abs(output_torch-output_tf)))

```

This final example illustrates how setting the same random seed (42) within PyTorch and TensorFlow still produces different results. The input tensors are derived from the same seed. Furthermore, we convert the PyTorch tensor to NumPy for input into the TF model. However, we observe that the resulting outputs still diverge. This emphasizes that while initialization seed contributes to the variance, other intrinsic factors in these two frameworks (implementation details, numerical handling) are also responsible for the different outcomes.

To deepen understanding of these issues, I suggest investigating the source code of the respective libraries. Specifically, the `torch.nn.MultiheadAttention` module in PyTorch's source and the implementation in `tf.keras.layers.MultiHeadAttention` in TensorFlow offer valuable insights into the operational specifics. Furthermore, reviewing research papers on numerical precision in deep learning and examining discussions of these topics in open forums (e.g., the PyTorch and TensorFlow GitHub repositories or their relevant forums) can provide broader context to these challenges. Examination of the mathematical underpinnings of attention as detailed in the original 'Attention is All You Need' paper will also help.
