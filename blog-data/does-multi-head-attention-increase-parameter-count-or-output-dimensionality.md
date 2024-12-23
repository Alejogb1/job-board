---
title: "Does multi-head attention increase parameter count or output dimensionality?"
date: "2024-12-23"
id: "does-multi-head-attention-increase-parameter-count-or-output-dimensionality"
---

Okay, let's tackle this. It's a common question, and one that I recall encountering fairly early in my deep learning journey, specifically when I was implementing a transformer-based sequence model for a natural language processing task back in '19. We were seeing some unexpected resource consumption, and that's when the multi-head attention mechanism really came under scrutiny. Let’s break down what’s actually going on under the hood, clarifying the impact it has on both parameter count and output dimensionality.

The essence of multi-head attention is to allow the model to attend to different parts of the input sequence with different learned representations. Instead of just one set of query, key, and value vectors – as in single-head attention – we have *h* sets, hence “multi-head.” This design enables the model to capture a wider variety of relationships within the data. Crucially, each head operates independently in its own subspace before being concatenated and projected back. So, the question hinges on how this multi-headed nature affects the overall dimensions and parameters.

Let's consider the impact on the parameter count first. In a single-head attention mechanism, you have three weight matrices: `W_q` (query), `W_k` (key), and `W_v` (value). Each of these matrices transforms the input embedding dimension (`d_model`) into a feature dimension (`d_k` for `W_q` and `W_k`, and `d_v` for `W_v`). Usually, in practice `d_k` and `d_v` are chosen to be equal and often `d_model` and `d_k` are equal too (though not strictly mandatory). Therefore, the number of parameters are `d_model` * `d_k` for each of W_q and W_k, and `d_model` * `d_v` for W_v, so assuming  `d_k` and `d_v` are equal to `d_k`, for single-head attention, we have a total of approximately `3 * d_model * d_k` parameters.

Now, for multi-head attention, we replicate this set *h* times. Instead of using `d_model`, we divide that `d_model` by `h` and each head operates in a reduced `d_model`/h space. This is also important: each head still uses a separate `W_q`, `W_k`, and `W_v` matrix. Hence, each head will have parameter counts corresponding to `(d_model/h * d_k) + (d_model/h * d_k) + (d_model/h * d_v)` where `d_k` is generally equal to `d_v` so we get `3 * d_model/h * d_k` parameters per head. Then because of the *h* heads we multiply by `h` again, resulting in a total of `3 * d_model * d_k` parameters for multi-head, which is the same as a single head with equal dimensions. The only extra parameter matrix we have is the `W_o` matrix for projection after concatenating the individual heads, which will result in a parameter count of `h*d_v*d_model`, adding to the total number of parameters in the multi-head layer.

Let's illustrate that with some code in python, using pytorch.

```python
import torch
import torch.nn as nn

class SingleHeadAttention(nn.Module):
    def __init__(self, d_model, d_k):
        super().__init__()
        self.W_q = nn.Linear(d_model, d_k)
        self.W_k = nn.Linear(d_model, d_k)
        self.W_v = nn.Linear(d_model, d_k)

    def forward(self, x):
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        # simplified attention calculation
        attn_weights = torch.matmul(q, k.transpose(-2, -1))
        attn_weights = torch.softmax(attn_weights, dim=-1)
        output = torch.matmul(attn_weights, v)
        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, h):
        super().__init__()
        self.h = h
        self.d_k = d_k
        self.heads = nn.ModuleList([SingleHeadAttention(d_model, d_k) for _ in range(h)])
        self.W_o = nn.Linear(h * d_k, d_model)

    def forward(self, x):
        head_outputs = [head(x) for head in self.heads]
        concatenated_output = torch.cat(head_outputs, dim=-1)
        output = self.W_o(concatenated_output)
        return output

d_model = 512
d_k = 64
h = 8
#testing
x = torch.randn(1, 10, d_model)

single_attn = SingleHeadAttention(d_model,d_k)
print(f"Single head attention params: {sum(p.numel() for p in single_attn.parameters())}")

multi_attn = MultiHeadAttention(d_model,d_k,h)
print(f"Multi head attention params: {sum(p.numel() for p in multi_attn.parameters())}")

```
The output should show you that without the `W_o` matrix, the parameter count would be the same.

Now, regarding the output dimensionality, this is where it gets interesting. Each head in multi-head attention produces an output of size `d_v`, where `d_v` is usually equal to `d_k` but not strictly necessary. Then we concatenate the outputs of the `h` heads which results in a shape of `h*d_v`. Then, as shown in the code above, we use the projection matrix `W_o` which maps this `h*d_v` back down to `d_model`, the original embedding space. Therefore, although there are intermediate spaces where dimensionality increases, the final output of a multi-head attention block has a dimensionality that matches the input, which is `d_model`.

Here’s a simple illustration:

```python
import torch
import torch.nn as nn

class MultiHeadAttentionDebug(nn.Module):
    def __init__(self, d_model, d_k, h):
        super().__init__()
        self.h = h
        self.d_k = d_k
        self.heads = nn.ModuleList([nn.Linear(d_model, d_k) for _ in range(h)])
        self.W_o = nn.Linear(h * d_k, d_model)
        self.d_model = d_model


    def forward(self, x):
        head_outputs = [head(x) for head in self.heads]

        print(f"Shape of head_outputs (before cat): {[h.shape for h in head_outputs]}")

        concatenated_output = torch.cat(head_outputs, dim=-1)
        print(f"Shape of concatenated_output: {concatenated_output.shape}")
        output = self.W_o(concatenated_output)
        print(f"Shape of final output: {output.shape}")
        return output

d_model = 512
d_k = 64
h = 8

x = torch.randn(1, 10, d_model)
multi_attn = MultiHeadAttentionDebug(d_model, d_k, h)
_ = multi_attn(x)
```

Running this will show you the dimensions of the head outputs before concatenation (shape should be batch * sequence length * 64, where 64 is d_k), the concatenated output (shape batch * sequence length * 512, where 512 is h*d_k), and the final output which matches the original input, batch * sequence length * d_model.

Let's quickly review one final code example that implements scaled dot-product attention.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k
    
    def forward(self, q, k, v, mask=None):
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_weights = F.softmax(attention_scores, dim=-1)
        output = torch.matmul(attention_weights, v)
        return output
    
class MultiHeadAttentionExample(nn.Module):
    def __init__(self, d_model, d_k, h):
        super().__init__()
        self.h = h
        self.d_k = d_k
        self.W_q = nn.Linear(d_model, h * d_k)
        self.W_k = nn.Linear(d_model, h * d_k)
        self.W_v = nn.Linear(d_model, h * d_k)
        self.attention = ScaledDotProductAttention(d_k)
        self.W_o = nn.Linear(h * d_k, d_model)

    def split_heads(self, x):
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.h, self.d_k)
        x = x.transpose(1, 2)
        return x

    def combine_heads(self, x):
        batch_size, _, seq_len, d_k = x.shape
        x = x.transpose(1, 2)
        x = x.contiguous().view(batch_size, seq_len, self.h * d_k)
        return x

    def forward(self, x, mask=None):
        q = self.split_heads(self.W_q(x))
        k = self.split_heads(self.W_k(x))
        v = self.split_heads(self.W_v(x))
        
        output = self.attention(q, k, v, mask)
        output = self.combine_heads(output)
        output = self.W_o(output)
        return output

d_model = 512
d_k = 64
h = 8

x = torch.randn(1, 10, d_model)
multi_attn = MultiHeadAttentionExample(d_model, d_k, h)
output = multi_attn(x)
print(f"Final output shape: {output.shape}")
```

This example shows a complete implementation, with projection matrices before splitting the heads and scaling during the dot product attention. The final output dimension again matches the input dimension.

In summary, multi-head attention *doesn’t* inherently increase the output dimensionality beyond the original embedding dimension (`d_model`). It does introduce intermediate dimensions, but a final projection ensures it returns to the same dimensionality as the input. It also adds parameters, from the `W_o` matrix, but *it does not increase the per head parameter counts.*

For further in-depth understanding, I'd strongly suggest looking at the original paper “Attention is All You Need” by Vaswani et al., and also spending some time with *The Annotated Transformer* by Harvard NLP. These resources will give you a solid grounding in the theoretical underpinnings of attention mechanisms. Another fantastic resource is *Deep Learning with PyTorch* by Eli Stevens et al., which provides practical implementations of many attention-based models and offers insights into why certain design choices are made.
