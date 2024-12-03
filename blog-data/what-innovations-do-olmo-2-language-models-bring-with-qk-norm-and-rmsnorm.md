---
title: "What innovations do OLMo 2 language models bring with QK-Norm and RMSNorm?"
date: "2024-12-03"
id: "what-innovations-do-olmo-2-language-models-bring-with-qk-norm-and-rmsnorm"
---

Hey so you wanna know about OLMo 2 language models QK-Norm and RMSNorm right cool stuff  I've been messing around with it lately its pretty neat actually.  Basically OLMo 2 is like this next-gen language model thing it’s all about improving on those older models like you know the ones that were kinda clunky and not super efficient.  Think of it as a serious upgrade a big step forward in how we make these things work.

And the big players here are QK-Norm and RMSNorm  they're not just some random things they are normalization techniques that totally change how the model behaves.  Normalization is basically keeping everything in a nice predictable range prevents things from exploding or vanishing completely which is a big problem in deep learning you'd probably seen that before, right?

QK-Norm specifically targets the query and key matrices in the attention mechanism.  Attention is a huge part of how these models understand context  it's like how we focus on different words in a sentence to get the meaning  QK-Norm makes this attention process much more stable and helps the model learn better representations.  Think of it as fine tuning the focus making it sharper and more accurate.  It’s pretty crucial for those transformer models so its worth paying attention to this aspect

RMSNorm is another normalization approach and its cool because it's layer-wise  it normalizes the activations of each layer independently which helps with training stability and performance.  It’s kind of like individually adjusting the volume knobs on each amplifier of your mega-awesome sound system making sure no single part gets too loud or too quiet.  It’s all about balance.

You should check out some papers on layer normalization and its variants  a good starting point would be the original paper on layer normalization itself. Then dive into papers comparing different normalization techniques in Transformers. I’m sure you could find stuff on those specific normalization techniques QK-Norm and RMSNorm by searching scholar.google.com with those keywords

Now let’s get into some code  I'm gonna use Python with PyTorch because it's what I'm most comfortable with but the concepts should be transferable to other frameworks like TensorFlow or JAX.

First let's just create a simple attention mechanism and throw some QK-Norm in there

```python
import torch
import torch.nn as nn

class QKNormAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.query_norm = nn.LayerNorm(d_model)
        self.key_norm = nn.LayerNorm(d_model)
        self.value = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value):
        query = self.query_norm(query)
        key = self.key_norm(key)
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.d_model**0.5)
        attention_weights = self.softmax(attention_scores)
        context_vector = torch.matmul(attention_weights, value)
        return context_vector

# Example usage
d_model = 512
query = torch.randn(1, 64, d_model)
key = torch.randn(1, 64, d_model)
value = torch.randn(1, 64, d_model)

attention = QKNormAttention(d_model)
output = attention(query, key, value)
print(output.shape) # Should be (1, 64, 512)

```

This code defines a simple attention mechanism with QK-Norm applied to the query and key matrices before the attention calculation.  It's a pretty basic implementation you could totally expand this to incorporate multi-head attention and stuff like that.  To grasp this code thoroughly you might need to go back to the basics of attention mechanisms and maybe look at some intro level material on transformer networks

For RMSNorm its a little different.  Let's see how we'd apply that to a feedforward network


```python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.eps = 1e-6
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return x / norm * self.scale

class FeedForward(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.rmsnorm1 = RMSNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.rmsnorm2 = RMSNorm(d_model)

    def forward(self, x):
        x = self.fc1(x)
        x = self.rmsnorm1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = self.rmsnorm2(x)
        return x

# Example usage
d_model = 512
hidden_dim = 1024
x = torch.randn(1, 64, d_model)
ffn = FeedForward(d_model, hidden_dim)
output = ffn(x)
print(output.shape)  # Should be (1, 64, 512)

```

Here I have defined  RMSNorm layer and incorporated it into a simple feedforward network.  RMSNorm normalizes the activations before the ReLU activation function and again at the output  This helps keep the activations in a reasonable range and helps in the gradient flow.  For a deeper understanding you need to focus on how RMSNorm differs from other normalization methods like BatchNorm or LayerNorm

Finally lets put it all together in a tiny transformer block


```python
import torch
import torch.nn as nn

# Assuming you have the QKNormAttention and RMSNorm defined above

class TransformerBlock(nn.Module):
    def __init__(self, d_model, hidden_dim, num_heads):
        super().__init__()
        self.attention = QKNormAttention(d_model) # using our custom QKNorm
        self.ffn = FeedForward(d_model, hidden_dim)
        self.rmsnorm_attn = RMSNorm(d_model) # Applying RMSNorm after attention
        self.rmsnorm_ffn = RMSNorm(d_model) # Applying RMSNorm after FFN
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        attn_output = self.attention(x, x, x) # self-attention
        x = self.rmsnorm_attn(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.rmsnorm_ffn(x + self.dropout(ffn_output))
        return x

# Example Usage
d_model = 512
hidden_dim = 1024
num_heads = 8
x = torch.randn(1, 64, d_model)
transformer_block = TransformerBlock(d_model, hidden_dim, num_heads)
output = transformer_block(x)
print(output.shape) # Should be (1, 64, 512)

```

This combines the previous examples showing how you might use QK-Norm and RMSNorm together in a transformer block.  Its a pretty stripped-down version but it shows the core ideas.  To truly get the hang of it check out some papers on deep transformer models.  Many good books and tutorials on this online.


So that's the gist of it OLMo 2 QK-Norm and RMSNorm.  It's all about improving efficiency and stability in large language models using smart normalization techniques. Remember those papers and books I suggested they'll be your best friends on this journey.  Let me know if you have any other questions  happy coding
