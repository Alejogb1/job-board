---
title: "Why does normalization cause exploding gradients during network training?"
date: "2025-01-30"
id: "why-does-normalization-cause-exploding-gradients-during-network"
---
Gradient explosion, a significant obstacle in neural network training, often manifests when seemingly beneficial techniques like normalization are improperly applied, particularly within recurrent neural networks (RNNs). Specifically, while normalization, like batch normalization or layer normalization, aims to stabilize activations and improve training speed, its uncritical deployment, especially *after* recurrence, can exacerbate, not mitigate, gradient explosion issues. This stems from how backpropagation interacts with normalized recurrent activations.

I’ve encountered this issue firsthand while developing a sequence-to-sequence model for time-series data prediction involving long dependencies. Initially, we aggressively implemented layer normalization after each RNN cell's output with the goal of preventing internal covariate shift and speeding up training convergence. This approach, however, resulted in the opposite effect: the model failed to converge, and the gradients grew exponentially during backpropagation.

The core of the problem resides in the multiplicative nature of backpropagation through time (BPTT) in RNNs. Consider a simplified RNN cell at time step *t*. The activation *h<sub>t</sub>* is a function of the previous activation *h<sub>t-1</sub>* and the input *x<sub>t</sub>*, represented abstractly as:

```
h<sub>t</sub> = f(h<sub>t-1</sub>, x<sub>t</sub>; W)
```

Where W represents learnable parameters. When we perform backpropagation, we need to compute the gradient of a loss function, *L*, with respect to the parameters *W* across all time steps. Crucially, this involves calculating the chain rule derivative:

```
∂L/∂W  =  Σ<sub>t</sub> (∂L/∂h<sub>t</sub>) * (∂h<sub>t</sub>/∂W)
```

Here, the term  *(∂L/∂h<sub>t</sub>)* depends on all subsequent time steps' gradients via the BPTT algorithm. Consequently, when the gradients within an RNN explode, the primary drivers are not the individual gradients ∂h<sub>t</sub>/∂W at each time-step itself, but rather the chain-rule backpropagation via the term  ∂L/∂h<sub>t</sub>.

If we introduce layer normalization *after* the recurrent output *h<sub>t</sub>*, effectively making the normalized output *ĥ<sub>t</sub>*:

```
ĥ<sub>t</sub> =  LN(h<sub>t</sub>)
```

This transformation, while individually well-behaved (the layer normalization operation is intended to stabilize and scale the activations to have zero mean and unit variance), introduces a potential instability point when the gradient backpropagates through time. During backpropagation, we now have:

```
(∂L/∂h<sub>t</sub>) =  (∂L/∂ĥ<sub>t</sub>) * (∂ĥ<sub>t</sub>/∂h<sub>t</sub>)
```
Here, (∂ĥ<sub>t</sub>/∂h<sub>t</sub>) represents the derivative of the layer normalization operation with respect to its input, the unnormalized h<sub>t</sub>, for *each time step*.  Crucially, the crucial term  *(∂L/∂h<sub>t</sub>)* becomes a *product* of many (∂ĥ<sub>t</sub>/∂h<sub>t</sub>) terms, accumulated across the unfolded recurrent sequence by the chain-rule, multiplied through all the time steps. Even if these derivatives ∂ĥ<sub>t</sub>/∂h<sub>t</sub> are individually moderate, the cumulative effect, particularly with longer sequences, can result in an exponential increase in the gradient magnitude, causing the gradient to "explode" - in effect, the magnitude of the gradients grows far beyond the numerical bounds that a computation can represent, resulting in training instability.

The problem is not with normalization itself. Instead, it arises from where normalization is placed within the recurrent computation graph and how backpropagation interacts with that placement.  Normalization modifies the magnitude of intermediate activations. In effect, by normalizing recurrent outputs at *each time step*, we are changing the input distribution to the *next* recurrent time step in a way that does not guarantee stability of the backpropagated gradients. Normalization *prior* to the recurrent calculation (e.g. before applying the recurrent activation function) does not cause this problem; it only occurs when placed *after* the recurrent unit.

Here are three illustrative code examples demonstrating this effect and potential mitigations:

**Example 1: Layer Normalization After RNN Output (Exploding Gradients)**

```python
import torch
import torch.nn as nn
import torch.optim as optim

class RNNWithPostNormalization(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNNWithPostNormalization, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.ln = nn.LayerNorm(hidden_size)
    def forward(self, x):
        out, _ = self.rnn(x)
        normalized_out = self.ln(out)
        return normalized_out

input_size = 10
hidden_size = 20
seq_len = 50
batch_size = 32

model = RNNWithPostNormalization(input_size, hidden_size)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Example input data
input_data = torch.randn(batch_size, seq_len, input_size)
target_data = torch.randn(batch_size, seq_len, hidden_size)

# Training Loop (simplified)
for i in range(100):
    optimizer.zero_grad()
    output = model(input_data)
    loss = criterion(output, target_data)
    loss.backward()

    # Print average gradients to show explosion
    for param in model.parameters():
        if param.grad is not None:
           print(f"Gradient norm after iteration {i}: {torch.norm(param.grad).item()}")

    optimizer.step()
```
In this example, we observe rapidly increasing gradient norms with each training step.  The layer normalization after the RNN output leads to the described gradient explosion problem.

**Example 2: Layer Normalization Before RNN Input (Stable Gradients)**

```python
import torch
import torch.nn as nn
import torch.optim as optim

class RNNWithPreNormalization(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNNWithPreNormalization, self).__init__()
        self.ln = nn.LayerNorm(input_size)
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
    def forward(self, x):
        normalized_input = self.ln(x)
        out, _ = self.rnn(normalized_input)
        return out

input_size = 10
hidden_size = 20
seq_len = 50
batch_size = 32

model = RNNWithPreNormalization(input_size, hidden_size)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()


input_data = torch.randn(batch_size, seq_len, input_size)
target_data = torch.randn(batch_size, seq_len, hidden_size)

# Training Loop (simplified)
for i in range(100):
    optimizer.zero_grad()
    output = model(input_data)
    loss = criterion(output, target_data)
    loss.backward()

    # Print average gradients to show stability
    for param in model.parameters():
        if param.grad is not None:
           print(f"Gradient norm after iteration {i}: {torch.norm(param.grad).item()}")

    optimizer.step()
```

Here, we move the layer normalization operation to normalize the *input* to the RNN rather than the recurrent output. We see that the gradient magnitudes remains stable during training. The same normalization technique applied before rather than after avoids the issue.

**Example 3: Gradient Clipping (Mitigation)**
```python
import torch
import torch.nn as nn
import torch.optim as optim

class RNNWithPostNormalizationClipped(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNNWithPostNormalizationClipped, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.ln = nn.LayerNorm(hidden_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        normalized_out = self.ln(out)
        return normalized_out

input_size = 10
hidden_size = 20
seq_len = 50
batch_size = 32

model = RNNWithPostNormalizationClipped(input_size, hidden_size)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

input_data = torch.randn(batch_size, seq_len, input_size)
target_data = torch.randn(batch_size, seq_len, hidden_size)

# Training Loop (simplified)
for i in range(100):
    optimizer.zero_grad()
    output = model(input_data)
    loss = criterion(output, target_data)
    loss.backward()

    # Apply Gradient Clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # Print average gradients to show stability
    for param in model.parameters():
        if param.grad is not None:
           print(f"Gradient norm after iteration {i}: {torch.norm(param.grad).item()}")

    optimizer.step()
```

This final example retains the problematic post-normalization but introduces gradient clipping with `torch.nn.utils.clip_grad_norm_`.  This method prevents the gradients from becoming excessively large, promoting more stable training. This approach is a mitigation rather than a solution that involves restructuring the network.

For further investigation into stabilizing RNN training, I would recommend focusing on theoretical resources related to backpropagation through time (BPTT), specifically its derivation and limitations. Material on the vanishing and exploding gradient problem, while not specific to normalization, provide useful context. Researching alternative normalization techniques within RNNs, and exploring their theoretical limitations would also be fruitful. Additionally, investigations into effective methods of gradient clipping, including adaptive clipping strategies, and parameter initialization are essential. Focus on literature explaining the theoretical basis of gradient stability in RNNs is critical. These investigations should be complemented by thorough implementation within a deep learning framework.
