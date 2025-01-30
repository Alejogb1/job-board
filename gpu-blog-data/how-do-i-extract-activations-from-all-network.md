---
title: "How do I extract activations from all network units across all layers and timesteps?"
date: "2025-01-30"
id: "how-do-i-extract-activations-from-all-network"
---
Extracting activations across all network units, layers, and timesteps, particularly in sequence-based models, requires a nuanced approach that combines model introspection, tensor manipulation, and a clear understanding of the underlying computational graph. During my time developing real-time anomaly detection systems for time-series sensor data, this was a frequent requirement for debugging and gaining insights into model behavior. The challenge isn't merely about *accessing* the activations; it's about doing so efficiently without disrupting the forward pass and handling the potential for high-dimensionality data.

Fundamentally, accessing activations involves hooking into the model's forward pass at the appropriate locations—after each layer’s computation but before the subsequent layer receives its input.  Modern deep learning frameworks such as TensorFlow or PyTorch provide mechanisms to do this. The core idea is to use these mechanisms to create "hooks" that intercept the output of specific layers during the forward propagation. This can be accomplished using callback functions that capture the relevant tensors. However, the specific implementation varies significantly between static and dynamic models.  In sequential models, such as Recurrent Neural Networks (RNNs) and their variants like LSTMs and GRUs, the presence of timesteps adds another dimension to the retrieval process. Activations are not only specific to the layer but also the particular time instance within the sequence.

Let’s explore the process using PyTorch as an example, which is the framework with which I’m most experienced. It provides a flexible mechanism through the use of `torch.nn.Module`’s built-in hook functionalities. A crucial understanding here is the nature of the data being passed through a recurrent layer. Consider, for example, an LSTM. At each timestep, it takes in an input and a hidden state, produces an output (which could be passed to the next layer or used for final classification/regression), and produces a new hidden state that’s passed to the next timestep. To capture all activations, we need to intercept *both* the layer’s output and hidden states.

Here's the first practical example showing activation capture in a simple feedforward network.

```python
import torch
import torch.nn as nn
from collections import defaultdict

class SimpleFeedForward(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleFeedForward, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def extract_activations(model, input_tensor):
    activations = defaultdict(list)
    def hook_fn(name):
        def hook(module, input, output):
            activations[name].append(output.detach()) # detach is key
        return hook

    for name, module in model.named_modules():
      if isinstance(module, nn.Linear) or isinstance(module, nn.ReLU):
        module.register_forward_hook(hook_fn(name))

    model(input_tensor)
    return activations

# Example usage:
model = SimpleFeedForward(input_size=10, hidden_size=20, output_size=5)
input_data = torch.randn(1, 10) # Batch of 1, 10 features
all_activations = extract_activations(model, input_data)
for layer_name, acts in all_activations.items():
  print(f"Layer {layer_name} has activations with shapes: { [a.shape for a in acts] }")
```

This code defines a basic feedforward network and the `extract_activations` function. The `hook_fn` is a closure that captures the layer's name and registers a hook that appends the output tensor to the dictionary. The `detach()` operation is essential; it prevents further computation from tracking the history associated with this tensor, which would otherwise create unnecessary computational overhead and memory consumption.

In the second example, let’s focus on extracting activations from a vanilla recurrent neural network. The primary alteration here is the need to iterate over timesteps.

```python
import torch
import torch.nn as nn
from collections import defaultdict

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output, _ = self.rnn(x)  # output: batch, seq_len, hidden_size
        out_last_timestep = output[:, -1, :] # output at last timestep
        x = self.fc(out_last_timestep)
        return x

def extract_rnn_activations(model, input_tensor):
    activations = defaultdict(list)

    def hook_fn(name):
        def hook(module, input, output):
            activations[name].append(output.detach())
        return hook
    
    # registering hook before the rnn layer
    for name, module in model.named_modules():
        if isinstance(module, nn.RNN):
            module.register_forward_hook(hook_fn(name))

    # explicitly saving hidden states as they are not outputs of the model's forward call
    hidden_states = []
    def hidden_state_hook(module, input, output):
        hidden_states.append(output[1].detach())

    # register hook to retrieve hidden state
    model.rnn.register_forward_hook(hidden_state_hook)

    model(input_tensor)
    activations['hidden_states'] = hidden_states

    return activations

# Example usage:
model = SimpleRNN(input_size=5, hidden_size=10, output_size=2)
input_data = torch.randn(1, 10, 5) # Batch of 1, 10 timesteps, 5 features
all_activations = extract_rnn_activations(model, input_data)
for layer_name, acts in all_activations.items():
    if 'hidden' in layer_name:
        print(f"Layer {layer_name} has activation shape: { [a.shape for a in acts] } ")
    else:
        print(f"Layer {layer_name} has activation shape: { [a.shape for a in acts] }")
```

In this example, we've made some critical changes. First, notice we now provide a sequence as an input: batch size 1, sequence length 10, and 5 features at each time instance. Critically, we now also collect hidden states.  The RNN layer returns *both* the output for all timesteps, *and* the final hidden state. As we are interested in per timestep activations, we need to capture the hidden state output as well, which we do using an additional hook, `hidden_state_hook`. Note that both `activations` and `hidden_states` will hold tensors for every timestep, as we capture them in the hooks within the forward function of the model, every time the forward is called. This will grow larger depending on sequence length. This is essential to get a complete picture of a sequence model’s computation.

Finally, let's extend this to demonstrate capturing both outputs and hidden states across all layers and timesteps using an LSTM. This also demonstrates how to use `named_children` to get to submodules, instead of having to enumerate all of them ourselves using `named_modules`.

```python
import torch
import torch.nn as nn
from collections import defaultdict

class MultiLayerLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(MultiLayerLSTM, self).__init__()
        self.lstms = nn.ModuleList([nn.LSTM(input_size if i == 0 else hidden_size, hidden_size, batch_first=True) for i in range(num_layers)])
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output = x
        for lstm in self.lstms:
            output, _ = lstm(output)
        out_last_timestep = output[:, -1, :]
        x = self.fc(out_last_timestep)
        return x


def extract_lstm_activations(model, input_tensor):
    activations = defaultdict(list)
    hidden_states = defaultdict(list)

    def hook_fn(name):
        def hook(module, input, output):
            activations[name].append(output[0].detach()) # output at all timesteps
            # explicitly save each hidden state as it is not available via output of model
            hidden_states[name].append(output[1].detach())
        return hook

    for name, module in model.named_children(): # children modules, not all submodules, good for top-level layers
        if isinstance(module, nn.ModuleList):
            for i, sub_module in enumerate(module.children()): # children of the ModuleList
              name_layer = f"{name}_{i}"
              sub_module.register_forward_hook(hook_fn(name_layer)) # append to activation dictionary
        elif isinstance(module, nn.Linear):
           module.register_forward_hook(hook_fn(name)) # register a hook

    model(input_tensor)
    return activations, hidden_states

# Example usage:
model = MultiLayerLSTM(input_size=5, hidden_size=10, num_layers=2, output_size=2)
input_data = torch.randn(1, 10, 5) # Batch of 1, 10 timesteps, 5 features
all_activations, all_hidden_states = extract_lstm_activations(model, input_data)
for layer_name, acts in all_activations.items():
    print(f"Layer {layer_name} activations shapes: {[a.shape for a in acts]}")
for layer_name, hidden in all_hidden_states.items():
    print(f"Layer {layer_name} hidden shapes: {[a.shape for a in hidden]}")

```

Here, we have constructed a multi-layer LSTM. We are now using `named_children()` to get top level components. We are also traversing each LSTM in the ModuleList `lstms` and creating unique identifiers for each. Critically, we are now storing both activations and hidden states, for all layers, across all timesteps.  The result is that `activations` will contain, for each layer in the network, a list of tensors which contain the output of each layer after the forward pass, and similarly `hidden_states` will store a list of tuples for each layer.

In conclusion, capturing activations requires a deliberate approach. Understanding the mechanics of forward propagation, the structure of the model, and the specifics of the deep learning framework is essential. The strategies presented here, while using PyTorch as a reference, are conceptually applicable to other frameworks like TensorFlow by making use of appropriate callbacks, and by utilizing the provided functions to hook into the model’s computational graph. For deeper understanding of these principles, exploring introductory resources on deep learning and specific deep learning framework documentation on forward hooks and callback mechanisms is strongly advised. Additional research into tensor manipulation techniques is essential for efficient post-processing of these activations.
