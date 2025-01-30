---
title: "How can I use an RNN model trained with PyTorch without PyTorch?"
date: "2025-01-30"
id: "how-can-i-use-an-rnn-model-trained"
---
The core challenge in deploying a PyTorch-trained RNN without PyTorch lies in translating the model's computational graph and learned parameters into a format executable by a different inference engine. PyTorch's dynamic computational graph, while powerful during training, doesn't directly lend itself to static execution environments. Therefore, the crucial step involves exporting the trained RNN model into an interoperable format, such as ONNX (Open Neural Network Exchange), and using an ONNX runtime to perform inference.

I've encountered this scenario multiple times in my work, notably when deploying models to embedded systems with limited computational resources where Python or PyTorch were not viable options. Often, the requirement isn't just inference; the goal is often high performance without the overhead of a complete deep learning framework. Exporting to ONNX becomes the bridge, enabling us to leverage the model outside of the training environment.

Let's break down the process into distinct stages: model preparation in PyTorch, model export to ONNX, and finally, inference using an ONNX runtime.

**Model Preparation in PyTorch:**

First, you need a trained RNN model in PyTorch. This model can be a standard `nn.RNN`, `nn.LSTM`, or `nn.GRU`, or any custom variation you create using PyTorch's building blocks. The important thing is that it has been trained and its parameters (weights and biases) are fixed. You'll need to load the model's saved state dictionary if the model was trained and then saved. I will simulate training and saving the model as a sample demonstration:

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple RNN model
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out[:, -1, :]) # take last timestep
        return out, hidden

# Model parameters
input_size = 10
hidden_size = 20
output_size = 5
model = SimpleRNN(input_size, hidden_size, output_size)

# Dummy data and training loop simulation
dummy_input = torch.randn(2, 5, input_size) # batch_size=2, seq_len=5
dummy_target = torch.randint(0, output_size, (2,))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Simulate training by taking one training step
for i in range(10):
  hidden = torch.zeros(1, 2, hidden_size) # initial hidden state
  optimizer.zero_grad()
  output, hidden = model(dummy_input, hidden)
  loss = criterion(output, dummy_target)
  loss.backward()
  optimizer.step()

# Save the model's state dictionary
torch.save(model.state_dict(), 'simple_rnn.pth')
print("Simulated trained model saved as simple_rnn.pth")
```

This code establishes a basic RNN model, simulates training on sample data and saves the learned parameters to 'simple_rnn.pth'. I've included the simulated training phase to emphasize that the state dictionary is saved for later exporting and that training is no longer required when performing inference.

**Model Export to ONNX:**

With a trained model (or loaded its state dictionary) in hand, the next step is exporting to the ONNX format. This involves creating a dummy input that mimics the expected data format for inference. It is essential that you create an input tensor based on expected values, especially for the sequence dimension if applicable, along with defining dynamic axes if there is a batch size dimension. The following code shows how to perform the export:

```python
import torch
import torch.nn as nn
import torch.onnx

# Define the same RNN model
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out[:, -1, :])
        return out, hidden

# Model parameters (match with the trained model)
input_size = 10
hidden_size = 20
output_size = 5
model = SimpleRNN(input_size, hidden_size, output_size)

# Load the trained state dictionary
model.load_state_dict(torch.load('simple_rnn.pth'))

# Set model to evaluation mode
model.eval()

# Create dummy input and initial hidden state
dummy_input = torch.randn(1, 5, input_size) # batch_size=1, seq_len=5
dummy_hidden = torch.zeros(1, 1, hidden_size)

# Export the model to ONNX
torch.onnx.export(
    model,
    (dummy_input, dummy_hidden),
    'simple_rnn.onnx',
    opset_version=13,
    input_names=['input', 'hidden'],
    output_names=['output', 'next_hidden'],
    dynamic_axes={'input': {0: 'batch_size'}, 'hidden': {1: 'batch_size'}, 'output': {0: 'batch_size'}, 'next_hidden': {1: 'batch_size'}}
)
print("Model exported to simple_rnn.onnx")
```

This code loads the model's state dictionary, creates a dummy input, and utilizes `torch.onnx.export` to generate the ONNX file.  Pay close attention to setting the `dynamic_axes` argument; this tells ONNX that the batch size dimension can be dynamic when performing inference, a crucial step for flexibility.  The `opset_version` defines the set of ONNX operations. It is recommended to pick the latest that is well-supported by your target runtime, often opset version 13 or later.  You will want to use the input, output, and dynamic axes arguments to make sure your model can be inferred using an ONNX runtime with ease.

**Inference with ONNX Runtime:**

Once the model is exported to ONNX, we can use the ONNX Runtime (or another compatible runtime) to execute the model. Here is an example of how to perform inference using the ONNX Runtime:

```python
import onnxruntime
import numpy as np

# Load the ONNX model
ort_session = onnxruntime.InferenceSession('simple_rnn.onnx')

# Create a sample input
input_data = np.random.randn(1, 5, 10).astype(np.float32)
hidden_data = np.zeros((1, 1, 20)).astype(np.float32)

# Run inference
ort_inputs = {'input': input_data, 'hidden': hidden_data}
ort_outputs = ort_session.run(['output', 'next_hidden'], ort_inputs)

# Retrieve the output and the hidden state
output, next_hidden = ort_outputs

print("Output shape:", output.shape)
print("Next hidden state shape:", next_hidden.shape)
print("Output:", output)
```

This code loads the exported ONNX model using `onnxruntime.InferenceSession`. It creates a new dummy input of similar shape and type to the dummy input used for the export but now with random values.  It provides the input to the model and retrieves the predicted output and next hidden state, which can be used in a recurrent manner. This last code segment demonstrates that with proper configuration you can execute your PyTorch model without the presence of the PyTorch framework, making deployment in constrained environments achievable.

For further exploration of ONNX, I would recommend reviewing the official ONNX documentation for understanding its specific operators and format. For practical implementation, you might also explore the ONNX Runtime documentation to understand the specific APIs and configurations.  Additionally, consider consulting literature on model optimization and quantization, techniques that can significantly improve the inference performance of models once they are exported to a framework independent format like ONNX.  You may also want to review the documentation for other ONNX inference runtime environments like Tensorflow Lite.
