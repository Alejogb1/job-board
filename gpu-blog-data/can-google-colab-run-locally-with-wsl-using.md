---
title: "Can Google Colab run locally with WSL using a local GPU?"
date: "2025-01-30"
id: "can-google-colab-run-locally-with-wsl-using"
---
The specific challenge of leveraging a local GPU with Google Colab through Windows Subsystem for Linux (WSL) arises because Colab is fundamentally a cloud-based service. Directly accessing hardware resources outside of Google’s infrastructure isn't a standard feature. However, a bridge can be engineered, though it diverges significantly from the typical Colab experience. I've spent a considerable amount of time exploring these workarounds, and while they offer some local compute functionality, they come with crucial trade-offs regarding environment consistency and ease of use.

The core issue lies in Colab’s execution model: code runs in a virtualized environment hosted on Google's servers. Even when a Colab notebook specifies GPU usage, it’s referring to a GPU within Google’s data centers. To use a local GPU, one must essentially bypass the cloud execution and direct the notebook's operations to a locally running Python environment that can access the physical hardware. This process necessitates setting up a local Jupyter server, which Colab can then connect to, making Colab act primarily as an interactive interface rather than the primary execution engine.

This isn't a simple toggle. It requires a nuanced understanding of how Colab communicates with its backend and the tooling required to mimic that interaction locally. The chosen method fundamentally alters the user experience; local environment configuration becomes paramount, and reliance on Google’s pre-built environment diminishes. The approach I've found to be most effective involves using the `jupyter notebook` server and the `jupyter_http_over_ws` extension, which enables communication between a Colab notebook and a local kernel over a web socket. This approach requires a consistent local development environment that is compatible with dependencies used within the Colab notebook.

First, one must ensure the presence of a suitable NVIDIA driver for the local GPU within WSL. It is crucial to use a recent Windows operating system version, ideally Windows 11, as it offers a mature WSL implementation capable of full GPU passthrough. Once the driver is set up and functioning correctly, the local Python environment within WSL should be configured. I generally advise using `conda` or `venv` to manage Python dependencies.

Specifically, the key steps are:

1. Install and configure a suitable GPU driver in Windows, ensuring the driver is visible in WSL2.
2. Create and activate a virtual environment within WSL2 with `conda` or `venv`.
3. Install Jupyter within the virtual environment: `pip install notebook jupyter_http_over_ws`.
4. Install any other necessary packages required by your Colab notebook (e.g. `torch`, `tensorflow`, `numpy`, etc.).
5. Start the local notebook server with the websocket extension enabled: `jupyter notebook --allow-origin=* --NotebookApp.allow_origin_whitelist="http://localhost:8080" --NotebookApp.iopub_data_rate_limit=100000000 --NotebookApp.tornado_settings="{'websocket_max_message_size': 1000000000}" --NotebookApp.disable_check_xsrf=True --NotebookApp.token=""`.
6. In Colab, connect to the local runtime using the ‘Connect to a local runtime’ option, which will prompt you to copy/paste a URL.

Note, the command-line arguments for the Jupyter server above are designed to prevent common errors and connection issues. The parameters `allow-origin=*`, `allow_origin_whitelist`, `disable_check_xsrf` and `token=""` are crucial for establishing a stable connection. The websocket configuration ensures larger data transfers do not cause disconnections. In my own experience, failure to properly configure the websocket often leads to runtime errors when transferring large datasets or complex computations.

Here are three practical code examples, each with specific purpose and rationale:

**Example 1: Basic GPU Verification**

This demonstrates that the local environment is correctly configured and is indeed using the intended GPU. This example uses PyTorch but can be adapted to Tensorflow as needed.

```python
import torch

if torch.cuda.is_available():
  device = torch.device("cuda")
  print(f"GPU is available: {torch.cuda.get_device_name(0)}")
  x = torch.randn(10, 10).to(device)
  y = torch.matmul(x, x)
  print(y)
else:
    print("CUDA is not available.")

```

This script confirms the local environment can detect the NVIDIA GPU. It creates a small tensor, moves it to the GPU, and performs a matrix multiplication. This simple operation verifies that the necessary drivers and libraries are functional. The output should include the name of the GPU, and the matrix result, indicating the process is using the GPU resource. Without local GPU support, this snippet would indicate “CUDA is not available”.

**Example 2: Running a Simplified Training Loop**

This expands on the first example, performing a slightly more involved task such as training on dummy data to further test the local setup.

```python
import torch
import torch.nn as nn
import torch.optim as optim

if torch.cuda.is_available():
  device = torch.device("cuda")
  print(f"Using GPU: {torch.cuda.get_device_name(0)}")

  model = nn.Linear(10, 1).to(device)
  optimizer = optim.SGD(model.parameters(), lr=0.01)
  criterion = nn.MSELoss()

  for i in range(100):
      inputs = torch.randn(1, 10).to(device)
      target = torch.randn(1, 1).to(device)
      optimizer.zero_grad()
      output = model(inputs)
      loss = criterion(output, target)
      loss.backward()
      optimizer.step()

      if (i+1) % 10 == 0:
        print(f'Iteration: {i+1}, Loss: {loss.item()}')
else:
  print("CUDA not available, cannot run training.")
```
This example defines a very simple linear regression model and trains it on synthetic data. The model is transferred to the GPU, all calculations are performed on the GPU, and the model loss is printed every 10 iterations. This showcases a more typical scenario that leverages the GPU for computation. It’s important to remember to initialize tensors and the model on the selected device, which is done by using `.to(device)`. Failure to include this step would result in a run-time error.

**Example 3: Transferring Data from Colab to Local Kernel**

This demonstrates the practical scenario of accessing datasets that reside in Colab’s environment. Typically, datasets are downloaded or generated within the Colab environment, but a more complex data processing pipeline would involve transferring them to the local environment for local processing.

```python
import numpy as np
import os

# In Colab
data = np.random.rand(100000, 100)
# Code for transfer of data to the local kernel is embedded within the runtime
# No explicit code is needed, the runtime will handle data transfer
# Within the local kernel
if 'data' in globals(): # checking that 'data' variable was transferred
  print(f"Shape of transferred data {data.shape}")
  output_file = os.path.join(".", "local_data.npy")
  np.save(output_file, data) # processing the data locally
  print(f"Data stored locally in {output_file}")
else:
  print("Data was not successfully transferred")
```

This example emphasizes the transfer of larger datasets from Colab to the local kernel. First, dummy data is generated within the Colab notebook. When this code is executed, the variable data will be automatically copied over to the local kernel, allowing it to be used in other processing steps. This specific example stores the transferred data into a local file. This illustrates that code in Colab can generate data for local processing.

A few key points must be considered: the local environment requires a consistent library setup with the intended use case. Specifically, the libraries used in Colab need to be present and working in the local kernel. Moreover, local network configurations can interfere, and troubleshooting can be challenging. This entire process can be prone to intermittent connectivity issues. The benefit, however, is the ability to fully utilize a local GPU for computationally intensive tasks directly from a Colab notebook, allowing access to a familiar interactive environment while leveraging the horsepower of local hardware.

For a more comprehensive understanding of the local Jupyter server setup, consulting the official Jupyter documentation is essential. The documentation for the `jupyter_http_over_ws` extension will give precise details on its configuration. Deep learning frameworks like PyTorch and TensorFlow have exhaustive documentation and community support which is invaluable for setting up the necessary drivers and associated CUDA toolkit, and debugging unexpected issues. Finally, it can be advantageous to examine the documentation for Windows Subsystem for Linux to fully utilize the available options.
