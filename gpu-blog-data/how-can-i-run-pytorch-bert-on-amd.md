---
title: "How can I run PyTorch BERT on AMD hardware?"
date: "2025-01-30"
id: "how-can-i-run-pytorch-bert-on-amd"
---
Utilizing PyTorch BERT effectively on AMD hardware requires careful consideration of software libraries, optimized kernel implementations, and potential workarounds due to the varying support landscape compared to NVIDIA GPUs. While the core PyTorch framework functions on AMD, maximizing performance, particularly for computationally intensive models like BERT, necessitates leveraging the ROCm (Radeon Open Compute) ecosystem. This response focuses on practical steps and considerations I've encountered during my experience optimizing model training on AMD platforms.

The central challenge stems from CUDA being the dominant acceleration backend for PyTorch. While PyTorch provides CPU execution, performance can become a significant bottleneck, especially with large models. Therefore, to leverage AMD GPUs, one must transition from the traditional CUDA-centric workflow to the ROCm framework. This transition involves ensuring that a compatible version of ROCm is installed and that PyTorch is built with ROCm support.

Specifically, ROCm provides the low-level APIs and libraries necessary to communicate with AMD GPUs and allows PyTorch to delegate compute operations to the hardware. The core components are the ROCm runtime (HIP, or Heterogeneous-compute Interface for Portability), the ROCm compiler (ROCm Clang), and the necessary device drivers. The initial setup is crucial, as mismatched driver versions or an incomplete ROCm installation will hinder PyTorch’s ability to utilize the GPU, leading to errors or suboptimal performance.

Once ROCm is installed and verified, one must ensure that a version of PyTorch with ROCm/HIP support is used. This often means installing PyTorch from specific channels which are built against ROCm rather than relying on default PyTorch binaries. Often, pre-built binaries from PyTorch's official website do not include ROCm support. You will need to explore alternative channels like those from AMD or third parties which provide specific ROCm versions. This can also mean building PyTorch from source which is more involved, but provides the most flexibility for control.

The core difference in code between using CUDA and ROCm with PyTorch for BERT largely manifests in the device selection. Where `torch.device("cuda")` is used for NVIDIA, `torch.device("cuda")` continues to be the interface for AMD GPUs, with ROCm emulating a CUDA API. Despite the consistent API, the underlying hardware will differ, so it’s essential to confirm the device is recognized before running your model training. This might involve inspecting `torch.cuda.is_available()` and `torch.cuda.get_device_name(0)` to see which backend is being used and if your device is present.

The first code example demonstrates basic device assignment with a PyTorch BERT model. Note that the model and tensors are explicitly moved to the designated device.

```python
import torch
from transformers import BertModel, BertTokenizer

# Check if a CUDA device is available, even with ROCm, it reports as cuda
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using AMD GPU with CUDA emulation!")
    print("Device Name:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("Using CPU.")

# Load a pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

# Move the model to the selected device
model.to(device)

# Prepare some sample input
text = "This is an example sentence for BERT."
inputs = tokenizer(text, return_tensors="pt").to(device)

# Pass the input through the BERT model
outputs = model(**inputs)

print("Output shape:", outputs.last_hidden_state.shape)
```

This code begins by checking if a CUDA-like device is available. As noted, ROCm presents an interface compatible with CUDA calls, so `torch.cuda.is_available()` returns True if ROCm is correctly set up. The code then retrieves the device name (which will identify the AMD GPU) and moves the model and input tensors to the device. The `transformers` library works seamlessly with the appropriate device type. The output shape is printed to verify the computation.

The second example illustrates how to use `torch.nn.DataParallel` to distribute model training across multiple GPUs, if you have more than one AMD device. This section also emphasizes the necessity of checking for CUDA compatibility.

```python
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

# Check for multiple GPUs
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    device_ids = list(range(torch.cuda.device_count())) # List of GPUs available for DP
else:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using AMD GPU with CUDA emulation!")
        print("Device Name:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("Using CPU.")
    device_ids = None

# Load a pre-trained BERT model
model = BertModel.from_pretrained("bert-base-uncased")

if device_ids:
    # Wrap the model for data parallel training
    model = nn.DataParallel(model, device_ids = device_ids)
    model.to(device_ids[0])
else:
    # Move the model to single device
    model.to(device)


# Load a pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Prepare sample input
texts = ["This is the first sentence.", "And this is the second."]
inputs = tokenizer(texts, padding = True, truncation = True, return_tensors = 'pt')

if device_ids:
    inputs = inputs.to(device_ids[0])
else:
    inputs = inputs.to(device)


# Run the model
outputs = model(**inputs)

print("Output Shape:", outputs.last_hidden_state.shape)
```

This section uses `torch.cuda.device_count()` to see if multiple GPUs are present and the code uses `nn.DataParallel` to distribute the training process. It ensures all tensors are sent to the first device in the device list in the multi-gpu case or the only device in the single GPU or CPU case. It’s crucial to understand that data parallelism will not always guarantee the best scaling on AMD. While this works with ROCm in the same way as NVIDIA, specialized data parallel methods might be more performant. The use of `DataParallel` is provided to illustrate the analogous approach with PyTorch on AMD, highlighting how it is often possible to use existing CUDA-based code with minimal changes.

Finally, here is a third example that highlights potential data loading and training implications with more realistic data.

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer
import numpy as np

# Custom Dataset for sample data
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        inputs = self.tokenizer(text, padding = True, truncation = True, return_tensors="pt")
        return inputs, torch.tensor(label)

# Check device and data parallel setup
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    device_ids = list(range(torch.cuda.device_count())) # List of GPUs available for DP
else:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using AMD GPU with CUDA emulation!")
        print("Device Name:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("Using CPU.")
    device_ids = None

# Load a pre-trained BERT model
model = BertModel.from_pretrained("bert-base-uncased")

if device_ids:
    # Wrap the model for data parallel training
    model = nn.DataParallel(model, device_ids = device_ids)
    model.to(device_ids[0])
else:
    # Move the model to single device
    model.to(device)

# Load a pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Create sample data
texts = [
    "This is the first sentence of the dataset.",
    "Here is another sentence for training.",
    "A final example before we start the process."
]
labels = [0, 1, 0]
dataset = TextDataset(texts, labels, tokenizer)
dataloader = DataLoader(dataset, batch_size = 2)

# Optimizer and Loss function
if device_ids:
    optimizer = torch.optim.Adam(model.module.parameters(), lr = 1e-5)
else:
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-5)
loss_function = nn.CrossEntropyLoss()

# Train loop (basic)
for inputs, labels in dataloader:
    if device_ids:
        inputs = {k:v.to(device_ids[0]) for k,v in inputs.items()}
        labels = labels.to(device_ids[0])
    else:
         inputs = {k:v.to(device) for k,v in inputs.items()}
         labels = labels.to(device)

    optimizer.zero_grad()
    outputs = model(**inputs).last_hidden_state
    output_logits = torch.mean(outputs, dim=1) # Simple classification head
    loss = loss_function(output_logits, labels)
    loss.backward()
    optimizer.step()

    print("Loss:", loss.item())
```

Here a `TextDataset` is constructed to highlight how data loading would occur. The dataset will generate batches of tokenized text and labels, and the training loop processes these, calculating loss, performing backpropagation, and updating model parameters. The relevant device assignment for both model and tensor data is important. This example also showcases how data is moved to the correct devices when running on more than one GPU. It demonstrates a more practical use case, where the full pipeline from data loading to optimization is represented with AMD compatibility.

For resources, I recommend starting with the official AMD ROCm documentation; it provides detailed instructions on installation, driver configuration, and API usage. The PyTorch documentation on GPU usage is also invaluable, although it may be biased towards NVIDIA, it provides core understanding of device allocation and execution, often this works with AMD due to the CUDA interface emulation. Furthermore, exploring the documentation for any AMD-optimized libraries available for linear algebra, as well as the Hugging Face Transformers library for BERT model specifics, is also worthwhile. Performance tuning can require exploring compiler options within the ROCm toolchain and potentially experimenting with different optimization flags. The primary sources for information and troubleshooting must be centered around the AMD ROCm website and community forums.
