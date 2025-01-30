---
title: "How can two-stream CNNs be parallelized for inference using PyTorch?"
date: "2025-01-30"
id: "how-can-two-stream-cnns-be-parallelized-for-inference"
---
In my experience optimizing video analysis pipelines for real-time applications, parallelizing two-stream Convolutional Neural Networks (CNNs) during inference is crucial for meeting latency requirements. The two-stream architecture, often consisting of a spatial and a temporal stream, lends itself naturally to parallel processing since the computations within each stream are largely independent. Exploiting this independence significantly accelerates inference throughput.

The core challenge involves feeding data to each stream, running them concurrently, and then recombining their outputs correctly. In a single-threaded or naive implementation, one would first process the spatial stream and then the temporal stream, incurring significant sequential overhead. To address this, I leverage PyTorch's multi-processing capabilities, specifically by utilizing the `torch.multiprocessing` module alongside the `torch.nn.DataParallel` functionality. The `DataParallel` class is designed to distribute computations across multiple GPUs, but its application can be conceptually extended to parallelize the two streams across different devices, even within a single GPU if necessary, or across CPU cores depending on resource availability.

The general principle I follow entails creating two separate instances of the relevant stream modules. These instances are loaded with pre-trained weights or trained alongside each other using identical models for both streams if you are training your model. Each instance is then fed its designated input: the spatial stream typically receives single frames or a stack of frames, while the temporal stream receives optical flow or a series of stacked frame differences. Crucially, the processing of each stream is managed within a separate thread (or process) facilitated by `torch.multiprocessing`.  The final step involves merging or concatenating the output feature vectors of the spatial and temporal streams, a process typically followed by a fully connected classification layer.

Let's examine this approach through code examples.

**Code Example 1: Basic Two-Stream Model Setup**

This example sets up a basic, simplified two-stream architecture for illustration.

```python
import torch
import torch.nn as nn
import torch.multiprocessing as mp

class SpatialStream(nn.Module):
    def __init__(self):
        super(SpatialStream, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        return x.view(x.size(0), -1) # Flatten for FC

class TemporalStream(nn.Module):
    def __init__(self):
        super(TemporalStream, self).__init__()
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
    def forward(self, x):
       x = self.conv1(x)
       x = self.relu(x)
       return x.view(x.size(0), -1) # Flatten for FC

class TwoStreamModel(nn.Module):
    def __init__(self):
        super(TwoStreamModel, self).__init__()
        self.spatial_stream = SpatialStream()
        self.temporal_stream = TemporalStream()
        self.fc = nn.Linear(16*25*25 + 16*25*25, 10) # For a 25x25 feature map 
    def forward(self, spatial_input, temporal_input):
        spatial_features = self.spatial_stream(spatial_input)
        temporal_features = self.temporal_stream(temporal_input)
        combined_features = torch.cat((spatial_features, temporal_features), dim=1)
        output = self.fc(combined_features)
        return output

if __name__ == '__main__':
    mp.set_start_method('spawn')  # Ensure correct multiprocessing behavior
    model = TwoStreamModel()
    spatial_data = torch.randn(1, 3, 25, 25)
    temporal_data = torch.randn(1, 2, 25, 25)
    output = model(spatial_data, temporal_data)
    print("Output shape:", output.shape)
```

This initial example defines a minimal two-stream model where the spatial stream takes a color image, and the temporal stream takes optical flow. Note the `mp.set_start_method('spawn')`. This is crucial for consistent behavior across different systems when using multiprocessing. The code demonstrates the structure but does not yet introduce parallel execution. It shows how inputs are channeled separately and their outputs are merged.

**Code Example 2: Parallel Execution with `torch.multiprocessing`**

Now, let's parallelize the execution of the two streams using `torch.multiprocessing` with separate processes.

```python
import torch
import torch.nn as nn
import torch.multiprocessing as mp

class SpatialStream(nn.Module):
    def __init__(self):
        super(SpatialStream, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        return x.view(x.size(0), -1)


class TemporalStream(nn.Module):
    def __init__(self):
        super(TemporalStream, self).__init__()
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
    def forward(self, x):
       x = self.conv1(x)
       x = self.relu(x)
       return x.view(x.size(0), -1)

class TwoStreamModel(nn.Module):
    def __init__(self):
        super(TwoStreamModel, self).__init__()
        self.spatial_stream = SpatialStream()
        self.temporal_stream = TemporalStream()
        self.fc = nn.Linear(16*25*25 + 16*25*25, 10) # Assuming 25x25 feature maps
    def forward(self, spatial_features, temporal_features):
        combined_features = torch.cat((spatial_features, temporal_features), dim=1)
        output = self.fc(combined_features)
        return output


def spatial_process(model, data, queue):
    spatial_features = model(data)
    queue.put(spatial_features)


def temporal_process(model, data, queue):
    temporal_features = model(data)
    queue.put(temporal_features)

if __name__ == '__main__':
    mp.set_start_method('spawn')
    spatial_model = SpatialStream()
    temporal_model = TemporalStream()
    two_stream_model = TwoStreamModel()
    spatial_data = torch.randn(1, 3, 25, 25)
    temporal_data = torch.randn(1, 2, 25, 25)
    spatial_queue = mp.Queue()
    temporal_queue = mp.Queue()

    spatial_process_instance = mp.Process(target=spatial_process, args=(spatial_model, spatial_data, spatial_queue))
    temporal_process_instance = mp.Process(target=temporal_process, args=(temporal_model, temporal_data, temporal_queue))

    spatial_process_instance.start()
    temporal_process_instance.start()

    spatial_features = spatial_queue.get()
    temporal_features = temporal_queue.get()

    spatial_process_instance.join()
    temporal_process_instance.join()

    output = two_stream_model(spatial_features, temporal_features)
    print("Output shape:", output.shape)
```

This example demonstrates how separate processes are launched for each stream's computation.  The `spatial_process` and `temporal_process` functions are designed to operate in these parallel processes. Each process computes features and transmits them back through a `Queue`.  The main process receives these results and feeds them into the final layer. This showcases basic process-level parallelism.

**Code Example 3: Parallel Execution using DataParallel (Conceptual Extension)**

While `DataParallel` is designed for GPU parallelism, it can, conceptually, be repurposed to run on separate devices. Below is a simplified example that illustrates this concept, even though running on different *CPU* devices might not provide substantial speedup compared to threading.

```python
import torch
import torch.nn as nn
import torch.multiprocessing as mp

class SpatialStream(nn.Module):
    def __init__(self):
        super(SpatialStream, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        return x.view(x.size(0), -1)


class TemporalStream(nn.Module):
    def __init__(self):
        super(TemporalStream, self).__init__()
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        return x.view(x.size(0), -1)


class TwoStreamModel(nn.Module):
    def __init__(self):
        super(TwoStreamModel, self).__init__()
        self.spatial_stream = SpatialStream()
        self.temporal_stream = TemporalStream()
        self.fc = nn.Linear(16*25*25 + 16*25*25, 10) # Assuming 25x25 feature maps
    def forward(self, spatial_features, temporal_features):
        combined_features = torch.cat((spatial_features, temporal_features), dim=1)
        output = self.fc(combined_features)
        return output

if __name__ == '__main__':
    mp.set_start_method('spawn')
    spatial_model = SpatialStream()
    temporal_model = TemporalStream()
    two_stream_model = TwoStreamModel()
    spatial_data = torch.randn(1, 3, 25, 25)
    temporal_data = torch.randn(1, 2, 25, 25)

    spatial_parallel_model = nn.DataParallel(spatial_model, device_ids=[0]) # Conceptual: Single CPU
    temporal_parallel_model = nn.DataParallel(temporal_model, device_ids=[0]) # Conceptual: Single CPU


    spatial_features = spatial_parallel_model(spatial_data)
    temporal_features = temporal_parallel_model(temporal_data)

    output = two_stream_model(spatial_features, temporal_features)
    print("Output shape:", output.shape)

```
Here, we demonstrate the idea of using DataParallel conceptually, even though we only assign a device ID of 0 (which represents a CPU in this case), or potentially distinct GPU IDs. In practice, running two data parallel processes on the same device will not likely yield large speedups. The principle remains, though.

This final example is not designed to show performance gains on the same CPU, but rather demonstrates that `DataParallel` can be used for conceptual parallelism. The principle remains useful when using multiple GPUs where you can map two separate device IDs to two separate streams.

For further study, I recommend exploring the PyTorch documentation on `torch.multiprocessing`, focusing on Queue usage and process management. I would also suggest investigating the intricacies of distributed training for multi-GPU workflows.  Researching specific techniques for optical flow computation and its proper application in temporal stream processing would also provide valuable insights. Finally, understanding the trade-offs between CPU and GPU processing for your specific application is critical in deciding the most efficient parallelization strategy. Specifically, look into optimizing data transfer operations between CPU and GPU which can become a bottleneck. This should be paired with careful performance testing to select the appropriate parallelization method.
