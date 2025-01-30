---
title: "How can GPU support be enabled in AWS SageMaker?"
date: "2025-01-30"
id: "how-can-gpu-support-be-enabled-in-aws"
---
The effective utilization of GPUs within AWS SageMaker requires a nuanced understanding of instance types, container configurations, and the specific compute needs of the machine learning workload. Ignoring this interplay can lead to either resource underutilization or performance bottlenecks. I've personally seen training jobs take hours longer than necessary due to improper GPU configuration on SageMaker, which is an avoidable pitfall.

GPU support in SageMaker isn't automatically enabled; it necessitates a conscious selection of compute instances that incorporate GPUs, as well as the correct setup within the training or inference environment. This primarily involves choosing an appropriate instance type during the creation of a training job, endpoint, or processing job, and ensuring that the container image being used is built to leverage the specific GPU hardware available on that instance.

The fundamental concept centers around matching the computational demands of your model with the processing capabilities of the underlying hardware. SageMaker provides a variety of instance types with GPUs, categorized broadly by generation (e.g., `p3`, `p4`, `g4`) and size, each with varying amounts of GPU memory, processing power, and network bandwidth. Selecting the correct instance type is crucial for ensuring performance efficiency and cost effectiveness. Furthermore, many deep learning frameworks have specific instructions for utilizing GPUs, often requiring modifications to code or environment configurations.

Let's break this down with specific examples.

**Example 1: Training with a TensorFlow Model**

Consider a scenario where I need to train a deep convolutional neural network using TensorFlow on a SageMaker training job. This will need specific modifications to utilize GPUs. Initially, I would define the `TrainingJob` specifications using the SageMaker Python SDK, selecting an instance that supports GPUs.

```python
import sagemaker
from sagemaker.tensorflow import TensorFlow

role = sagemaker.get_execution_role()

estimator = TensorFlow(
    entry_point='train.py', # Your training script
    source_dir='src',        # Path to source code
    role=role,
    instance_count=1,
    instance_type='ml.p3.2xlarge', # Choosing a GPU instance
    framework_version='2.11',
    py_version='py39',
    distribution={'parameter_server': {'enabled': True}},  # Enable distributed training if needed
    hyperparameters={
        'epochs': 50,
        'batch_size': 32,
        'learning_rate': 0.001
    }
)

estimator.fit({'training': 's3://my-bucket/my-training-data'})
```

Here, `instance_type='ml.p3.2xlarge'` specifies that I’m requesting a p3 instance with an NVIDIA Tesla V100 GPU. If your machine learning framework is not already configured to work on GPU in your docker image, or you have issues with running the code on a GPU instance, you need to modify your training script (the entry point).

Within my `train.py` script, I ensure that TensorFlow is set up to use the available GPUs:

```python
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPUs are available")
    try:
        # Restrict TensorFlow to use only specific GPUs
        tf.config.set_visible_devices(gpus[0], 'GPU')

        # Ensure that TF is running on the GPU, not the CPU
        tf.config.experimental.set_memory_growth(gpus[0], True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

    except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
        print(e)
else:
    print("No GPUs available")
# ... the rest of your TensorFlow model building and training code ...
```

This code snippet first checks for the presence of GPUs. It then attempts to set the first GPU as visible to TensorFlow, preventing it from using other devices. The `set_memory_growth` line is often critical for GPU memory management in TensorFlow. Finally, it prints the number of available physical and logical GPUs. If you are using multiple GPUs, you would configure accordingly. Without such explicit steps within the training script, the training process might default to using the CPU, which negates the benefits of the GPU instance.

**Example 2: Inference Endpoint with PyTorch**

A similar principle applies when deploying a PyTorch model for inference on a SageMaker endpoint. Consider the following configuration for deploying a pre-trained ResNet model using the SageMaker PyTorch SDK:

```python
import sagemaker
from sagemaker.pytorch import PyTorchModel
from sagemaker.predictor import Predictor

role = sagemaker.get_execution_role()

model_data = 's3://my-bucket/my-model/model.tar.gz'

pytorch_model = PyTorchModel(
    model_data=model_data,
    role=role,
    entry_point='inference.py',  # Your inference script
    source_dir='src',
    framework_version='2.0',
    py_version='py310',
    instance_type='ml.g4dn.xlarge' # Example GPU instance for inference
)

predictor = pytorch_model.deploy(
    initial_instance_count=1,
    instance_type='ml.g4dn.xlarge',
    predictor_cls=Predictor # or your custom Predictor class
)
```

In this setup, `instance_type='ml.g4dn.xlarge'` is chosen, which provides an NVIDIA T4 GPU. In the accompanying `inference.py`, similar logic needs to be added. The code should load the PyTorch model and move it to the GPU:

```python
import torch
import torch.nn as nn

# Load your model
model = ... # instantiate your PyTorch model

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)
model.eval()  # Set model in evaluation mode

def model_fn(model_dir):
    # Load model from model_dir
    model = ... # Re-instantiate and load the PyTorch model
    model.to(device)
    model.eval()
    return model

def predict_fn(data, model):
    data_tensor = ... # Convert input data to tensor
    data_tensor = data_tensor.to(device)
    with torch.no_grad():
        output = model(data_tensor)
    return output.cpu().numpy()
```

This `inference.py` will first check for CUDA availability and load model into the GPU device if available. It then uses the predict function to perform inference on the GPU device and return the results. Without specifically ensuring that the model is moved to the GPU, the inference workload will be performed on the CPU, which greatly impacts the performance, especially for more complex deep learning models. The SageMaker runtime environment often handles device allocation, but explicit device control is necessary.

**Example 3: Processing Job for Data Transformation**

GPUs can also accelerate data transformation tasks, not just model training and inference. This is especially true when you have a lot of images, audio, or similar data that might be difficult to work with on CPU only. Here’s how you might specify a processing job with GPU enabled:

```python
import sagemaker
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor

role = sagemaker.get_execution_role()

processor = ScriptProcessor(
    command=['python3'],
    image_uri='your-custom-image-uri',  # Your custom container image
    role=role,
    instance_count=1,
    instance_type='ml.g5.2xlarge', # Example of a more recent generation GPU instance
    base_job_name='data-processing',
)

inputs = [
    ProcessingInput(
        source='s3://my-bucket/input-data',
        destination='/opt/ml/processing/input',
        input_name='input_data',
    )
]

outputs = [
    ProcessingOutput(
        source='/opt/ml/processing/output',
        destination='s3://my-bucket/output-data',
        output_name='output_data'
    )
]

processor.run(
    inputs=inputs,
    outputs=outputs,
    code='process.py'
)
```

Again, `instance_type='ml.g5.2xlarge'` selects a GPU instance, specifically one based on NVIDIA A10G GPUs. The image, indicated with `image_uri`, should contain all necessary dependencies, including your framework libraries and CUDA drivers. The `process.py` needs to be written to take the available data and process them using the GPU if needed.

```python
import torch
import os

# Check CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load and process data (Example)
input_path = '/opt/ml/processing/input/input_data'
output_path = '/opt/ml/processing/output'

if device.type == "cuda":
    # Process data using GPU
    print ("Processing on GPU")
    data = torch.rand(1000, 1000).to(device)
    result = torch.matmul(data, data.T)
else:
    print("Processing on CPU")
    data = torch.rand(1000,1000)
    result = torch.matmul(data, data.T)

os.makedirs(output_path, exist_ok=True)
# Save results to output directory.
# example: np.save(os.path.join(output_path, "processed_data.npy"), result.cpu().numpy())
print(f"Data processing complete. Saved to {output_path}")
```

The `process.py` here includes the device check and uses GPU if available. The data is moved to GPU before processing and the result is then saved.

In summary, enabling GPU support in SageMaker requires a deliberate choice of GPU-equipped instances and meticulous configuration of the machine learning framework to effectively utilize these GPUs. It's essential to check which frameworks support GPU acceleration, and install the necessary drivers (usually pre-configured in many of the SageMaker docker images). This combination of environment configuration and code optimization is what ultimately provides the desired compute acceleration.

For further learning, I recommend consulting resources such as the AWS documentation for SageMaker instance types, the documentation for specific machine learning frameworks (e.g., TensorFlow, PyTorch, MXNet) regarding GPU usage, and community tutorials on GPU-accelerated deep learning. These sources provide more detail on driver management and specific configuration issues for various frameworks. Also, I would suggest familiarizing oneself with container concepts using Docker as well as container image repositories.
