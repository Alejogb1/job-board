---
title: "How do I save a torch model in Azure ML?"
date: "2025-01-30"
id: "how-do-i-save-a-torch-model-in"
---
The core challenge in saving a PyTorch model within Azure Machine Learning (Azure ML) stems from the need to encapsulate the model's architecture, learned parameters, and any associated metadata in a format that can be reliably stored, versioned, and deployed. Azure ML offers several mechanisms for this, but I've found utilizing the `mlflow` integration, specifically with `mlflow.pytorch`, provides the most streamlined approach. I've spent the last five years deploying deep learning models on various cloud platforms, and the native integration with tracking systems has proven invaluable.

The fundamental principle involves serializing the PyTorch model, which typically exists as an in-memory object, into a persistent storage format. The most common formats are `pickle` and the ONNX (Open Neural Network Exchange) format. While `pickle` is straightforward, it is generally recommended to use ONNX for production deployments because it promotes interoperability with different frameworks and runtime environments. When deploying to Azure ML, this serialization is handled by the `mlflow` library, which also allows us to log associated metrics and parameters for experiment tracking and reproducibility. This capability is essential when you manage multiple experiments and models within an organization.

Here's how this process typically unfolds, beginning with the model training:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch

# Define a simple model
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)

# Initialize the model, loss, and optimizer
model = SimpleNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Dummy training data
X = torch.randn(100, 10)
y = torch.randint(0, 2, (100,))

# Start an mlflow run
with mlflow.start_run():
  # Log model parameters, for example hidden layer size.
  mlflow.log_param("hidden_size", 10)

  # Basic training loop
  for epoch in range(2): # Just for brevity
      optimizer.zero_grad()
      outputs = model(X)
      loss = criterion(outputs, y)
      loss.backward()
      optimizer.step()

      # Log training metrics, like the loss, at each step.
      mlflow.log_metric("training_loss", loss.item(), step=epoch)

  # Log the trained model with mlflow
  mlflow.pytorch.log_model(model, "simple_model")
```

**Commentary:** This code snippet illustrates the core elements of integrating PyTorch with `mlflow`. It defines a simple linear model and performs training iterations. The key point is the line `mlflow.pytorch.log_model(model, "simple_model")`. This single function call does a lot. It serializes the trained model into a format that `mlflow` can understand and stores it in a dedicated location within the MLflow artifact store associated with the current Azure ML experiment. By specifying "simple\_model" as the artifact path, we can easily retrieve the model later using that specific name. The `mlflow.start_run()` ensures that these logging operations are scoped to a specific MLflow run, providing experiment tracking. Logging training metrics like loss allows for better analysis and validation of model performance after experiments.

Now, consider a scenario where you need to incorporate custom pre- or post-processing functions with the model. In those cases, using the `PyTorchModel` flavor of `mlflow` proves very useful:

```python
import torch
import torch.nn as nn
import mlflow
import mlflow.pyfunc

# Simple custom preprocessing
def preprocess(input_data):
  return torch.tensor(input_data).float() / 255.0

# Simple custom post-processing
def postprocess(prediction):
  return torch.argmax(prediction, dim=1).tolist()

# Redefine the same model
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)


class CustomModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model, preprocessor, postprocessor):
        self.model = model
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor

    def predict(self, context, model_input):
        processed_input = self.preprocessor(model_input)
        raw_predictions = self.model(processed_input)
        return self.postprocessor(raw_predictions)


model = SimpleNet()

# Wrap model, preprocessor and postprocessor into custom mlflow wrapper class
wrapped_model = CustomModelWrapper(model, preprocess, postprocess)

# Dummy data to train the model
X = torch.randn(100, 10)
y = torch.randint(0, 2, (100,))
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# Start mlflow run and log
with mlflow.start_run():
  for epoch in range(2):
      optimizer.zero_grad()
      outputs = model(X)
      loss = criterion(outputs, y)
      loss.backward()
      optimizer.step()
  mlflow.pyfunc.log_model(python_model=wrapped_model, artifact_path="custom_model")
```

**Commentary:** In this example, I've used the `mlflow.pyfunc.PythonModel` interface to wrap our PyTorch model with custom preprocessing and postprocessing. This encapsulates the complete pipeline, crucial for consistent deployments.  The `predict` method takes care of transforming the input data before it is fed to the model and post-processing the prediction. When using the `log_model` function, the full wrapper object is serialized to a file to persist. This method ensures that any logic and transformations around model inference are also saved with the model.

Finally, when you have a more complex model and want to specify custom environments or dependencies, leveraging the `conda_env` parameter is highly recommended. This ensures consistent runtime behavior between environments.

```python
import torch
import torch.nn as nn
import mlflow
import mlflow.pytorch
import os

# Same basic model from before
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)

model = SimpleNet()

# Create a custom conda environment
conda_env = {
    "channels": ["conda-forge"],
    "dependencies": [
        "python=3.8",
        "pytorch",
        "mlflow",
        "numpy"
    ],
    "name": "custom_env"
}


# Dummy training data and initialization (similar to previous example)
X = torch.randn(100, 10)
y = torch.randint(0, 2, (100,))
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# Start a mlflow run and log
with mlflow.start_run():
  for epoch in range(2):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
  mlflow.pytorch.log_model(model, "simple_model_with_env", conda_env=conda_env)
```

**Commentary:** Here, the `conda_env` parameter within `mlflow.pytorch.log_model` allows us to define the necessary Python environment. Specifying the versions of key packages such as `pytorch` or `mlflow` or defining the channels where packages should be installed from are all very useful tools for ensuring compatibility and consistency.  When this model is loaded, `mlflow` will ensure this environment is created and used to run the model. This is essential when you have complex setups, custom packages, or specific library requirements. It greatly minimizes issues due to inconsistencies in package versions across development and deployment environments.

For further study, I would recommend exploring the official documentation for the `mlflow` library, specifically focusing on the `mlflow.pytorch` and `mlflow.pyfunc` modules. In particular, the examples provided in the `mlflow` documentation, particularly regarding custom deployment flavors, are quite useful. Additionally, the Azure ML documentation regarding model management provides very valuable insights into the integration between `mlflow` and the broader Azure ecosystem. Studying the best practices on Azure ML deployment documentation and the guidance on environment management can also provide practical tips for building robust, repeatable deployment workflows. A deeper dive into the ONNX format will help clarify the interoperability benefits it brings when compared to native framework serialization approaches such as pickle. These resources combined will provide the necessary theoretical and practical background for deploying PyTorch models reliably within Azure ML.
