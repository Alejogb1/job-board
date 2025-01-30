---
title: "How can I load a model only once in a fastAPI application?"
date: "2025-01-30"
id: "how-can-i-load-a-model-only-once"
---
A common bottleneck in FastAPI applications that leverage machine learning models arises from repeatedly loading the model for each incoming request. This incurs significant latency and unnecessary resource consumption. My experience building a real-time fraud detection API taught me the critical need for model persistence; reloading a model, often a complex neural network, every time a request came through rendered the application unusable. The solution, which I’ve implemented across several projects, lies in loading the model once during application startup and sharing it across request handlers.

**Explanation: Global Application State and FastAPI Dependency Injection**

The fundamental principle here is to establish a global application state that persists throughout the application’s lifecycle. This state holds the model object in memory, making it directly accessible by any endpoint handling a request. FastAPI’s dependency injection system provides the means to propagate this global state effectively. We avoid using global variables directly which can lead to unexpected behavior in concurrent environments. Instead, we will employ the dependency injection mechanisms to ensure thread-safe access to our model.

The core components involved are:

1.  **Model Loading Function:** A function responsible for loading the model from its source (e.g., a saved file or cloud storage). This function is called only once at application startup.
2.  **Global State Container:** An object, usually a dictionary or a custom class, that holds the loaded model instance. This container is outside the scope of any specific endpoint.
3.  **Dependency Function:** A function that retrieves the model from the global state container. This function is injected into endpoint dependencies, granting access to the loaded model.
4.  **Startup Event:** A FastAPI event handler that executes during application startup. This handler is where the model loading function is called, and the model instance is added to the global state.

This approach circumvents the repeated model loading for each request and ensures that only a single instance of the model is present within the application runtime, thus improving performance significantly. It is also important to be mindful of the resources consumed by the model, especially in production environments. I've often found it useful to monitor memory usage and potentially explore model quantization or model serving solutions depending on the scale and constraints.

**Code Examples**

Here are three code snippets illustrating various approaches:

**Example 1: Using a Dictionary as Global State**

```python
from fastapi import FastAPI, Depends
from typing import Dict
import time
import torch # Replace with your model loading library

app = FastAPI()

model_cache: Dict = {}  # Initialize an empty dictionary for caching

def load_model():
    """Simulates loading a model."""
    print("Model loading...")
    time.sleep(2)  # Simulate loading time
    model = torch.nn.Linear(10, 2)  # Mock model, replace with actual model load
    print("Model loaded successfully")
    return model

async def get_model() :
   if 'model' not in model_cache:
      model_cache['model'] = await load_model() # await if it's an async loading op
   return model_cache['model']

@app.on_event("startup")
async def startup_event():
    print("Initializing application...")
    _ = await get_model()
    print("Application initialized.")


@app.get("/predict")
async def predict(model = Depends(get_model)):
    """ Endpoint that uses the model. """
    input_tensor = torch.randn(1, 10)
    output = model(input_tensor)
    return {"prediction": output.tolist()}

```

**Commentary:**

In this example, `model_cache` acts as the global state. The `load_model` function simulates loading a model and is called from within `get_model` if the model is not already present. The `@app.on_event("startup")` decorator ensures `get_model()` executes during startup before the API accepts any requests. Then, within `/predict` endpoint, `get_model` is used as a dependency that provides access to the model loaded in memory. The first call will load the model, subsequent calls will return the model object in memory.
This example demonstrates a basic implementation and can be used with simple model loading strategies. Note the async functions here; these are particularly useful if model loading (or other startup actions) involve I/O bound operations and allow non-blocking behavior.

**Example 2: Using a Custom Class as Global State**

```python
from fastapi import FastAPI, Depends
from typing import Dict
import time
import torch

app = FastAPI()

class ModelManager:
    def __init__(self):
        self.model = None

    async def load_model(self):
        if self.model is None:
            print("Model loading...")
            time.sleep(2)
            self.model = torch.nn.Linear(10, 2)
            print("Model loaded successfully")

    def get_model(self):
        return self.model

model_manager = ModelManager()

async def get_cached_model( ) :
    if model_manager.model is None:
      await model_manager.load_model()
    return model_manager.get_model()


@app.on_event("startup")
async def startup_event():
   print("Initializing application...")
   _ = await get_cached_model()
   print("Application initialized.")

@app.get("/predict")
async def predict(model = Depends(get_cached_model)):
    input_tensor = torch.randn(1, 10)
    output = model(input_tensor)
    return {"prediction": output.tolist()}
```

**Commentary:**

This approach encapsulates the model and its loading logic within a `ModelManager` class, providing a slightly more structured way to handle the model state. `model_manager` is instantiated globally. `startup_event` now initializes the model via the manager during startup. The endpoint uses the `get_cached_model()` as a dependency to access the loaded model. Using a class becomes particularly beneficial when your model requires additional setup like defining inference parameters or when performing resource management.

**Example 3: Using a Singleton Pattern**
```python
from fastapi import FastAPI, Depends
import time
import torch
from typing import Optional

app = FastAPI()

class ModelSingleton:
    _instance: Optional["ModelSingleton"] = None

    def __init__(self):
        print("Model loading...")
        time.sleep(2)
        self.model = torch.nn.Linear(10, 2)
        print("Model loaded successfully")

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_model(self):
        return self.model

async def get_singleton_model() :
    return ModelSingleton.get_instance().get_model()

@app.on_event("startup")
async def startup_event():
    print("Initializing application...")
    _ = await get_singleton_model()
    print("Application initialized.")

@app.get("/predict")
async def predict(model = Depends(get_singleton_model)):
    input_tensor = torch.randn(1, 10)
    output = model(input_tensor)
    return {"prediction": output.tolist()}
```

**Commentary:**

Here, I've implemented a singleton pattern using a class `ModelSingleton`. The class ensures that only one instance of itself is ever created. During startup, the class's `get_instance` method is called the first time, loading the model, subsequent calls will return the same class instance. Within the `/predict` handler, we inject the model via `get_singleton_model` dependency. This design pattern is often useful for scenarios where only one instance of an object is needed across the application.

**Resource Recommendations**

To deepen your understanding, I recommend exploring the following concepts:

1.  **Dependency Injection:**  Study dependency injection patterns. Look into resources that specifically discuss dependency injection within the context of Python and its web frameworks. This would provide clarity on structuring reusable components and making your code more maintainable.
2.  **Singleton Design Pattern:** Dive into the concept of singleton pattern. Understand its uses and when to apply it. Research its implementation in Python.
3.  **FastAPI Events:** Become familiar with FastAPI’s lifecycle events. Specifically, learn how to effectively employ the `startup` and `shutdown` events to manage the initialization and cleanup of application resources.
4.  **Concurrency and Asynchronous Programming:** Explore concurrency and asynchronous programming concepts. Specifically, examine how they relate to handling multiple concurrent requests. Look for materials on Python's `asyncio` module to gain proficiency in its use.

Mastering these concepts allows you to optimize your FastAPI applications for efficient model serving, enhancing performance and reducing resource consumption. I hope this provides you a solid foundation for implementing efficient model loading within your FastAPI projects.
