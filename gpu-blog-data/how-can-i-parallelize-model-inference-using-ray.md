---
title: "How can I parallelize model inference using Ray without reloading the model?"
date: "2025-01-30"
id: "how-can-i-parallelize-model-inference-using-ray"
---
Achieving efficient parallel model inference without redundant model loading is a critical aspect of deploying machine learning applications at scale. Ray, a unified framework for scaling Python applications, offers a robust solution to this challenge through its distributed object store and task-based parallelism. My experience building large-scale recommender systems has repeatedly highlighted the inefficiencies of reloading models across worker processes. Ray's actor model allows us to load a model once and share its reference across parallel tasks, significantly reducing resource overhead and latency.

The core idea is to encapsulate model loading within a Ray actor. An actor is a class whose methods can be invoked remotely and concurrently, persisting state between invocations. This differs from standard Ray tasks which are stateless functions. The actor holds the loaded model in memory, and worker processes, invoked as tasks, then interact with this actor for inference. This architecture avoids the cost of model re-initialization for each inference request.

Here’s how this process works in practice, and its specific advantages:

1.  **Model Loading within the Actor:** The first step involves defining a Ray actor class. The model is loaded within the actor's `__init__` method. This method is executed only once when the actor is instantiated, guaranteeing the model is loaded only once. The model instance becomes part of the actor’s state.

2.  **Inference as an Actor Method:** The actor class also includes a method, for example, `infer`, that performs inference using the loaded model. This method accepts the input data and returns the model's predictions. Each call to the actor's method utilizes the already loaded model, avoiding any re-initialization.

3.  **Task Submission to the Actor:** Ray tasks are then created which invoke the actor's inference method. These tasks operate on different input data in parallel, each utilizing the same model loaded into the actor’s memory. Ray handles scheduling tasks across cluster resources, distributing the workload efficiently.

This methodology significantly reduces overhead because the potentially large and time-consuming operation of loading the model, is performed only once, no matter how many inference requests are made. This can result in substantial performance gains, particularly when dealing with large models. Additionally, the model resides in the shared memory space of the actor, eliminating the need to transfer the model itself with each inference request.

Here are some practical examples demonstrating this pattern:

**Example 1: Basic Model Inference with a Simple Scikit-learn Model**

This example shows a basic regression model. Although not computationally expensive to load, the principle remains the same.

```python
import ray
from sklearn.linear_model import LinearRegression
import numpy as np

@ray.remote
class ModelActor:
    def __init__(self):
        self.model = LinearRegression()
        X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
        y = np.dot(X, np.array([1, 2])) + 3
        self.model.fit(X, y)

    def infer(self, input_data):
        return self.model.predict(input_data)

if __name__ == '__main__':
    ray.init()
    model_actor = ModelActor.remote()

    input_data = [np.array([[3, 3]]), np.array([[4, 4]]), np.array([[5, 5]])]
    results = ray.get([model_actor.infer.remote(data) for data in input_data])
    print(results)
    ray.shutdown()
```

*Commentary:* First, we define a `ModelActor` class, decorated with `@ray.remote` to make it a Ray actor. The `__init__` method loads the LinearRegression model. The `infer` method uses the instance of the model. We then instantiate the actor and use `infer.remote` to asynchronously call the inference method with different input. `ray.get` is used to collect the results. The model is loaded only once within the actor's initialization, regardless of how many inference requests we dispatch.

**Example 2: Inference with a More Complex TensorFlow Model**

This demonstrates a realistic scenario where model loading has a significant time cost.

```python
import ray
import tensorflow as tf
import numpy as np

@ray.remote
class TFModelActor:
    def __init__(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation='relu', input_shape=(784,)),
            tf.keras.layers.Dense(10)
        ])
        self.model.compile(optimizer='adam', loss='mse')
        # Dummy training step to finalize graph
        self.model.train_on_batch(np.random.rand(1, 784), np.random.rand(1,10))


    def infer(self, input_data):
        return self.model.predict(input_data)

if __name__ == '__main__':
    ray.init()
    model_actor = TFModelActor.remote()

    input_data = [np.random.rand(1, 784) for _ in range(3)]

    results = ray.get([model_actor.infer.remote(data) for data in input_data])
    print(results)
    ray.shutdown()
```

*Commentary:* In this case, we load a TensorFlow Keras model. Similar to the previous example, the model is initialized within the actor, avoiding reloads. A dummy training batch is included to fully initialize the TensorFlow graph. We use the `predict` method for inference, and send multiple batches of random data to the model using ray task parallelism. The computational cost of model initialization is paid only once.

**Example 3: Handling Model Updates/State with Actor Methods**

This highlights the actor's capability to modify its own state, demonstrating how one might update a model.

```python
import ray
import numpy as np
from sklearn.linear_model import LinearRegression

@ray.remote
class UpdatableModelActor:
    def __init__(self):
       self.model = LinearRegression()
       X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
       y = np.dot(X, np.array([1, 2])) + 3
       self.model.fit(X, y)

    def infer(self, input_data):
       return self.model.predict(input_data)

    def update_model(self, new_X, new_y):
       self.model.fit(new_X, new_y)

if __name__ == '__main__':
    ray.init()
    model_actor = UpdatableModelActor.remote()

    # Initial inference
    initial_data = np.array([[3, 3]])
    initial_result = ray.get(model_actor.infer.remote(initial_data))
    print(f"Initial inference: {initial_result}")


    # Update the model
    new_X = np.array([[3, 3],[4, 4]])
    new_y = np.array([12, 16])

    ray.get(model_actor.update_model.remote(new_X, new_y))
    # New inference with updated model
    updated_result = ray.get(model_actor.infer.remote(initial_data))
    print(f"Inference after model update: {updated_result}")
    ray.shutdown()
```

*Commentary:* Here, besides the `infer` method, we’ve included `update_model`. The model is trained initially within the `__init__` method. We perform an initial inference, update the model state using `update_model`, then perform another inference using the same instance of the model object with updated state. This showcases how actors are not just used for stateless inference but can also represent persistent entities with a modifiable state.

**Resource Recommendations**

To deepen understanding, I suggest consulting documentation on distributed computing concepts, specifically focusing on shared-memory paradigms. Further research into asynchronous programming in Python, using libraries like `asyncio`, will also clarify how the Ray API functions. Reviewing advanced usage of object stores in distributed systems will also provide a solid theoretical foundation. Additionally, exploring the actor model and message passing patterns used in various frameworks can improve understanding of Ray's design choices. Frameworks like Erlang’s OTP can provide helpful background. Finally, examining the documentation of machine learning frameworks (TensorFlow, PyTorch) regarding their resource management practices when used in distributed settings is invaluable. These will provide a much richer view of how Ray simplifies these complex issues.
