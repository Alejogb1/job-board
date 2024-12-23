---
title: "Why is model.predict() raising a NotImplementedError?"
date: "2024-12-23"
id: "why-is-modelpredict-raising-a-notimplementederror"
---

Okay, let's talk about that `NotImplementedError` you’re seeing with `model.predict()`. I've definitely been down that road myself a few times, and it’s usually not a sign of a major catastrophic problem but rather an indication of how the underlying structure of your model is set up, or perhaps *not* set up, to handle predictions. Think of it like trying to plug a power cord into a socket that’s just not designed for that type of plug - it’s a mismatch.

The core issue, in most cases, is that the base class or abstract class you're likely inheriting from to build your custom model—especially in frameworks like TensorFlow, PyTorch, or scikit-learn— often provides a blueprint, not a complete, ready-to-run predictive engine. This base class might define the `predict()` method, but it typically leaves the specific implementation details up to you, the model developer. It's a way to enforce a structure, to ensure that all models have this prediction capability, but it doesn't magically understand how *your* specific model works.

Let me give you some practical context by recounting a situation I faced early on in my deep learning journey. I was attempting to build a custom image classification model using TensorFlow. I'd meticulously constructed my layers, defined my loss function, and was feeling pretty confident. The training phase was smooth, no errors, and my metrics were showing promising signs. But then, when I tried to evaluate it using `model.predict()`, boom – `NotImplementedError`. It was jarring, to say the least.

It quickly dawned on me, that while I had defined the forward pass—i.e., the training logic—I had completely overlooked implementing the logic for inference (prediction) in a separate manner within my custom model’s class definition. The `predict()` method in the base class was there, but it was just waiting for me to *actually* write the specific code for how my model should process new, unseen data and generate outputs.

This is quite a common scenario. If you’re using a framework that offers a `Model` class that you’re subclassing, you'll typically need to override methods like `call` (or `forward` in PyTorch) for the core model logic and *sometimes* also `predict` if it is not handled by the framework.

Let’s delve into some specific examples with code:

**Example 1: TensorFlow Custom Model (Illustrating the Problem)**

```python
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, units=128):
      super(MyModel, self).__init__()
      self.dense1 = tf.keras.layers.Dense(units)
      self.dense2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

model = MyModel()
try:
    prediction = model.predict(tf.random.normal((1, 100)))
except NotImplementedError as e:
  print(f"Error Caught: {e}") # This will print the error message
```

Here, the `call` method is implemented, which is used in training; however, the `predict` method is not overridden. This model can train, but `model.predict()` raises the `NotImplementedError` because the base `tf.keras.Model` `predict` method hasn’t been given a specific implementation. It just throws the error.

**Example 2: TensorFlow Custom Model (Correcting the Issue)**

```python
import tensorflow as tf
import numpy as np


class MyCorrectedModel(tf.keras.Model):
    def __init__(self, units=128):
      super(MyCorrectedModel, self).__init__()
      self.dense1 = tf.keras.layers.Dense(units)
      self.dense2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

    def predict(self, inputs):
        return self.call(inputs).numpy()

model_correct = MyCorrectedModel()
prediction_correct = model_correct.predict(tf.random.normal((1, 100)))
print(f"Prediction Shape: {prediction_correct.shape}") # Correctly produces a prediction with shape (1, 10)
```

In this modified example, I've added a simple implementation to the `predict` method. We are essentially re-using the `call` method and converting it to a numpy array for readability, but importantly, the error is eliminated. `predict()` now calls `call()` which implements the forward pass of the model. Note: depending on your output and how your use case is structured, you might need to apply some transformations to this to tailor to your needs. This implementation would still benefit from further validation, ensuring that inputs are the expected shape etc, which is outside the scope of addressing the `NotImplementedError` itself.

**Example 3: PyTorch (Illustrating similar concept)**

```python
import torch
import torch.nn as nn

class MyPyTorchModel(nn.Module):
    def __init__(self, units=128):
        super(MyPyTorchModel, self).__init__()
        self.linear1 = nn.Linear(100, units)
        self.linear2 = nn.Linear(units, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x


model_torch = MyPyTorchModel()
with torch.no_grad():
  prediction_torch = model_torch(torch.randn(1, 100)) # Note: no direct .predict() but usage of forward() in evaluation mode is the standard approach
  print(f"Prediction Shape (PyTorch): {prediction_torch.shape}")
```

PyTorch's behavior is slightly different than TensorFlow in this respect. PyTorch models by default don't provide a dedicated `predict()` method. Instead, they rely on the `forward()` method for both training and evaluation. You simply set the model to evaluation mode, which disables things like dropout via `model.eval()`, and then pass the input through `model(input)` – which calls the `forward` function. The use of  `torch.no_grad()` is optional, however, it is considered best practice to disable gradient computations which would be done when training.

These three examples, I feel, illustrate the gist of it quite well. It's generally a matter of ensuring that the model class you’re working with, whether in Tensorflow, PyTorch or another framework, has a concrete implementation of how to use your trained model for inference. The lack of a concrete implementation of `predict()` is a very common, even if slightly irritating, source of these issues.

**Resources for Further Learning**

To get a deeper understanding of model building, I'd recommend starting with these resources:

*   **"Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville:** This is a comprehensive textbook that covers both the theoretical underpinnings and the practical aspects of deep learning, including model architectures and implementation details. Focus on chapters pertaining to model building using specific frameworks, which often explain the importance of correctly overriding methods like `call`, `forward` and sometimes `predict`.
*   **TensorFlow documentation:** The official TensorFlow documentation is extremely thorough and will help you grasp the nuances of defining custom models and the importance of properly implementing methods for inference. Pay particular attention to their sections on subclassing tf.keras.Model and how that works with the `call` and `predict` methods.
*   **PyTorch documentation:** Similar to TensorFlow’s docs, PyTorch’s documentation is your best friend. Focus on model building and how forward passes work in training vs. evaluation mode.
*  **"Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow" by Aurélien Géron:** This book provides a practical, hands-on approach to building machine learning models with these specific frameworks, which is a great stepping stone once you have a good theoretical foundation.

Hopefully, this clarifies why you might be encountering that `NotImplementedError`. Remember, it’s a standard error, more of a nudge that tells you a little implementation is still required and you’ll get past it. It's all part of the fascinating process of building and deploying effective models. Good luck!
