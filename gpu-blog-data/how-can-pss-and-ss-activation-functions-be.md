---
title: "How can PSS and SS activation functions be implemented in Keras?"
date: "2025-01-30"
id: "how-can-pss-and-ss-activation-functions-be"
---
The inherent limitations of sigmoid-based activation functions, particularly the vanishing gradient problem, necessitate careful consideration when choosing an activation function for deep learning models. While the Parametric Sigmoid Sigmoid (PSS) and Smooth Sigmoid (SS) functions offer potential improvements over the standard sigmoid, their implementation within the Keras framework requires a nuanced understanding of custom layer creation and numerical stability.  My experience in developing high-performance neural networks for image recognition highlighted the need for precise control over these less-common activation functions.


**1. Clear Explanation:**

The standard sigmoid activation function, σ(x) = 1 / (1 + exp(-x)), suffers from the vanishing gradient problem, where gradients become extremely small during backpropagation, hindering effective training of deep networks.  PSS and SS functions aim to mitigate this issue by modifying the sigmoid's shape and gradient characteristics.

The Parametric Sigmoid Sigmoid (PSS) function introduces a parameter, typically denoted as ‘α’, which controls the steepness of the sigmoid curve.  A higher α value results in a steeper curve, potentially addressing the vanishing gradient issue by ensuring larger gradients in certain regions.  The formula for PSS is generally expressed as σ(αx), effectively scaling the input before applying the standard sigmoid function.

The Smooth Sigmoid (SS) function introduces smoothness through a carefully chosen approximation. While various approximations exist, a common approach uses a rational function or a polynomial approximation to mimic the sigmoid's behavior while providing smoother gradients throughout its range.  The specific formula varies depending on the chosen approximation method, but the goal is to maintain a sigmoidal shape while avoiding the sharp transitions that can cause gradient instability.

Implementing these in Keras requires leveraging Keras's custom layer functionality.  This allows for precise definition of the activation function's forward and backward propagation steps, ensuring correct gradient calculation during training.

**2. Code Examples with Commentary:**

**Example 1: Implementing PSS using a custom layer:**

```python
import tensorflow as tf
from tensorflow import keras

class PSSLayer(keras.layers.Layer):
    def __init__(self, alpha=2.0, **kwargs):
        super(PSSLayer, self).__init__(**kwargs)
        self.alpha = alpha

    def call(self, inputs):
        return tf.keras.activations.sigmoid(self.alpha * inputs)

    def get_config(self):
        config = super(PSSLayer, self).get_config()
        config.update({'alpha': self.alpha})
        return config


model = keras.Sequential([
    keras.layers.Dense(64, input_shape=(784,)),
    PSSLayer(alpha=3.0), # Example usage with alpha = 3.0
    keras.layers.Dense(10, activation='softmax')
])

model.compile(...) #Rest of model compilation
```

This example defines a `PSSLayer` that inherits from `keras.layers.Layer`. The `__init__` method initializes the alpha parameter. The `call` method performs the forward pass using TensorFlow's built-in sigmoid activation function after scaling the input by alpha. The `get_config` method is crucial for saving and loading the model, ensuring the alpha value is preserved.


**Example 2: Implementing a Polynomial Approximation of SS using a Lambda layer:**

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np

def smooth_sigmoid(x):
    # Example polynomial approximation.  Higher-order polynomials can improve accuracy.
    return x / (1 + np.abs(x))


model = keras.Sequential([
    keras.layers.Dense(64, input_shape=(784,)),
    keras.layers.Lambda(lambda x: smooth_sigmoid(x)),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(...) #Rest of model compilation

```

This approach utilizes Keras's `Lambda` layer for a more concise implementation. The `smooth_sigmoid` function represents a polynomial approximation of the smooth sigmoid.  Note that the choice of polynomial and its coefficients significantly impacts the accuracy and gradient smoothness. This example uses a simple approximation; more sophisticated approximations would need to be derived based on specific requirements for accuracy and smoothness.  Careful consideration of numerical stability is vital for preventing overflow or underflow errors.

**Example 3:  Implementing a Rational Approximation of SS with a custom layer for greater control:**

```python
import tensorflow as tf
from tensorflow import keras

class RationalSSLayer(keras.layers.Layer):
    def __init__(self, a=1.0, b=1.0, **kwargs): # Adjustable parameters for the rational function.
        super(RationalSSLayer, self).__init__(**kwargs)
        self.a = a
        self.b = b

    def call(self, inputs):
        numerator = self.a * inputs
        denominator = 1 + self.b * tf.abs(inputs)
        return numerator / denominator

    def get_config(self):
        config = super(RationalSSLayer, self).get_config()
        config.update({'a': self.a, 'b': self.b})
        return config


model = keras.Sequential([
    keras.layers.Dense(64, input_shape=(784,)),
    RationalSSLayer(a=1.5, b=0.8), #Example parameters - tuning required.
    keras.layers.Dense(10, activation='softmax')
])

model.compile(...) #Rest of model compilation
```

This example uses a rational function as the approximation for the smooth sigmoid. The parameters `a` and `b` control the shape and thus allow for more fine-grained tuning of the activation function's characteristics. The use of a custom layer gives greater control over the implementation, and the `get_config` method ensures model persistence.  Careful selection of `a` and `b` is critical for ensuring numerical stability and preventing division by zero errors.


**3. Resource Recommendations:**

*  "Deep Learning" by Goodfellow, Bengio, and Courville
*  "Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow" by Aurélien Géron
*  Relevant research papers on activation functions and their variants.  Look for works comparing PSS and SS to other activation functions in specific contexts.  Pay close attention to papers that discuss the numerical stability and performance of custom activation functions within deep learning frameworks.



Remember that the choice of alpha in PSS and the specific approximation used for SS will significantly impact the performance of your model.  Thorough experimentation and validation are necessary to find the optimal parameters for your specific task and dataset.  Furthermore, always consider the potential for numerical instability when implementing custom activation functions, especially those involving non-linear operations or potential division by zero.  Rigorous testing and validation are essential for ensuring the reliability and accuracy of your models.
