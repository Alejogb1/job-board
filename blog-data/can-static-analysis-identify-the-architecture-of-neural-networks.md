---
title: "Can static analysis identify the architecture of neural networks?"
date: "2024-12-23"
id: "can-static-analysis-identify-the-architecture-of-neural-networks"
---

Okay, let's talk about this. I’ve actually spent a fair bit of time trying to extract architecture details from neural network code without running it, and it's a problem with some fascinating nuances. The simple answer is: it depends, but generally, yes, static analysis *can* identify key aspects of a neural network’s architecture, although it's not always trivial and has its limitations.

When we say ‘architecture’, we're usually referring to things like the number and type of layers (convolutional, recurrent, fully connected, etc.), the number of neurons or filters in each layer, the activation functions used, and sometimes, even things like dropout rates or specific connections between layers, if they’re complex. Purely based on examining code – the static analysis part – we're essentially reverse engineering these details from the code itself, without executing it.

The ability to do this accurately hinges significantly on the coding style and the level of abstraction used within the framework being leveraged (like TensorFlow, PyTorch, Keras, etc.). If the architecture is defined using high-level, declarative APIs, static analysis is significantly more straightforward. Imagine a situation where the code clearly states `layers.Conv2D(32, kernel_size=(3, 3), activation='relu')`. A static analyzer can easily detect a 2d convolutional layer with 32 filters, kernel size of 3x3, and a relu activation. If, however, someone’s hand-rolling everything at a low level, or obscuring layer definitions within complex functions, it's a different ball game entirely and the task becomes much harder.

The fundamental approach revolves around parsing the source code – either directly if it’s Python or some other scripting language – or parsing the intermediate representation (IR) if it’s compiled. We look for patterns in the way layers are instantiated and connected. This often involves graph traversal and data flow analysis within the codebase itself. We're effectively trying to reconstruct the neural network definition based on how it’s programmed, which means understanding the API calls related to building networks within the frameworks being used.

I recall a project a few years ago where we were tasked with creating a security scanner for machine learning models. One component was to statically analyze a directory containing Python training scripts. We weren’t interested in the data, we needed to know the shape and makeup of the neural network. We found that while basic sequential models were fairly easy to parse using custom AST traversals, more complicated model architectures that employed subclassing and function calls to create layers introduced substantial complexity. The code wasn’t *bad,* it was simply that the architectural definitions weren’t laid out explicitly in linear form, making analysis challenging.

Here are three practical scenarios that highlight these points, with accompanying code examples in Python:

**Example 1: Simple Sequential Model (Easy)**

Here is the code:
```python
import tensorflow as tf
from tensorflow.keras import layers

def create_simple_model():
    model = tf.keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(784,)),
        layers.Dense(10, activation='softmax')
    ])
    return model

if __name__ == "__main__":
    model = create_simple_model()
    print(model.summary()) # just to show that this works and the model is buildable
```

In this case, the static analysis is straightforward. The analysis would identify a `Sequential` model in Tensorflow/Keras. The layers within are easily identifiable as a `Dense` layer with 128 units and relu activation, followed by another dense layer with 10 output units and softmax activation. The `input_shape=(784,)` argument can reveal the expected input shape of the model. An analyzer could simply read the code directly, parse the arguments given to those functions and reconstruct a structural description.

**Example 2: Model with Subclassing (More Complex)**

Consider this code:
```python
import tensorflow as tf
from tensorflow.keras import layers

class CustomModel(tf.keras.Model):
    def __init__(self, num_units_1, num_units_2):
        super(CustomModel, self).__init__()
        self.dense1 = layers.Dense(num_units_1, activation='relu')
        self.dense2 = layers.Dense(num_units_2, activation='sigmoid')

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

def create_custom_model():
    return CustomModel(256, 10)


if __name__ == "__main__":
    model = create_custom_model()
    print(model(tf.random.normal((1, 784))))# to demonstrate the model works, no real data here.
    print(model.summary())
```

This version uses class-based model creation, a common pattern in TensorFlow. The analyzer needs to go a level deeper. Instead of directly parsing a list of layer definitions, it must follow the logic in the `__init__` method, look at member variables and follow the `call` method to infer the data flow. The number of units and activation function are still discoverable, but now more information is contained in the class definition and not just at the instantiation stage. The analyzer would now need to understand that `self.dense1` and `self.dense2` are layers that have been defined in the init and how these layers are sequenced through the `call` method. This requires more sophisticated static analysis methods.

**Example 3: Model with Hidden Layer Construction (Harder)**

Here’s the trickiest example:
```python
import tensorflow as tf
from tensorflow.keras import layers

def build_hidden_layer(units, activation):
   return layers.Dense(units, activation=activation)

def create_complex_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    x = build_hidden_layer(64, 'relu')(inputs)
    x = layers.Dropout(0.2)(x)
    outputs = build_hidden_layer(10, 'softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

if __name__ == "__main__":
    model = create_complex_model((784,))
    print(model.summary())
```

In this code, a separate function `build_hidden_layer` encapsulates the creation of a `Dense` layer. The layers are not constructed directly in the main model definition. The analyzer would need to trace calls to `build_hidden_layer` and understand the arguments that are passed. The model architecture, though relatively simple, now requires understanding function call semantics and resolving which parameters are passed through the chain to the actual layer creation. This sort of indirection makes static analysis significantly more challenging. Dataflow analysis techniques combined with inter-procedural analysis are required.

So, what are the limitations? Well, dynamic behavior is completely opaque to static analysis. If the layer architecture depends on runtime variables, or if layers are created inside conditional loops based on user-provided data, the analyzer cannot possibly determine the full architecture. Furthermore, obfuscated code, or models that heavily leverage dynamically created tensors in framework-specific ways, can thwart static analysis.

For a deep dive, I would highly recommend studying the principles behind static program analysis, with a focus on abstract interpretation. Nielson, Nielson, and Hankin's *Principles of Program Analysis* is a great starting point. For techniques specific to TensorFlow and similar frameworks, look into the publications from researchers working on compilers for machine learning accelerators (e.g., look for papers on TensorFlow XLA or MLIR). You’ll find that these researchers have to solve related problems to optimize models, such as reasoning about the graph structure. Additionally, textbooks such as "Deep Learning with Python" by François Chollet provide detailed information about common model-building patterns that can guide building static analysis tools.

In summary, while static analysis can reveal considerable information about neural network architectures, it is far from perfect. It provides a valuable tool, especially for security and audit purposes, but it needs to be applied thoughtfully with an awareness of its inherent limitations. The level of success is directly proportional to the explicitness of the codebase and inversely proportional to the level of abstraction utilized in the framework itself.
