---
title: "Why am I getting an AttributeError when plotting a Keras model?"
date: "2025-01-30"
id: "why-am-i-getting-an-attributeerror-when-plotting"
---
The likely cause for an `AttributeError` when attempting to plot a Keras model stems from the object you're trying to apply the plotting function to, which is often `keras.utils.plot_model`. This method expects a Keras model object as its first argument, not, for example, a compiled model returned from training, or data loaded for training. I've encountered this several times during my work on deep learning pipelines, especially when transitioning between model building and evaluation stages.

Specifically, the error arises when the `.plot_model()` method is called upon something that lacks the underlying structure and attributes needed for graph visualization. This method internally relies on the model object’s layers, connections, and tensor shapes to construct the graph; hence, it demands a representation of the Keras model itself. If you are getting this error, it strongly indicates that the first argument you passed to this function is not a Keras model object. Let's examine the details.

The typical workflow in Keras involves building a model, often with either the Sequential API or the Functional API, and *then* training it. The `.plot_model()` function should be used *after* the model is built, but *before* training. Once you have a Keras model object, you pass it to the plotting function. A frequent pitfall, however, is trying to plot a model after training, typically after calling `model.fit()`. The return value of `model.fit()` is a `History` object, not the model itself.

Consider, for instance, a scenario where a user mistakenly reassigns the model variable to the output of the training routine, and consequently passes the history object to `plot_model()` which causes the error. Similarly, if a model is loaded from a saved file, one has to take care that the loaded model object, not a filepath, gets to the plotting function. Furthermore, the object must also be an instance of a Keras `Model` object, and not a model building structure, which is often used in the Functional API.

Let's illustrate this with code examples and common scenarios.

**Example 1: Incorrect use after training**

```python
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense

# Define a simple sequential model
model = keras.Sequential([
    Dense(10, activation='relu', input_shape=(100,)),
    Dense(1, activation='sigmoid')
])

# Generate dummy data
import numpy as np
x_train = np.random.rand(100, 100)
y_train = np.random.randint(0, 2, 100)

# Compile and train the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=10, verbose=0) # Return value re-assigned

# Attempt to plot AFTER TRAINING (INCORRECT)
try:
    keras.utils.plot_model(history, show_shapes=True, to_file='model_plot_incorrect.png')
except AttributeError as e:
    print(f"Caught expected AttributeError: {e}")

# Correct plotting is done BEFORE training
keras.utils.plot_model(model, show_shapes=True, to_file='model_plot_correct.png')

print("Correct plotting after model creation, before training - image saved to file 'model_plot_correct.png'")
```

In this example, the first attempt to plot the model passes the `history` object resulting from the `fit` function to `plot_model` instead of the model itself, generating an `AttributeError`. The second call, which plots the model *before* training, works correctly. This example highlights the fact that one must retain the reference to the model *as constructed*, and not the `History` object returned by training.

**Example 2: Incorrect usage with model definition in the functional API**

```python
from tensorflow import keras
from keras.layers import Input, Dense
import numpy as np

# Define model using Functional API
inputs = Input(shape=(50,))
x = Dense(20, activation='relu')(inputs)
outputs = Dense(1, activation='sigmoid')(x)

# Model not yet created

try:
    keras.utils.plot_model(inputs, show_shapes=True, to_file='model_plot_functional_incorrect.png')
except AttributeError as e:
    print(f"Caught expected AttributeError: {e}")


# Build the model object
model_functional = keras.Model(inputs=inputs, outputs=outputs)


# Plotting the model
keras.utils.plot_model(model_functional, show_shapes=True, to_file='model_plot_functional_correct.png')

print("Correct plotting of the built functional API model - image saved to file 'model_plot_functional_correct.png'")
```

Here, we define the model using the Functional API. Initially, the code attempts to plot the `inputs` tensor object before a complete model object is constructed, resulting in an error. The second call, which plots the *constructed* `model_functional`, works as expected, correctly creating a visualization of the network architecture. It is important to remember that `plot_model()` expects the final `Model` object, not a tensor or a definition for the model.

**Example 3: Incorrect object passed from file load**

```python
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
import numpy as np

# Create and save a model
model_to_save = keras.Sequential([
    Dense(10, activation='relu', input_shape=(100,)),
    Dense(1, activation='sigmoid')
])
model_to_save.save("saved_model")

# Load the saved model
loaded_model = keras.models.load_model("saved_model")

try:
    keras.utils.plot_model("saved_model", show_shapes=True, to_file='model_plot_load_incorrect.png')
except AttributeError as e:
    print(f"Caught expected AttributeError: {e}")


# Correct plotting after loading the model
keras.utils.plot_model(loaded_model, show_shapes=True, to_file='model_plot_load_correct.png')
print("Correct plotting of the loaded model - image saved to 'model_plot_load_correct.png'")

```

In this example, after saving and loading a model, there's an attempt to plot the model, using the saved file path string which is incorrect, and this again gives an `AttributeError`. The correct call plots the loaded model, not the path where the model was loaded from. This further emphasizes that the Keras `Model` object, and not a filename, must be passed.

In summary, the key to avoiding the `AttributeError` is ensuring that you are passing a constructed Keras model object to the `keras.utils.plot_model` function. The model object must be an instance of `keras.Model`, and not data loaded to train the model, a returned training object, or a model building structure.

To further refine your workflow and prevent these errors in the future, I recommend these resources. First, delve deeply into the official Keras documentation, especially the sections on model construction, training, and visualization utilities. Second, explore practical examples and tutorials on building Keras models using both the Sequential and Functional APIs, as that will give practical hands-on experience on building models and avoid passing the incorrect object. Finally, familiarize yourself with the `tf.keras.Model` class and its attributes, which is required for correct usage of model plotting. This thorough understanding of Keras objects and their lifecycle will help prevent this error and enable more confident deep learning workflow.
