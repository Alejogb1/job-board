---
title: "How can I get pretty-printed output for TensorFlow variable types?"
date: "2025-01-30"
id: "how-can-i-get-pretty-printed-output-for-tensorflow"
---
TensorFlow's default output for variable types, particularly complex structures or tensors with many dimensions, can be cumbersome for debugging and analysis. I've frequently encountered this issue while building intricate models, often needing to delve into variable contents to understand training dynamics or identify unexpected data shapes. Achieving a pretty-printed, human-readable format necessitates moving beyond the basic `print()` function and leveraging TensorFlow's specific utilities and, when those fall short, employing Pythonic formatting tools.

The core problem lies in how TensorFlow manages internal data representation. Its variables, even seemingly simple ones, are objects containing not just the data itself but also associated metadata regarding the tensor's shape, data type, and device placement. The standard `print()` method invokes the `__str__` or `__repr__` methods of these objects, which are geared more towards internal representation rather than human readability. They often present long, nested lists or uninformative strings. To obtain a clearer picture, we must access the numerical data using `numpy()` and then strategically format it. We also must pay attention to the object type. `tf.Variable` objects have properties that allow us to inspect specific properties.

First, if we're dealing with a `tf.Variable` itself, we can start with retrieving basic descriptive information and then use Numpy to extract the numerical data for more flexible formatting:

```python
import tensorflow as tf
import numpy as np

# Create a sample variable
my_variable = tf.Variable(tf.random.normal([2, 3, 4]), name="my_var")

# 1. Print basic metadata
print(f"Variable name: {my_variable.name}")
print(f"Variable shape: {my_variable.shape}")
print(f"Variable dtype: {my_variable.dtype}")

# 2. Extract numerical data
numpy_array = my_variable.numpy()

# 3. Simple print of the numpy data
print("\nNumpy Array:\n", numpy_array)

# 4. Formatting numpy data (e.g., using numpy printing options)
np.set_printoptions(precision=3, suppress=True) # Configure np formatting
print("\nFormatted Numpy Array:\n", numpy_array)
```

In this first example, I begin by creating a 3-dimensional tensor variable. I then access the variable’s `name`, `shape`, and `dtype`. This provides a high-level understanding of what the variable represents. After this basic introspection, I extract the numerical data using `.numpy()` and then show two ways to print it. The initial print, while functional, can be verbose, especially with higher dimensions. In the subsequent `np.set_printoptions` call, I demonstrate a simple formatting option that sets precision to 3 decimal places and suppresses scientific notation for a cleaner visual presentation. Using numpy options for formatting is key to controlling the visual output of the actual numerical data.

The key insight here is that TensorFlow objects often require extractions of their underlying numpy objects to present data in user-friendly ways. The printing of the properties of the variable are provided directly by Tensorflow whereas the numeric content requires conversion to a Numpy object.

Next, consider the scenario where you're dealing with a complex nested tensor – common in sequence-to-sequence models or those using recurrent layers. Simply converting the whole thing at once might not be illustrative. Instead, I advocate for recursive iteration:

```python
import tensorflow as tf
import numpy as np

def recursive_print(tensor_obj, level=0):
    indent = "  " * level
    if isinstance(tensor_obj, tf.Tensor):
        print(f"{indent}Tensor shape: {tensor_obj.shape}, dtype: {tensor_obj.dtype}")
        if tf.rank(tensor_obj) <= 3:
          np_array = tensor_obj.numpy()
          print(f"{indent}Data (if low rank):\n{np_array}")
        else:
          print(f"{indent}Data (too many dimensions to print)")


    elif isinstance(tensor_obj, list):
        print(f"{indent}List (length {len(tensor_obj)})")
        for i, item in enumerate(tensor_obj):
            print(f"{indent}  Item {i}:")
            recursive_print(item, level + 2)
    elif isinstance(tensor_obj, tuple):
        print(f"{indent}Tuple (length {len(tensor_obj)})")
        for i, item in enumerate(tensor_obj):
           print(f"{indent}  Item {i}:")
           recursive_print(item, level + 2)

    elif isinstance(tensor_obj, dict):
        print(f"{indent}Dictionary (keys: {list(tensor_obj.keys())})")
        for key, value in tensor_obj.items():
           print(f"{indent} Key: {key}")
           recursive_print(value, level + 2)

    else:
      print(f"{indent}Non-Tensor Object: {type(tensor_obj)}")

# Example usage:
nested_tensor = [
    tf.random.normal((2,2)),
    tf.constant([[1,2],[3,4]]),
    (tf.random.uniform((3,3)),tf.random.uniform((2,2))),
    {"first_key": tf.random.normal((2,2,2)), "second_key": [tf.constant(5)]}
]
print("Nested Structure:\n")
recursive_print(nested_tensor)
```

Here, `recursive_print` handles lists, tuples, dictionaries, and tensors. If the tensor has low enough rank, I also print its contents after extracting them with `numpy()`. For tensors with rank above three, I note that they have "too many dimensions to print", since I found them typically to be excessively verbose in debugging. This recursive approach allows detailed inspections into nested data structures. During model development, I found this invaluable for dissecting complex output dictionaries or feature maps. It's important to handle common container types like lists, dictionaries, and tuples, as they often package TensorFlow objects.

Finally, consider a custom TensorFlow object within a training loop. Here we will define our own class, subclassing `tf.keras.Model`, and use formatting within its methods:

```python
import tensorflow as tf
import numpy as np

class MyCustomModel(tf.keras.Model):
  def __init__(self, num_units, output_dim):
      super(MyCustomModel, self).__init__()
      self.dense1 = tf.keras.layers.Dense(num_units, activation='relu')
      self.dense2 = tf.keras.layers.Dense(output_dim)

  def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

  def custom_print_info(self, input_tensor):
        print("Printing model details...\n")
        print(f"Input Shape:{input_tensor.shape}")
        hidden_layer_output = self.dense1(input_tensor)
        print(f"Hidden Layer Output Shape:{hidden_layer_output.shape}")
        output = self.dense2(hidden_layer_output)
        print(f"Final Output Shape:{output.shape}")

        print("\nHidden Layer Weights:\n")
        print(self.dense1.weights[0].numpy()) # print the weight matrix
        print("\nOutput Layer Weights:\n")
        print(self.dense2.weights[0].numpy())

# Example usage:
model = MyCustomModel(num_units=64, output_dim=10)
sample_input = tf.random.normal((1,100))

model.custom_print_info(sample_input)
```

Here, I defined a custom Keras model and implemented a dedicated `custom_print_info` method for printing useful variable information. It prints not only input and output shapes through the forward pass but also the weight matrices of each layer. This granular level of control demonstrates that formatting printouts can also be a part of class definitions. For custom model objects, creating specific methods for formatted printing is a practice I have found to simplify debugging and monitoring complex layers.

For further exploration and more complex formatting scenarios, I recommend consulting the TensorFlow documentation on tensor manipulation and inspection functions and Numpy's documentation on printing options and `ndarray` properties. Additionally, online books covering advanced Python data visualization and formatting techniques can offer alternative strategies for customized printing needs, particularly when dealing with very large tensors. Textbooks that detail specific debugging tools and techniques, such as those for machine learning model interpretation, are also good resources for handling the visual output of high-dimensional data. Finally, experimenting with Jupyter Notebook widgets can lead to interactive and enhanced representations of variable contents.
