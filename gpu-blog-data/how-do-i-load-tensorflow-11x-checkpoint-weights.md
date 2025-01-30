---
title: "How do I load TensorFlow 1.1x checkpoint weights into a TF2.2 LSTM layer without discrepancies?"
date: "2025-01-30"
id: "how-do-i-load-tensorflow-11x-checkpoint-weights"
---
Migrating from TensorFlow 1.x to 2.x, specifically with respect to loading model weights, often introduces subtle discrepancies due to fundamental changes in how variables are managed and named. When dealing with recurrent layers like LSTMs, these discrepancies become particularly evident. The challenge lies not just in the format differences of the checkpoint files, but also in how TensorFlow 2.x automatically infers the variable shapes, which may not perfectly align with the implicit structure in TensorFlow 1.x. Directly loading the weights without careful manipulation will likely lead to unexpected errors or, more insidiously, incorrect model behavior. Here’s how I’ve approached this in my past projects.

The crux of the issue is that TensorFlow 1.x checkpoint files store weights under specific variable names. These names often include the original layer's scope or a generated identifier, like `lstm_cell/kernel`, `lstm_cell/bias`, etc. In TensorFlow 2.x, while the logical structure of an LSTM remains similar, these variable names are typically different and are not predictable. Moreover, TF2.x uses `tf.keras` which uses a model building paradigm that differs substantially from the graph definition approach of TF1.x. Simply loading the checkpoint weights into a TF2.x LSTM layer without considering these naming differences and architecture differences will fail. The key is identifying the correctly corresponding tensors in the TF1.x checkpoint and transferring those numerical values to newly instantiated TF2.x layers with their distinct naming conventions.

First, we must read the TF1.x checkpoint and extract the desired weight tensors. I usually employ `tf.train.NewCheckpointReader` for this task in a TF1.x environment to inspect the variable names contained in the checkpoint, although we will not be using this directly within our TF2.x context. This inspection is crucial to map the old variable names to what our newly instantiated TF2.x LSTM layer expects. This requires a separate inspection script. Once we've mapped the TF1.x variable names to TF2.x's, we can proceed by manually loading these mapped weight values. This manual loading approach is necessary since TF2.x’s default mechanism for loading weights assumes matching variable names.

Here's an approach I have utilized with success, and broken down in a step-by-step manner. Note that TF2.x is used in the following examples.

**Example 1: Inspecting TF1.x checkpoint using a TF1.x environment.**

The first step is *not* part of the weight loading to be performed in TF2.x. But is a pre-requisite to our transfer process. This example assumes you have access to a TF1.x environment and the checkpoint path of your TF1.x saved model. The key here is to understand the scope and variable names so we can then find the matching layers in TF2.x. This step helps with manual mapping.

```python
# This code should be run in a TF1.x environment

import tensorflow as tf
checkpoint_path = "path/to/your/tf1_checkpoint" # Replace with the actual path

reader = tf.train.NewCheckpointReader(checkpoint_path)
var_map = reader.get_variable_to_shape_map()

for var_name in sorted(var_map):
    print(var_name, var_map[var_name])

# Example Output (names will vary based on the TF1.x architecture used):
# lstm/basic_lstm_cell/bias [400]
# lstm/basic_lstm_cell/kernel [100, 400]
# output_layer/bias [10]
# output_layer/kernel [20, 10]
```

This script uses `tf.train.NewCheckpointReader` to load the checkpoint file and then iterates over the variable names contained within. I've found this step vital because the naming can be unpredictable in TF1.x, and we need these exact names to extract the numerical weights. The printed variable names and shapes (like `lstm/basic_lstm_cell/bias` and `lstm/basic_lstm_cell/kernel`) will be referenced later when loading the weights in TF2.x.

**Example 2: Loading the weights into a TF2.x LSTM layer**

This code example directly addresses loading TF1.x LSTM weights into a TF2.x environment. We assume we know the correct TF1.x checkpoint paths and variable names from the inspection step. For clarity, I will map only a single LSTM cell to avoid excess verbosity. I assume we identified that TF1.x's `lstm/basic_lstm_cell/kernel` needs to be loaded into the kernel of our TF2.x layer, and `lstm/basic_lstm_cell/bias` needs to be loaded into the bias.

```python
# This code should be run in a TF2.x environment

import tensorflow as tf
import numpy as np

checkpoint_path = "path/to/your/tf1_checkpoint" # Replace with the actual path

# Define a TF2.x LSTM layer. Ensure the unit size match
lstm_units = 100
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(lstm_units, input_shape=(None, 20))  # Example input shape
])


# Load the TF1.x checkpoint (using numpy for direct value read)
# This part uses the information extracted from the TF1.x inspection code
reader_tf2 = tf.train.load_checkpoint(checkpoint_path)
tf1_lstm_kernel_name = 'lstm/basic_lstm_cell/kernel' # TF1.x name of kernel
tf1_lstm_bias_name = 'lstm/basic_lstm_cell/bias' # TF1.x name of bias

tf1_kernel = reader_tf2.get_tensor(tf1_lstm_kernel_name)
tf1_bias   = reader_tf2.get_tensor(tf1_lstm_bias_name)


# Manually load the weights into the corresponding TF2.x LSTM layer's variables
tf2_kernel = model.layers[0].kernel
tf2_recurrent_kernel = model.layers[0].recurrent_kernel
tf2_bias = model.layers[0].bias

# TF2.x uses concatenated kernels and biases (input and recurrent). These need to be split for TF1
input_size = model.layers[0].input_shape[-1]
tf2_kernel.assign(np.concatenate(np.split(tf1_kernel, 4, axis=1)[:2], axis = 0)) # split kernel for input (i and f gates)
tf2_recurrent_kernel.assign(np.concatenate(np.split(tf1_kernel, 4, axis=1)[2:], axis=0)) # split kernel for recurrent (c and o gates)
tf2_bias.assign(tf1_bias) # bias is directly mapped

# Verify weights are loaded (optional)
print("TF2 Kernel shape:", tf2_kernel.shape)
print("TF2 Recurrent Kernel shape:", tf2_recurrent_kernel.shape)
print("TF2 Bias shape:", tf2_bias.shape)

```

In this example, I explicitly map TF1.x's `lstm/basic_lstm_cell/kernel` and `lstm/basic_lstm_cell/bias` to their corresponding components of the instantiated `tf.keras.layers.LSTM` in TF2.x. *Crucially*, this assumes you have already identified the correct correspondence during the TF1.x inspection phase.  TF2's LSTM kernels are concatenated and must be split as they store both the kernel and recurrent kernel in one tensor. The logic to split into the correct sections is included. I use `.assign()` here to directly overwrite the variables within the `tf.keras.layers.LSTM` object. This method bypasses TF2.x's standard initialization procedures, allowing us to inject the pre-trained weights.  We also use the checkpoint `reader_tf2` in TF2, which is different than the reader used in TF1.

**Example 3: Adding more error handling and flexibility**

This example demonstrates a generalized method, including error handling and flexibility, to automatically match the TF1.x checkpoint weights to a custom TF2.x LSTM layer. While the code becomes more complex, it is more practical for more complicated architectures. This section adds more robust exception handling to prevent unexpected application crashes. This would handle multiple LSTM layers.

```python
# This code should be run in a TF2.x environment

import tensorflow as tf
import numpy as np

def load_tf1_lstm_weights(tf1_checkpoint_path, tf2_model):
    """Loads TF1.x LSTM weights into a TF2.x Keras model.
    Args:
        tf1_checkpoint_path: Path to the TF1.x checkpoint.
        tf2_model: A Keras model containing LSTM layers.
    """
    try:
      reader_tf2 = tf.train.load_checkpoint(tf1_checkpoint_path)
    except tf.errors.NotFoundError as e:
       print("Error: TF1.x checkpoint not found", str(e))
       return

    for idx, layer in enumerate(tf2_model.layers):
      if isinstance(layer, tf.keras.layers.LSTM):
        try:
          # Assume TF1.x naming convention
          tf1_kernel_name = f'lstm_{idx}/basic_lstm_cell/kernel'
          tf1_bias_name = f'lstm_{idx}/basic_lstm_cell/bias'
          tf1_kernel = reader_tf2.get_tensor(tf1_kernel_name)
          tf1_bias   = reader_tf2.get_tensor(tf1_bias_name)
          # Fetch weights
          tf2_kernel = layer.kernel
          tf2_recurrent_kernel = layer.recurrent_kernel
          tf2_bias = layer.bias
          # Split kernel and load values
          input_size = layer.input_shape[-1]
          tf2_kernel.assign(np.concatenate(np.split(tf1_kernel, 4, axis=1)[:2], axis = 0)) # split kernel for input (i and f gates)
          tf2_recurrent_kernel.assign(np.concatenate(np.split(tf1_kernel, 4, axis=1)[2:], axis=0)) # split kernel for recurrent (c and o gates)
          tf2_bias.assign(tf1_bias)
        except tf.errors.NotFoundError as e:
          print("Error: Could not load layer " + str(idx) + " weights, corresponding TF1.x name not found in checkpoint: " + str(e))
    print("Weights transfer completed")

# Create a TF2.x model with some LSTM layers
input_shape = (None, 20)
lstm_units1 = 100
lstm_units2 = 150

model_tf2 = tf.keras.Sequential([
    tf.keras.layers.LSTM(lstm_units1, input_shape=input_shape, return_sequences=True),
    tf.keras.layers.LSTM(lstm_units2, return_sequences=False),
])

tf1_checkpoint_path = "path/to/your/tf1_checkpoint" # Replace this path
load_tf1_lstm_weights(tf1_checkpoint_path, model_tf2)
# Now model_tf2 has weights loaded from the TF1.x checkpoint
```

In this example, I introduce a function (`load_tf1_lstm_weights`) that iterates through the layers of a provided Keras model. It checks if a layer is an LSTM and attempts to load the corresponding TF1.x weights based on the *assumed* naming convention like `lstm_0/basic_lstm_cell/kernel`. The error handling tries to prevent common issues by catching potential errors when reading tensors from the checkpoint, which may occur if the correct variables are not found. While I've added error handling, the key remains that we need to know the actual names of TF1.x's LSTM layers in advance to establish the proper mapping. The provided code can handle multiple LSTM layers.

The approach I’ve outlined requires careful inspection of the TF1.x checkpoint and a deep understanding of the structure of your TF1.x and TF2.x models.  This is a manual process and must be executed with care.

For further study, I would recommend the official TensorFlow documentation on saving and restoring checkpoints, specifically focusing on the differences between TF1.x and TF2.x formats. There are also several excellent tutorials on migrating TensorFlow models available. Understanding the variable storage mechanics in detail will allow one to debug and transfer weights more effectively. Furthermore, understanding the mechanics of the underlying LSTM cell helps in accurately splitting and mapping the weights of the recurrent and non-recurrent kernels.
