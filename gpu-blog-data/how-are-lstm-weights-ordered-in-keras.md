---
title: "How are LSTM weights ordered in Keras?"
date: "2025-01-30"
id: "how-are-lstm-weights-ordered-in-keras"
---
LSTM weights in Keras are not stored as a single, monolithic block but rather as a structured collection of matrices and vectors, each designed to perform specific transformations within the recurrent cell. Understanding this structure is critical for debugging, weight initialization, or custom manipulation scenarios. I've encountered numerous situations where a misunderstanding of this order led to incorrect layer configurations, necessitating a deeper dive into Keras source code and experimentation.

Fundamentally, Keras, utilizing TensorFlow as its backend, organizes LSTM weights based on the input, forget, cell, and output gates. Each gate has its dedicated weight matrices for the input and recurrent connections, and corresponding bias vectors. Furthermore, the internal state transition involves cell state updates governed by an additional set of parameters. The order is not explicitly documented in the form of a single comprehensive table, but it can be deduced by inspecting the code and observing the dimensions of the weight tensors after layer creation.

The input gate determines which new information will be added to the cell state. It has both an input-to-gate weight matrix (W_i), a recurrent-to-gate matrix (U_i) and a bias vector (b_i). The forget gate controls which parts of the old cell state are discarded. It has similar structures with weight matrices (W_f, U_f) and a bias vector (b_f). The cell candidate or 'cell gate' controls how the cell state gets updated. It too has weight matrices (W_c, U_c) and a bias vector (b_c). Lastly, the output gate determines which parts of the cell state will be output, with associated weight matrices (W_o, U_o) and a bias vector (b_o). These parameters, taken together, govern the state transition and output of the LSTM cell at each time step. These are ordered sequentially in Keras's internal representations.

Let me illustrate this with code examples, working with a basic LSTM layer.

**Example 1: Understanding the initial weights.**

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Define an LSTM layer
lstm_layer = keras.layers.LSTM(units=64, input_shape=(None, 10), return_sequences=True)
# Create a random input tensor to init the weights
dummy_input = tf.random.normal((1, 5, 10))
_ = lstm_layer(dummy_input)  # Force weight initialization

# Access the weights
weights = lstm_layer.get_weights()

print(f"Number of weight tensors: {len(weights)}")

# Print the shapes of the weight tensors
for i, w in enumerate(weights):
    print(f"Weight tensor {i} shape: {w.shape}")
```

The output of this code will demonstrate that the layer initially has three weight tensors, even with biases considered as separate tensors in the first versions of Keras but unified into single tensors in recent ones.. The first tensor (index 0) is composed of input weights, concatenated across all four gates (W_i, W_f, W_c, W_o). The next tensor (index 1) contains the recurrent weights also concatenated across the four gates (U_i, U_f, U_c, U_o), and the final tensor (index 2) holds the bias vectors which are also concatenated across the gates (b_i, b_f, b_c, b_o). The order within these concatenated matrices and vectors is consistently input, forget, cell, and output gate. Therefore, the order within each of those tensors follow the IFCO convention. Specifically, for an LSTM with an input size of 10 and 64 units: weight tensor 0 will be of the shape (10, 256) (10 being the input dimension and 256 is 4x the number of units); weight tensor 1 will be of shape (64, 256); and weight tensor 2 is of shape (256,).

**Example 2: Accessing individual gate weights.**

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Define an LSTM layer
lstm_layer = keras.layers.LSTM(units=64, input_shape=(None, 10), return_sequences=True)
dummy_input = tf.random.normal((1, 5, 10))
_ = lstm_layer(dummy_input)  # Force weight initialization

# Access all weights
weights = lstm_layer.get_weights()

# Define the number of units and input dimension
units = 64
input_dim = 10

# Extract input weights
W = weights[0]
W_i = W[:, :units]
W_f = W[:, units:2*units]
W_c = W[:, 2*units:3*units]
W_o = W[:, 3*units:]

# Extract recurrent weights
U = weights[1]
U_i = U[:, :units]
U_f = U[:, units:2*units]
U_c = U[:, 2*units:3*units]
U_o = U[:, 3*units:]

# Extract bias weights
b = weights[2]
b_i = b[:units]
b_f = b[units:2*units]
b_c = b[2*units:3*units]
b_o = b[3*units:]

print(f"Input weight matrix for input gate shape:{W_i.shape}")
print(f"Recurrent weight matrix for output gate shape:{U_o.shape}")
print(f"Bias vector for forget gate shape:{b_f.shape}")

```

This example shows how to extract the separate matrices and vectors associated with each gate by utilizing the previously determined order, applying the learned pattern, slicing the input weight matrix, recurrent weight matrix, and bias vector appropriately. Notice that after slicing, we obtain the expected shapes for the respective matrices and vectors. Using this method, one can inspect the actual values of the weights and modify them if required. This is useful when training from scratch or when manipulating pre-trained weights.

**Example 3: Modifying the biases.**

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Define an LSTM layer
lstm_layer = keras.layers.LSTM(units=64, input_shape=(None, 10), return_sequences=True)
dummy_input = tf.random.normal((1, 5, 10))
_ = lstm_layer(dummy_input) # Force weight initialization
# Access weights
weights = lstm_layer.get_weights()

units = 64
# Separate the biases
b = weights[2]
b_i = b[:units]
b_f = b[units:2*units]
b_c = b[2*units:3*units]
b_o = b[3*units:]


# Modify the forget gate biases, to encourage initial forgetting
b_f = np.ones(units) * 5.0

# Reconstruct the bias vector
new_b = np.concatenate([b_i, b_f, b_c, b_o])

# Replace the bias in the original weights
weights[2] = new_b

# Set the new weights into the layer
lstm_layer.set_weights(weights)


new_weights = lstm_layer.get_weights()

new_b_values = new_weights[2]
print(f"First 6 forget gate bias after modification: {new_b_values[units:units+6]}")
```

Here, the focus is on manipulating a portion of the bias vector. The forget gate biases are set to a high positive value.  This demonstrates how a very basic form of custom initialization could be realized by manipulating specific weights, by replacing the old tensor with a new tensor that contains the modified bias. This particular operation, setting high initial forget gate biases, is a common technique to promote longer-term memory capabilities in LSTMs during the initial training phase, or when performing fine-tuning. Examining the first few bias values after setting, we can verify that the forget gate biases have been modified as intended.

In summary, Keras LSTM weight ordering follows a consistent input, forget, cell, output (IFCO) gate pattern, both within the concatenated weight matrices and the bias vectors. The full weights are stored in a list of three tensors. Understanding the ordering permits fine control of the LSTM parameters. The approach used involves extracting the weights and slicing along the appropriate dimensions, based on the number of LSTM units, input dimensions, and the order of operations within the LSTM cell.

For additional information, resources on the mathematical formulations of LSTMs are beneficial. Sources covering recurrent neural networks often provide diagrams and equations that complement the practical aspects explored above. Research publications detailing specific initialization strategies or memory-control techniques in LSTMs can also prove valuable when delving into more advanced usage scenarios. The Keras documentation, although not explicitly defining the ordering, offers descriptions of the LSTM layer's parameters, which also assists in building a solid understanding.  Furthermore, looking into the TensorFlow source code and implementation of the LSTM cells could solidify the interpretation of the order.
