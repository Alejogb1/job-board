---
title: "How can time series be forecasted using LSTMs implemented from scratch with NumPy?"
date: "2024-12-23"
id: "how-can-time-series-be-forecasted-using-lstms-implemented-from-scratch-with-numpy"
---

 Time series forecasting with LSTMs implemented from scratch using NumPy is indeed a challenge, but certainly not insurmountable. I've had my share of experiences with this, notably during a project involving predictive maintenance for a fleet of industrial robots where off-the-shelf solutions fell short due to the specific data characteristics. It pushed me to delve deep into the mechanics of LSTMs and their numerical underpinnings.

First off, building an LSTM from the ground up with NumPy forces you to confront the internal workings of the model, something often abstracted away by higher-level libraries. This understanding is crucial, particularly when you need to debug issues or adapt the model to novel situations. We're essentially going to replicate the functionality of an LSTM layer through careful matrix operations and manual backpropagation – it’s tedious, but enlightening.

Let’s decompose the problem. At the heart of an LSTM lies its ability to maintain a state, which allows it to capture temporal dependencies in sequential data. An LSTM cell comprises a forget gate, input gate, output gate, and a cell state. These gates control the flow of information into and out of the cell, enabling the network to retain relevant past information while discarding irrelevant noise. We won’t discuss the math in intricate detail, but it's vital to understand that each gate is essentially a sigmoid activation applied to a linear combination of the input and the previous hidden state, and that their output is used to manipulate the cell state and the output of the cell. We need to define each of those components in our NumPy implementation.

Now, for the practical part. Let's start with the core mathematical operations that each layer will perform, specifically focusing on the forward propagation of a basic LSTM layer. Before presenting code, it is important to stress that we are working with a simplified version, lacking various optimizations and complexities that are found in optimized libraries. This implementation should help you see the bare bones of what's happening under the hood.

**Snippet 1: Forward Propagation of an LSTM Layer**

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def lstm_forward(x, h_prev, c_prev, Wx, Wh, b):
    """
    Performs forward pass of a single LSTM cell.

    Args:
        x: Input at current timestep (vector).
        h_prev: Hidden state from previous timestep (vector).
        c_prev: Cell state from previous timestep (vector).
        Wx: Input weight matrix.
        Wh: Hidden weight matrix.
        b: Bias vector.

    Returns:
        h_next: Next hidden state (vector).
        c_next: Next cell state (vector).
        gates: A tuple containing all the gates for backpropagation purposes.
    """
    z = np.dot(x, Wx) + np.dot(h_prev, Wh) + b
    
    # Split z into the respective gates: forget, input, candidate cell state, and output
    f = sigmoid(z[:z.shape[0]//4])
    i = sigmoid(z[z.shape[0]//4:z.shape[0]//2])
    c_tilde = tanh(z[z.shape[0]//2:3*z.shape[0]//4])
    o = sigmoid(z[3*z.shape[0]//4:])


    c_next = f * c_prev + i * c_tilde
    h_next = o * tanh(c_next)
    
    gates = (f, i, c_tilde, o, c_next, h_next)
    
    return h_next, c_next, gates
```

In this function, we compute the activations for all gates and the cell state. Note how we are splitting the combined linear transformation result (`z`) into the respective gates. Each gate, through the sigmoid function, outputs a value between 0 and 1 that regulates the flow of information. The `tanh` function provides non-linearity to the candidate cell state. The function returns not only the new hidden and cell state but also the activations for the backpropagation step.

Now that we have the fundamental building block, we can construct an LSTM layer that processes an entire sequence. The following snippet showcases how we can execute a series of time steps using our function, assuming you have your input and initial hidden/cell state already prepped.

**Snippet 2: Processing a Time Series Sequence through an LSTM layer**

```python
def lstm_layer_forward(X, h0, c0, Wx, Wh, b):
    """
    Performs forward pass of an entire LSTM layer.

    Args:
        X: Input sequence (time steps x input features).
        h0: Initial hidden state (vector).
        c0: Initial cell state (vector).
        Wx: Input weight matrix.
        Wh: Hidden weight matrix.
        b: Bias vector.

    Returns:
        H: Sequence of hidden states (time steps x hidden features).
        C: Sequence of cell states (time steps x hidden features).
        gates_list: list of tuple gates for all the timesteps (for backpropagation)
    """
    T = X.shape[0]  # Time steps
    H = np.zeros((T, h0.shape[0]))
    C = np.zeros((T, c0.shape[0]))
    gates_list = []

    h_prev = h0
    c_prev = c0

    for t in range(T):
        h_next, c_next, gates = lstm_forward(X[t], h_prev, c_prev, Wx, Wh, b)
        H[t] = h_next
        C[t] = c_next
        gates_list.append(gates)
        h_prev = h_next
        c_prev = c_next
    
    return H, C, gates_list
```
Here, we iterate over each time step, passing the input and previous hidden state to the `lstm_forward` function. The hidden states and cell states for each time step are recorded, and returned. Importantly, all of the gate activations are stored as a list, which will be necessary for backpropagation.

Finally, to generate a forecasted value, you’d typically feed the output of the LSTM layer to a fully connected layer followed by a suitable activation (often linear or sigmoid depending on the forecast task). The following snippet exemplifies this last stage, where we are not implementing the backpropagation for brevity, but only the final computation.

**Snippet 3: Fully Connected Layer output from an LSTM layer**

```python
def fully_connected_forward(H, W_fc, b_fc):
    """
    Performs forward pass of a fully connected layer.

    Args:
        H: Hidden states from LSTM (time steps x hidden features).
        W_fc: Weights of fully connected layer.
        b_fc: Biases of fully connected layer.

    Returns:
        Y_hat: Forecasted value (time steps x output features)
    """
    Y_hat = np.dot(H, W_fc) + b_fc
    return Y_hat
```
This demonstrates how the output from the LSTM layer `H` is connected to another layer, with `W_fc` and `b_fc` being the weights and biases of the new, fully-connected layer. The output `Y_hat` will be our final prediction for the next timestep.

Building the backpropagation for these steps is involved, and would go beyond the scope of this response. However, it is imperative to realize the process requires the derivation of the gradients of all operations with respect to the corresponding weight matrices and biases, following the chain rule and through the sequence. It would be necessary to also implement a function similar to the `lstm_forward` function, but for the backpropagation step, to compute the gradients. Once all the gradients are computed, the weights can be updated using a gradient descent algorithm (or a similar optimization algorithm), and the training process can start.

A few critical resources for further understanding are as follows:
* "Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville. This is a comprehensive text covering the fundamental concepts of deep learning, including recurrent neural networks and LSTMs.
* Christopher Olah’s blog posts, particularly "Understanding LSTM Networks," which provides an intuitive explanation of how LSTMs work.
* The original LSTM paper: “Long Short-Term Memory” by Hochreiter and Schmidhuber (1997). Reading the seminal work provides a deeper appreciation for the original concept.

In conclusion, crafting an LSTM from scratch with NumPy is more about solidifying your fundamental understanding of its mathematical workings than achieving state-of-the-art performance. While these examples are simplified, they provide a good starting point. The real value lies in understanding each individual operation, gradient, and how they all contribute to the final prediction. If you work on each of the different code snippets presented here, and add the backpropagation logic, you will find it extremely useful when debugging model issues later on, even if you transition to using higher level libraries like TensorFlow or PyTorch. My advice would be to practice and play around with the functions presented here, and start building your model from the bottom up, one step at a time. This will provide you with a much deeper understanding than simply working with pre-built libraries.
