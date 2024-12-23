---
title: "What is the computational complexity of a feedforward neural network language model using word2vec?"
date: "2024-12-23"
id: "what-is-the-computational-complexity-of-a-feedforward-neural-network-language-model-using-word2vec"
---

Alright, let’s unpack this. I've spent a fair bit of time knee-deep in neural networks and natural language processing, including a particularly challenging project involving building a chatbot from the ground up a few years ago. That experience really hammered home the intricacies of computational complexity, particularly when dealing with models like the one you're describing. So, let’s look at the complexity of a feedforward neural network language model that uses word2vec embeddings as its input.

First off, we need to be precise about what we mean by “computational complexity.” In our case, we’re primarily concerned with how the runtime of the model grows as the size of the input, namely the vocabulary size and the sequence length, increases. We're generally looking at Big O notation here, which gives us an asymptotic upper bound on the growth rate. It doesn’t tell us the exact runtime, but how runtime changes in the worst-case scenario as our data gets larger.

The core elements that contribute to the complexity of this particular setup are the embedding lookup, the feedforward network layers, and the output calculation. Let’s break these down:

**1. Word2Vec Embedding Lookup:**

Word2vec, at its heart, is a pre-trained embedding matrix. Each word in your vocabulary is represented as a dense vector. When we feed a sequence of words into the network, the first operation involves looking up the embedding vector for each word. This is typically done with a matrix lookup. Given that the lookup is performed on an existing matrix and that there is no inherent iteration involved in doing so, the complexity here is O(1) *per word*. Therefore if we have a sequence of words, N, the time complexity is O(N). This also has an impact on memory, since we need to hold the whole word2vec embeddings, which is on the order of O(V*D), where V is the size of our vocabulary and D is the embedding vector dimension. This is usually fixed during model training, but it’s important to consider as it consumes precious resources.

**2. Feedforward Network Layers:**

The feedforward network involves multiple matrix multiplications and additions, along with activation functions applied layer by layer. Assuming we have *L* layers, and each layer has *H* hidden units, the computational complexity of each layer is *O(I*H* + H*H*)*, where *I* is the dimension of the input from previous layer (which at first layer is the embedding dimension). This *I* would normally be equal to *H* also for all the layers after the first layer. So the complexity can be simplified to *O(H^2)* per input (single word). If you have a sequence of *N* words as input, and *L* number of layers, the overall complexity of the feedforward operation, in the worst case, becomes *O(N * L * H^2)*. Let's illustrate with a Python snippet using numpy:

```python
import numpy as np

def feedforward_layer(input_data, weights, biases, activation_func):
    """Simulates a single feedforward layer operation."""
    z = np.dot(input_data, weights) + biases
    return activation_func(z)

def feedforward_network(input_seq, layers, activation_funcs):
    """Simulates the full forward pass through multiple layers."""
    current_output = input_seq # input_seq is the embedded sequence, shape (N, D)
    for layer_index in range(len(layers)):
      current_output = feedforward_layer(current_output, layers[layer_index]['weights'], layers[layer_index]['biases'], activation_funcs[layer_index])
    return current_output


#Example Usage:
vocabulary_size = 10000
embedding_dim = 100
seq_length = 50
hidden_units = 256
num_layers = 3

# Generate example weight and bias:
layers_list = []
for layer_index in range(num_layers):
  if layer_index == 0:
    #first layer, input comes from embeddings which has the dimension embedding_dim
     layer = {'weights': np.random.randn(embedding_dim, hidden_units),
          'biases': np.random.randn(hidden_units) }
  else:
     layer = {'weights': np.random.randn(hidden_units, hidden_units),
          'biases': np.random.randn(hidden_units) }

  layers_list.append(layer)


activation_funcs = [lambda x: np.maximum(0, x)] * num_layers #ReLU functions
input_embeddings = np.random.randn(seq_length, embedding_dim) # random input
output = feedforward_network(input_embeddings, layers_list, activation_funcs)
print(f'Output shape:{output.shape}')
```
This simplified example shows matrix operations taking place in each layer; where the number of operations grows quadratically with the number of units in the layer.

**3. Output Layer & Softmax Calculation:**

The final layer usually transforms the output of the preceding layer into a probability distribution over all words in the vocabulary, which is achieved through a softmax function. This has a computational cost. The last layer will generally have the same complexity as the other layers, namely *O(H^2)*, because the transformation would be done by matrix multiplication. The softmax function itself, if done naively, has a complexity of O(V), where *V* is the vocabulary size since it calculates an exponential for every entry in the vector of the last layer. However, in practice with a large vocabulary, you'd use techniques like hierarchical softmax or sampled softmax, which can bring down this cost. As an example of a standard output layer with softmax using numpy:

```python
def output_layer(input_data, weights, biases):
    """Simulates a single output layer operation."""
    z = np.dot(input_data, weights) + biases
    return z

def softmax(z):
    """Naive softmax implementation."""
    e_z = np.exp(z - np.max(z)) #subtracting max to improve numerical stability
    return e_z / np.sum(e_z, axis=-1, keepdims=True) # keepdims is important to maintain shape


#Example Usage
hidden_units = 256
vocabulary_size = 10000
output_weights = np.random.randn(hidden_units, vocabulary_size)
output_biases = np.random.randn(vocabulary_size)
hidden_rep = np.random.randn(1, hidden_units) # output of last hidden layer, batch size 1
output_pre_softmax = output_layer(hidden_rep, output_weights, output_biases)
probabilities = softmax(output_pre_softmax)
print(f'Probabilities shape: {probabilities.shape}')
```

This snippet shows how matrix multiplication and softmax are combined at the output layer. Note how softmax complexity will rise linearly with the size of the vocabulary.

**Overall Complexity:**

Putting it all together, the overall computational complexity for a feedforward neural network language model with word2vec embeddings is dominated by *O(N*L*H^2)* from the forward pass of the hidden layers. The vocabulary lookup costs *O(N)* which is comparatively small and therefore negligible relative to the complexity of the hidden layers. The output layer, if done naively with standard softmax, contributes *O(V)*, if we consider the whole vocabulary. In reality, the softmax complexity is reduced through methods like sampled softmax or hierarchical softmax, which results in a less dramatic growth. Therefore we should always consider the details of the particular implementations when calculating complexity for this step. However, the main cost comes from the matrix multiplications in each hidden layer. *H* and *L* are generally fixed when we use the model after training, so the largest impact comes from *N*, the sequence length.

**Important Considerations:**

*   **Batching:** These complexities assume single examples or very small sequences. When you process in batches, which is the norm when training neural networks, the complexity is similar, but operations are being done in parallel. For batch size *B*, the complexity becomes *O(B*N*L*H^2)*. This will decrease the training time because operations are parallelized.
*   **Implementation:** The actual speed can significantly depend on the underlying implementation. Optimized libraries like NumPy, TensorFlow, or PyTorch have highly efficient implementations of matrix multiplications and other operations.
*   **Practical implications:** In my experience building models in the past, the matrix multiplication cost from the neural networks can greatly influence model performance. I had cases where just tuning the number of units, or layers, made all the difference when trying to meet latency requirements for user interaction.

**Recommended Resources:**

For a deeper understanding of computational complexity, I'd highly recommend diving into "Introduction to Algorithms" by Thomas H. Cormen et al. It’s a classic for a reason. Additionally, “Deep Learning” by Ian Goodfellow et al. is an excellent resource for the complexities inherent in neural networks, which also has a chapter on Recurrent Neural Networks and its complexity. Finally, for a more in-depth understanding of word embeddings and natural language processing, "Speech and Language Processing" by Daniel Jurafsky and James H. Martin is an invaluable resource.

In conclusion, while it might look a bit daunting at first, understanding how complexity scales with the various components will help you build more efficient and effective language models. The key takeaway is that the forward pass through the hidden layers of a feedforward network dominates the computational cost, and this cost is directly related to the square of the hidden units in the layers, the number of layers, and the length of the input sequence. Keep those factors in mind and you will be in good shape when dealing with similar issues in the future.
