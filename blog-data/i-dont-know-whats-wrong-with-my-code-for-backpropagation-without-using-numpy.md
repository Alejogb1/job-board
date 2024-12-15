---
title: "I don't know what's wrong with my code for Backpropagation without using numpy?"
date: "2024-12-15"
id: "i-dont-know-whats-wrong-with-my-code-for-backpropagation-without-using-numpy"
---

alright, so you’re having trouble with backpropagation when you’re avoiding numpy, huh? i get it. it's a classic situation. been there, done that, got the t-shirt and the battle scars to prove it. let me walk you through my experience and some approaches that might get you unstuck.

i remember when i first tried to implement backprop from scratch. this was way back, before everyone and their cat were using pytorch or tensorflow. i was working on a tiny project, a simple neural network to classify handwritten digits – yeah, the good old mnist dataset. i thought, "how hard can it be?" famous last words. i decided against using any high level libraries, because the point of the project was precisely to grasp the inner workings, so no numpy for me either.

i started with the forward pass, which felt easy enough, just matrix multiplications and activation functions. then backpropagation reared its ugly head. it was a nightmare of trying to keep track of partial derivatives, multiplying things in the correct order, and not losing my sanity in the process. the first version i wrote was just a mess, a tangled ball of loops and calculations. debugging it was a real pain. i spent hours staring at the code, scratching my head, and mumbling to my screen. it turns out, the biggest issue i had was not the math but more the bookkeeping. i was messing up the indexing of the partials and applying them to wrong weights. it's a very common mistake i've seen people make.

the crux of backpropagation is the chain rule, of course. you are essentially propagating the error from the output layer back to the input layer, layer by layer. this error informs how the network should tweak its weights to minimize loss. when i am doing it myself without libraries, it usually comes down to three things:

1.  calculating the error of the output layer.
2.  calculating the gradients of the error with respect to the weights and biases of each layer.
3.  updating the weights and biases with these gradients.

so lets assume we are working with a multi-layer perceptron (mlp). i'll show you a minimal implementation using standard python lists:

```python
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def forward_pass(inputs, weights, biases):
    layers_output = []
    layer_input = inputs
    for weight, bias in zip(weights, biases):
        layer_output = [sigmoid(sum(w*i for w, i in zip(weight_row, layer_input)) + b) for weight_row, b in zip(weight, bias)]
        layers_output.append(layer_output)
        layer_input = layer_output
    return layers_output

def backward_pass(layers_output, weights, biases, y_true, learning_rate):
    # i'm going to use this example to update the weights, but the implementation is the key here
    # this could be way faster and more memory effcient depending on the language
    # but let's keep it simple
    n_layers = len(weights)
    error = [(out - y) for out, y in zip(layers_output[-1], y_true)]
    delta = [error]
    for layer_index in range(n_layers - 1, 0, -1):
        layer_delta = []
        for node_index in range(len(layers_output[layer_index - 1])):
            node_sum_delta = 0
            for upper_node_index in range(len(delta[-1])):
                node_sum_delta += delta[-1][upper_node_index] * weights[layer_index][upper_node_index][node_index]
            layer_delta.append(node_sum_delta * sigmoid_derivative(layers_output[layer_index-1][node_index]))
        delta.append(layer_delta)

    delta.reverse() # back to the same direction as the forward pass.
    updated_weights = []
    updated_biases = []
    layer_input = [inputs] + layers_output[:-1]

    for layer_index in range(n_layers):
        updated_layer_weight = []
        updated_layer_bias = []
        for node_index in range(len(weights[layer_index])):
             new_weight_row = []
             for weight_index in range(len(weights[layer_index][node_index])):
                new_weight = weights[layer_index][node_index][weight_index] - learning_rate * delta[layer_index][node_index] * layer_input[layer_index][weight_index]
                new_weight_row.append(new_weight)
             updated_layer_weight.append(new_weight_row)
             new_bias = biases[layer_index][node_index] - learning_rate * delta[layer_index][node_index]
             updated_layer_bias.append(new_bias)

        updated_weights.append(updated_layer_weight)
        updated_biases.append(updated_layer_bias)

    return updated_weights, updated_biases
```

this is a simplified version, and it would need to be generalized for different network architectures and activation functions. the key point here is how to calculate the deltas and weights using loops, which mirrors how the chain rule works.

now, common issues i've noticed with this:

*   **incorrect indexing:** this one is huge. as you can see from the example, i've been extremely explicit with my indexing variables. it’s easy to mix up layer indices, node indices, and weight indices when not careful. that was one of my major struggles.

*   **error calculation:** the way you calculate the output error needs to match your chosen loss function, i.e. cross-entropy loss or mean squared error (mse), or whatever. usually, using mean squared error loss (mse) might seem intuitive and easy, but in reality, when you want to classify things with more than two categories, you'd better use cross-entropy loss instead. also the derivative of your chosen activation function must be correct. if you're using a sigmoid, the derivative needs to be `sigmoid(x) * (1 - sigmoid(x))`.

*   **updating weights:** the weight update rule must consider the learning rate and the calculated gradients. missing a minus sign in the update step will, for example, make the model learn the opposite of what it should. yeah, been there.

*   **numerical instability:** if you're using sigmoid as an activation function, you might encounter problems when the values get too big or too small (you may start seeing gradients becoming almost zero, causing learning to stall). this can be handled by using a different activation function or normalization techniques. relu is an option here, you should be able to implement it fairly easily.

let's move to another common issue people have. batch processing. the above code works with a single sample. in practice you want to pass more than one example to reduce noise and accelerate your training. that involves more than just doing everything inside a bigger for loop. you usually want to use vectorization, even without numpy. let me show you how to use the code above with batches:

```python
def batch_forward_pass(batch_inputs, weights, biases):
    batch_layers_output = []
    batch_size = len(batch_inputs)
    layer_input = batch_inputs
    for weight, bias in zip(weights, biases):
        layer_output = []
        for sample_index in range(batch_size):
            single_output = [sigmoid(sum(w*i for w, i in zip(weight_row, layer_input[sample_index])) + b) for weight_row, b in zip(weight, bias)]
            layer_output.append(single_output)
        batch_layers_output.append(layer_output)
        layer_input = layer_output
    return batch_layers_output


def batch_backward_pass(batch_layers_output, weights, biases, y_true_batch, learning_rate):
    n_layers = len(weights)
    batch_size = len(y_true_batch)
    updated_weights = [ [ [0 for _ in weights[layer][node]] for node in range(len(weights[layer])) ] for layer in range(n_layers) ]
    updated_biases = [ [0 for _ in biases[layer]] for layer in range(n_layers) ]


    for sample_index in range(batch_size):
       y_true = y_true_batch[sample_index]
       layers_output = [layer[sample_index] for layer in batch_layers_output] # getting just the sample data from the batch
       error = [(out - y) for out, y in zip(layers_output[-1], y_true)]
       delta = [error]
       for layer_index in range(n_layers - 1, 0, -1):
         layer_delta = []
         for node_index in range(len(layers_output[layer_index - 1])):
           node_sum_delta = 0
           for upper_node_index in range(len(delta[-1])):
             node_sum_delta += delta[-1][upper_node_index] * weights[layer_index][upper_node_index][node_index]
           layer_delta.append(node_sum_delta * sigmoid_derivative(layers_output[layer_index-1][node_index]))
         delta.append(layer_delta)

       delta.reverse()
       layer_input = [batch_inputs[sample_index]] + layers_output[:-1]
       for layer_index in range(n_layers):
          for node_index in range(len(weights[layer_index])):
              for weight_index in range(len(weights[layer_index][node_index])):
                   updated_weights[layer_index][node_index][weight_index] -= learning_rate * delta[layer_index][node_index] * layer_input[layer_index][weight_index]
              updated_biases[layer_index][node_index] -= learning_rate * delta[layer_index][node_index]
    return updated_weights, updated_biases

```
notice how i am now going through each sample, then accumulating the weights for each sample at each iteration. i am not going to explain that in more detail because the main intention of this response is not to solve the issue for you but to show you how to approach it and give you insights into the common mistakes.

and just to be clear, i am not using vectorization in the above example, just batch processing. true vectorization without numpy involves implementing matrix operations and data structures with custom loops. which might get painful and the benefits are not as high as one might think. python is slow for the job, c++ or rust would be better.

i'll be honest, implementing backprop from scratch without numpy is a valuable learning experience, but it's very time-consuming for practical applications. the code is slow, difficult to debug, and hard to extend for complex models. it is a good exercise, for sure! but once you understand the basic idea, using libraries like pytorch or tensorflow becomes much more efficient. i mean, come on, even the computer science jokes are funnier when using libraries, you know, why did the neural network cross the road? because it was trained on a really good dataset. hehe.

if you really want to deepen your understanding, instead of following a tutorial or some github gist (please don't!), i would recommend these resources:

*   **"deep learning" by ian goodfellow, yoshua bengio, and aaron courville.** this book is a comprehensive overview of deep learning, covering all aspects, including the mathematical foundations of backpropagation.
*   **"neural networks and deep learning" by michael nielsen.** this is an online book that provides a more accessible introduction to neural networks and backpropagation. it includes interactive visualizations to help you understand the concepts.

they might seem a bit daunting at first, but diving deep into these resources is the best way to get the knowledge and insights you need. reading the mathematics behind the operations will give you a much deeper understanding than just copying code and modifying parameters. also, make sure you go slowly and understand each step, even the most basic ones. understanding the math behind backpropagation is key to making sure you implement it correctly.

finally, when you're coding it up, start with a very simple example and verify it manually. calculate the gradients by hand, and make sure they match what your code is outputting. this will save you hours of debugging.

i hope that gives you a good starting point. remember, it is a learning process, don't get frustrated when things don't work immediately. keep coding, keep learning, and most importantly keep debugging.
