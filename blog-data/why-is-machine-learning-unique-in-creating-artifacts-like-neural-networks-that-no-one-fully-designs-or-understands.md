---
title: "Why is machine learning unique in creating artifacts like neural networks that no one fully designs or understands?"
date: "2024-12-11"
id: "why-is-machine-learning-unique-in-creating-artifacts-like-neural-networks-that-no-one-fully-designs-or-understands"
---

 so you wanna know why machine learning is all weird and creates these brain-like things nobody really gets right  its kinda mind blowing right  Like we build these neural networks these crazy interconnected webs of math and data and they learn stuff on their own  we dont tell them exactly what to do we just give them a bunch of examples and say "figure it out" and they do  its like having a super smart pet that learns way faster than you can teach it  and thats the weird part  we design the basic architecture the overall structure but the specifics the intricate connections the weights and biases that actually make it work thats all learned  its emergent behavior  its like building a really complex Lego castle but instead of following instructions the Legos themselves figure out how to assemble themselves into a cool castle

Its not like traditional programming where you explicitly tell the computer every single step its more like training a really really smart dog  you show it what a "fetch" is a bunch of times and eventually it understands even if you dont fully understand exactly how it internalized the concept  neural networks are similar  we define the learning process we give it data and a goal but the internal representation the actual "understanding" the network develops thats a black box for the most part  we can probe it we can analyze its performance but understanding the precise workings of a large complex neural network  thats a monumental task  think about a giant city  you can see the roads the buildings the overall layout but understanding every single interaction every single person's journey within that city  thats impossible  similarly understanding every connection and weight in a large neural network its just too complex  thats what makes it unique  its a form of automated design and automated understanding  we create the system that creates the understanding

And this isn't just some philosophical point its got real world implications  consider the problem of bias  if you train a neural network on biased data it will learn those biases  it will replicate them in its predictions even if you dont intend it to  thats why its crucial to have diverse and representative datasets  but even with perfect data there are still surprises  sometimes a network learns shortcuts it finds patterns in the data that we dont even see  its like it discovers hidden rules or connections that we missed completely its almost as if its making its own discoveries

This lack of complete understanding isnt necessarily bad  sometimes its an advantage  if we knew exactly how a network worked we might be limited by our own biases and assumptions  the networks ability to find unexpected solutions is a powerful tool  but its also a challenge  we need to develop new methods for interpreting and understanding these networks if we want to use them responsibly  think about medical diagnosis  imagine a neural network that can diagnose diseases more accurately than any doctor but we dont fully understand how it arrives at its diagnosis  thats both incredible and terrifying right

Now lets look at some code examples to make this a bit more concrete  Ill show you simple examples but real world networks are way more complex


**Example 1: A Simple Perceptron**

This is a basic building block of neural networks a single neuron

```python
import numpy as np

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def perceptron(inputs weights bias):
  summation = np.dot(inputs weights) + bias
  output = sigmoid(summation)
  return output

inputs = np.array([0.5 0.8])
weights = np.array([0.2 0.7])
bias = 0.3

output = perceptron(inputs weights bias)
print(f"Output: {output}")
```

This is super simple just one neuron  but you can see how weights and bias affect the output  and its this process of adjusting weights and biases based on data that leads to learning  we dont explicitly program the sigmoid function to recognize patterns we just train it to


**Example 2:  A Simple Neural Network with One Hidden Layer**

This adds a hidden layer making things a little more interesting


```python
import numpy as np

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def neural_network(inputs weights1 weights2 bias1 bias2):
  hidden_layer = sigmoid(np.dot(inputs weights1) + bias1)
  output = sigmoid(np.dot(hidden_layer weights2) + bias2)
  return output

inputs = np.array([0.1 0.9])
weights1 = np.array([[0.3 0.7] [0.2 0.6]])
weights2 = np.array([0.5 0.4])
bias1 = np.array([0.1 0.2])
bias2 = 0.3

output = neural_network(inputs weights1 weights2 bias1 bias2)
print(f"Output: {output}")
```

See  we have weights and biases for each connection  again  the magic is in how these are adjusted during training through backpropagation a process that we understand in principle but whose full effects on a large network are often opaque

**Example 3:  A Tiny bit of TensorFlow/Keras**

This shows a more realistic (though still extremely simplified) setup using a popular deep learning library


```python
import tensorflow as tf

model = tf.keras.Sequential([
  tf.keras.layers.Dense(128 activation='relu' input_shape=(784,))
  tf.keras.layers.Dense(10 activation='softmax')
])

model.compile(optimizer='adam'
              loss='sparse_categorical_crossentropy'
              metrics=['accuracy'])

#This would be where you'd load and preprocess your data
#  mnist = tf.keras.datasets.mnist.load_data()
#  (x_train y_train) (x_test y_test) = mnist
#  ...preprocess data...

#model.fit(x_train y_train epochs=10)

#This line is commented out because training takes time and this is a simplified example

```

This is a very basic example  but even here its clear that the specifics of what the network learns are not directly programmed  we specify the architecture the optimizer and the loss function but the actual learned weights  thats still mysterious


To understand this better look into books like "Deep Learning" by Goodfellow Bengio and Courville  "Pattern Recognition and Machine Learning" by Christopher Bishop or papers on specific network architectures or training methods  Its a vast field but these resources will give you a solid base

So yeah  machine learning's unique because it creates these systems that learn and adapt in ways we dont fully grasp  Its a powerful tool but its also a complex one  and understanding its complexities is a challenge we are still facing
