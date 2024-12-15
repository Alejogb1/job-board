---
title: "What are the constraints and independence of ANN?"
date: "2024-12-15"
id: "what-are-the-constraints-and-independence-of-ann"
---

Alright lemme break this down for you I've been wrestling with neural nets since way back when I swear the libraries were still in diapers so I've seen some things with constraints and independence that make you wanna pull your hair out

So constraints in ANNs yeah that's a big one It's like trying to fit a square peg in a round hole if you get it wrong You got various types of constraints to think about think of them like rules of the game or limitations of the tools we use

First off you got the architectural constraints Like you can't just slap any layer on top of another and expect magic You gotta think about the input and output shapes think about convolutional layers that only handle spatial data not like time series and things like that You've also got stuff like recurrent layers that are more like a chain where the output of one element in the sequence feeds into the next element think of it like data that has a time dependency or text data which one word has a relationship with the other

Then there's the constraint of the activation functions You can't just use anything you want for that There are the Sigmoids which are great for binary classification problems but have a vanishing gradient problem which slows down learning significantly. Then you've got ReLU it's pretty popular now it's fast to compute but if you're not careful you can get dead neurons where the output will be zero or close to zero and not updating And let's not forget about Tanh which outputs values between -1 and 1 which gives you a normalized output if your use case needs it

Let's check this example out using Python and Tensorflow for clarification

```python
import tensorflow as tf

# Example of an architecture constraint

model_architecture_example = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')
])

# This shows you that you can't just plug any shape input it needs to be 28x28x1
# If you try to input something else you get error.

print(model_architecture_example.summary())
```

You can see that we have defined the input shape at the convolutional layer to handle an image of 28x28 pixels with 1 channel for greyscale images if you try to pass this another shape your code will explode

And then you have resource constraints like you can't train a massive billion parameter model on your grandpa's computer it just won't work You need enough processing power like GPUs or TPUs and memory to handle all the calculations this is like trying to fill a swimming pool with a garden hose it will take a long time and maybe never complete. Then there is also time constraints where you only have a certain time to get the model trained

Also you have the training data itself this is like the quality of your ingredients in the cooking process if your ingredients are bad your food will also be bad. If your data is biased or has errors the model will learn these problems and then do the same or amplify them. You also have the problem of the size of your data if you don't have enough examples you will underfit and if you have a lot of noisy examples then you'll overfit

Now let's move on to independence in neural nets and this is where things get a little bit more interesting

First off you have independence between the individual neurons in a single layer This is like each neuron is its own processing unit and it's doing its own calculations based on its specific weights and biases then the outputs of these neurons become the inputs of the next layer

Then you've got independence between the layers in the network Each layer is learning features at a different level of abstraction if you are dealing with images in a convolutional neural network you will have the first layers learning about lines and corners and then later layers will learn complex shapes or patterns. If you are dealing with text you may have embedding layers first then recurrent layers and then attention layers and so forth and each layer learns in a different way

There is also the concept of independent model runs you don't expect the same exact results every time you train a model This is due to the random initialization of weights and biases in the beginning and other random stuff in the training algorithm. It's like rolling the dice each run can be different if you get it just right it will learn the information you want. I've seen some people running the same model with the same data and getting very different accuracy results. In fact, I had a model way back where I used random seed 42 and then the next day I forgot that I changed the random seed and got completely different results. I literally thought I broke something until I realized I messed with the seed variable.

Now let's see some Python code to demonstrate the independence of each neuron with different weights and biases

```python
import numpy as np
import tensorflow as tf

#Example of weights and biases for a simple neuron

#Random weights and biases
weights = np.random.rand(3)
bias = np.random.rand(1)

# Dummy input
inputs = np.array([0.5, 0.2, 0.8])

#Output of the neuron
output = np.dot(inputs, weights) + bias
print(output)

# Example of layers and how weights are independent between layers
model_independence_layers = tf.keras.models.Sequential([
    tf.keras.layers.Dense(16, activation = 'relu', input_shape = (4,)),
    tf.keras.layers.Dense(8, activation = 'relu'),
    tf.keras.layers.Dense(2, activation = 'softmax')
])

# Get the weights of each layer
for layer in model_independence_layers.layers:
    weights = layer.get_weights()
    if weights:
        print(f"Weights for layer {layer.name}: {weights}")

```

The first part shows you how each neuron has random weights and biases to start and the second part shows how layers have independent weights you will see that each layer learns its own parameters and their values are different

And I'll throw in one more for good measure and show how different runs of the same model will give you different results because of the randomness

```python
import tensorflow as tf
import numpy as np

# Simple model for comparison

def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# dummy data to fit
x_train = np.random.rand(100, 5)
y_train = tf.keras.utils.to_categorical(np.random.randint(0,2,100), num_classes = 2)

# Run model twice with different random states
model_1 = create_model()
history_1 = model_1.fit(x_train, y_train, epochs=10, verbose=0)
accuracy_1 = history_1.history['accuracy'][-1]

model_2 = create_model()
history_2 = model_2.fit(x_train, y_train, epochs=10, verbose=0)
accuracy_2 = history_2.history['accuracy'][-1]

print(f'Accuracy model 1: {accuracy_1}')
print(f'Accuracy model 2: {accuracy_2}')

# This is to show how using a seed will get you the same result
tf.random.set_seed(42)
model_3 = create_model()
history_3 = model_3.fit(x_train, y_train, epochs=10, verbose = 0)
accuracy_3 = history_3.history['accuracy'][-1]


tf.random.set_seed(42)
model_4 = create_model()
history_4 = model_4.fit(x_train, y_train, epochs=10, verbose = 0)
accuracy_4 = history_4.history['accuracy'][-1]


print(f'Accuracy model 3: {accuracy_3}')
print(f'Accuracy model 4: {accuracy_4}')

```

You will see that Model 1 and 2 will have different results. Model 3 and 4 will have the same results because we set a seed to control the randomness of the process. So yeah it's not perfect randomness but it's pretty close

So these are some basic examples of constraints and independence in neural nets I know that understanding them in detail will give you a significant advantage in your work

If you want to dig deeper and understand the math behind this I recommend checking out the "Deep Learning" book by Goodfellow Bengio and Courville and also the Deep Learning Specialization offered by Andrew Ng on Coursera. They are great resources for getting into the mathematical and the practical implementation of this stuff.
