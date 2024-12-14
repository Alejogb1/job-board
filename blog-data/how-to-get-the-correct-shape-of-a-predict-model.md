---
title: "How to get the correct shape of a predict model?"
date: "2024-12-14"
id: "how-to-get-the-correct-shape-of-a-predict-model"
---

well, alright then, let's talk about model shape. it's a common head-scratcher, i've been there, trust me. it's one of those things that feels simple until it's not. i'm assuming you're dealing with some kind of machine learning model here, and by "shape" you mean the dimensionality of the output, the number of predictions it spits out, and how that aligns with what you actually expect.

it's not always about the model's architecture. sure, a convolutional neural net versus a recurrent one will have vastly different inner workings, but the output shape is a separate issue. often it’s a layer we add or a function we use at the very end of the prediction pipeline that determines the final outcome shape.

i remember this one project back in '17, working on a natural language task, trying to build a sentiment analysis model. i was using a simple lstm, seemed straightforward enough. the training went smoothly, validation loss decreased nicely, but when i started making predictions, i was getting this weird 3d output instead of the 1d vector of probabilities i anticipated. i had completely overlooked the last layer, a dense layer with the wrong number of output units. wasted a good few hours because of that, mostly tracing code and printing shapes, a common habit i developed after debugging for hours like this one.

the fundamental issue comes from a mismatch between what the model calculates internally and what the actual prediction should be. the core is the last layer's size and activation function. if you want a single prediction score (like in classification) you will need a single neuron in that final layer. if you expect a probability distribution over multiple classes you'll need a neuron for each class.

let's look at some examples using common python libraries. suppose you're using keras with tensorflow as a backend, which is what i mostly use.

here's a simplified example for binary classification. imagine you want the model to predict whether a given piece of text is positive or negative, which is pretty much what i did with my sentiment analysis tool i mentioned before. you only want one output unit for the probability between 0 and 1:

```python
import tensorflow as tf
from tensorflow import keras
from keras import layers

# model setup (simplified, actual model can be more complex)
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(100,)), #input layer
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # output layer, one unit, sigmoid activation
])

# lets say we have 3 example vectors as inputs:
example_inputs = tf.random.normal(shape=(3, 100))

#get predictions from the model
predictions = model(example_inputs)
print(predictions.shape) # outputs (3, 1), 3 samples, 1 probability value
```

notice the last dense layer `layers.dense(1, activation='sigmoid')`. the `1` is important here, it sets the output to have a single unit. and `sigmoid` pushes the output to be between `0` and `1`.

now, let's say you're doing multiclass classification. suppose you want to predict if an image contains one of three classes: 'cat', 'dog', or 'bird'. in this case, the final layer needs three units, one for each class and the final activation function `softmax` will translate the raw values into probabilities for all classes that add up to 1. the example below simulates this, you won't see images but the final layers are key for the output shape:

```python
import tensorflow as tf
from tensorflow import keras
from keras import layers

model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),  # example input shape for images
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(10, activation='relu'),
    layers.Dense(3, activation='softmax') # 3 units for each class, softmax activation
])

#lets simulate batch of 2 examples with 32x32x3 images as inputs:
example_inputs = tf.random.normal(shape=(2, 32, 32, 3))

#get predictions from the model
predictions = model(example_inputs)
print(predictions.shape) #outputs (2, 3), 2 samples and 3 probability values
```

the last layer now has `layers.dense(3, activation='softmax')`, 3 output units and the `softmax` activation function, now each element of the `(2,3)` matrix will contain a probability for each of the 3 classes.

and what if you are doing a regression task? like predicting a house price based on its features. in this case, you often want a single numerical value output. this means one output unit and you don't need an activation function since we are predicting a real value which is not constrained between `0` and `1`. although you can add it if you need specific constraints on the output:

```python
import tensorflow as tf
from tensorflow import keras
from keras import layers

model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(5,)), #5 features as input
    layers.Dense(32, activation='relu'),
    layers.Dense(1)  # 1 unit, no activation
])

# simulating 4 houses with 5 features as input
example_inputs = tf.random.normal(shape=(4, 5))

predictions = model(example_inputs)

print(predictions.shape) # outputs (4, 1), 4 samples, each with one predicted value
```

here the last layer `layers.dense(1)` gives a single output value, which could be any real number, perfect for regression tasks.

now, let's talk a bit about sequence prediction problems, that are not uncommon, like predicting the next character in a text. in recurrent neural networks, you need to consider the sequence length. if you want to predict a sequence of, say 10 characters, your output shape needs to reflect that as well. you'll need a similar approach of output units per character but also considering the sequence length in the output matrix shape.

one key thing to remember is that often people mistakenly add activation functions that squeeze the range of the output when it isn't needed, like adding a sigmoid activation in a regression task or in sequence-to-sequence models for a task where each predicted unit doesn't need to be bounded by 0 or 1. i know it sounds trivial, but happens more than what one may think.

a good practice i always recommend is, always print the model's output shape right after the prediction step to see if that shape is what you expect. that helped me countless times in debugging. and don't forget to double check the final layer architecture. a tiny typo there, and you'll be staring at code for hours figuring out what went wrong.

debugging these kind of things is part of the fun, but only after you solve them, of course. there's this joke: why was the deep learning model so bad at tennis? because it had a poor serve!

as for some good resources, aside the official documentation of keras and tensorflow which i recommend to explore carefully, i would recommend "deep learning" by goodfellow, bengio, and courville which, while dense, it's one of the more complete theory resources you can find and the chapter regarding feedforward and recurrent neural networks is a must read. also, for more hands on practical explanations, "hands-on machine learning with scikit-learn, keras, and tensorflow" by aurelien geron is very helpful, specially when you want to learn how to use the libraries the way it was intended.
it always worth it to check the documentation from the libraries as they may change between version and keep up to date on the latest updates.

so in short, to get the correct shape, focus on the last layer: the number of units and the activation function must match your prediction needs. and always double check with printing shapes right after model predictions. it's a fundamental part of every single machine learning model pipeline. if you keep that in mind and practice, it becomes second nature and it won't be a problem anymore.
