---
title: "How to use Keras U-net implementation for IRIS segmentation with a weighted loss function?"
date: "2024-12-15"
id: "how-to-use-keras-u-net-implementation-for-iris-segmentation-with-a-weighted-loss-function"
---

alright, let's tackle this u-net for iris segmentation with weighted loss thing. i've been down this road before, believe me. it’s not as straightforward as plugging in some code and hoping for the best. there are a couple of gotchas that can really throw a wrench in your plans if you're not careful.

first off, when you're dealing with segmentation, especially with medical images or something like iris scans where the target area might be a small part of the image, class imbalance becomes a real pain. if your network is just trained on the raw pixel data, it’ll learn to predict the background and ignore the iris because the background is way more common. hence, the need for weighted losses which i'll get to in a bit.

now, u-net in keras. it’s a solid choice, a workhorse if you will, especially for image segmentation tasks like this. i remember the first time i implemented it for some satellite imagery work, and i spent like a week debugging why it wasn’t converging; it turned out to be a stupid mistake in data loading. anyway, you want to make sure you have a good implementation to start with. it is always a good idea to look at keras documentation or even at github's repos that implement u-net, plenty of examples there.

regarding the iris segmentation, i've worked on this problem in a project some years back. we had different lightning conditions and different types of camera sensors, so a generalized and robust solution was a must. this means you'll probably need a decent amount of data with good labeling, also you can investigate data augmentation to help with diversity, rotations and flips usually help. if you are doing medical imaging tasks also contrast and brightness augmentation.

so, on to the code, let’s assume you have your images and masks ready. you have to make sure your masks are one-hot encoded (if needed) and are the same dimensions as the input images, otherwise the keras model is gonna shout errors.

first step is to get a basic unet model using keras. here is an example of how you could implement a simple u-net:

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def unet(input_size=(256, 256, 3)):
    inputs = keras.Input(input_size)

    # encoder
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    # bottleneck
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv4)

    # decoder
    up5 = layers.Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(conv4)
    merge5 = layers.concatenate([conv3, up5], axis=3)
    conv5 = layers.Conv2D(256, 3, activation='relu', padding='same')(merge5)
    conv5 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv5)

    up6 = layers.Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv5)
    merge6 = layers.concatenate([conv2, up6], axis=3)
    conv6 = layers.Conv2D(128, 3, activation='relu', padding='same')(merge6)
    conv6 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv6)

    up7 = layers.Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv6)
    merge7 = layers.concatenate([conv1, up7], axis=3)
    conv7 = layers.Conv2D(64, 3, activation='relu', padding='same')(merge7)
    conv7 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv7)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(conv7) # for binary segmentation

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
```

this is a basic u-net architecture, i used relu activation functions, but you can try different ones. note that `outputs = layers.Conv2D(1, 1, activation='sigmoid')(conv7)` is a final `conv2d` with a sigmoid activation function, as we are dealing with a binary segmentation problem. if you need multi-class segmentation, then you need a softmax activation and a different number of output channels.

now, here is where the weighted loss comes into play, like i previously said. you want to give more importance to the iris pixels. if you use a standard binary cross-entropy loss, your network is more likely to become a background predictor because it’s the easier task. to tackle that, we use a weighted loss function. a common choice is to use a weighted binary cross-entropy:

```python
import tensorflow as tf

def weighted_binary_crossentropy(y_true, y_pred, weight_iris=10.0, weight_background=1.0):
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1])

    epsilon = tf.keras.backend.epsilon()
    y_pred_flat = tf.clip_by_value(y_pred_flat, epsilon, 1 - epsilon)

    bce = y_true_flat * tf.math.log(y_pred_flat) + \
          (1 - y_true_flat) * tf.math.log(1 - y_pred_flat)
    bce = -bce

    weights = y_true_flat * weight_iris + (1 - y_true_flat) * weight_background
    weighted_bce = weights * bce
    return tf.reduce_mean(weighted_bce)
```

in that function we define a weight for the iris class (`weight_iris`) and a weight for the background (`weight_background`). you might want to experiment with different values. if you're unsure how to calculate the correct weights, check out the paper “a survey on loss functions for semantic segmentation”, this can help you get a better grasp on the best loss function to use. in the past i used the formula `total_pixels/total_iris_pixels` as a way to calculate a better `weight_iris`, but it's not the only way. also `1/frequency_of_each_class` is another very common choice.

a critical step is to use this loss function while compiling the model. here is an example:

```python
import tensorflow as tf
from tensorflow import keras

model = unet()
optimizer = keras.optimizers.Adam()
model.compile(optimizer=optimizer, loss=weighted_binary_crossentropy, metrics=['accuracy'])
model.summary()
```

`model.summary()` is very useful for checking that your layers are right. i usually spend a little time there and it helps to catch errors or wrong configurations, it saves me a lot of time.

before running the training you have to deal with the data, make sure everything is normalized. it helps a lot with convergence. i've seen some training get stuck because of badly normalized data. also, make sure that your data loading is correct, this is fundamental; i have personally spent days with wrong data loaders. when you think you are sure, triple check your data loaders.

now, about resources, books are a treasure trove. “deep learning with python” by francois chollet is a great starter, although it might not cover advanced segmentation issues, it helps a lot with basic concepts. for more specialized stuff, maybe try "computer vision: algorithms and applications" by richard szeliski, it’s a very robust book with plenty of computer vision concepts. also the paper mentioned before “a survey on loss functions for semantic segmentation” will help you a lot with your segmentation loss functions.

one last thing, don’t expect perfection on the first try, training is a science but it's also a bit of an art. you'll probably need to tweak your learning rates, optimizers, maybe the architecture itself. the first couple of days i was using adam with a learning rate of 0.1, i was wondering why it was not learning anything!. just experiment and keep looking at the loss function and validation curves.

ah, i almost forgot a joke a python programmer would love. why do pythonistas hate private variables? because they are so hard to access.

anyway, hope this helps, let me know if you get stuck in some particular part, i'll try to give you some tips. good luck!
