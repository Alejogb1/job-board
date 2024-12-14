---
title: "Why isn't my Convolutional Neural Network reaching an acceptable accuracy?"
date: "2024-12-14"
id: "why-isnt-my-convolutional-neural-network-reaching-an-acceptable-accuracy"
---

well, alright, let's get into this. you've got a convolutional neural network, and it's just not performing like you expect, huh? i've been there, trust me. i've spent more late nights staring at training curves than i care to remember. it's frustrating when things don't just click into place, especially when you've put the work in.

first off, it's pretty common. cnn's aren't magic wands, they need the conditions to be just so. there's a whole bunch of potential culprits, and we need to systematically check through them. i'll walk you through some of the usual suspects, based on my experience in the trenches, and hopefully we can pinpoint what's causing your headache.

first thing to check, is your data. i mean it, really check it. a lot of the times, the root of the problem is the data itself, not the model architecture. are you using enough data? neural networks, especially cnn's, are data hungry beasts. a small dataset, say in the hundreds, will just not cut it, most likely. you'll need thousands of examples for each class, at the very least, but more is better. i recall one time, back when i was working on a cat vs. dog classifier (original, i know), i was getting lousy results. i was pulling my hair out, convinced it was a model problem, only to realize i'd only scraped about 200 images per category. after getting about 5000 images, the thing started actually learning.

data quality is also crucial. are your images labeled correctly? any noise? are the classes balanced? if one class has 10,000 samples and another one only has 100, the network will overwhelmingly focus on the larger one and your minority class will be ignored. it's like trying to learn a language by hearing one word repeatedly and only a few other words once in a blue moon. you can handle imbalanced data via techniques such as resampling and weights. take a look at the "neural networks and deep learning" book by michael nielsen, it has a good chapter on that topic.

here is a snippet, using the keras library, that can help visualize the data:

```python
import matplotlib.pyplot as plt
import tensorflow as tf

def show_images_from_directory(directory, num_images=9):
    """display sample images from a directory."""
    images = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels='inferred',
        label_mode='int',
        image_size=(256, 256),
        shuffle=True,
        batch_size=num_images
    )

    plt.figure(figsize=(10,10))
    for images_batch, labels_batch in images.take(1):
        for i in range(num_images):
            ax = plt.subplot(3,3, i + 1)
            plt.imshow(images_batch[i].numpy().astype("uint8"))
            plt.title(f"label: {labels_batch[i].numpy()}")
            plt.axis("off")
        plt.show()
# usage, replace the folder path
show_images_from_directory("/path/to/your/image/folder")

```

now, let's talk about the model. are you sure your architecture makes sense for the kind of data you have? are the layers configured well? maybe you went too deep with the model and it is simply not learning anything, or maybe it's too shallow and does not have enough capacity to learn complex features. i recall when i was experimenting with medical image processing, i was using way too many convolutional layers and my validation loss was just staying flat. simplifying the model greatly improved the performance. it's a balancing act. it is a good idea to research the type of layers used before jumping into building your own cnn.

another very important factor is the size of your convolutional kernel, also known as filter. for example a 3x3 filter will see the image differently than a 7x7. these filters are what 'extract features', the smaller ones being good for capturing fine details, the larger ones can capture more global information. the size is important but also the number of them. if you use too many filters, you run into what is called 'overfitting'.

also, what about the activation functions? are you using relu, sigmoid, tanh? those small things have a tremendous impact on the network and how it behaves. relu for example is really popular since it does not suffer from the 'vanishing gradients' problem, but it can sometimes be affected by what is called the 'dying relu' phenomenon, meaning it becomes inactive and therefore does not learn any more. the activation function is a whole area of study in itself. research those too! i recommend "deep learning" by ian goodfellow, yoshua bengio, and aaron courville for more details.

this is a sample configuration using tensorflow/keras:

```python
import tensorflow as tf

def create_simple_cnn(input_shape, num_classes):
  model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
  return model

#usage:
input_shape = (256,256,3) # example input size
num_classes = 10 # number of different classes
model = create_simple_cnn(input_shape,num_classes)
model.summary()
```

now about optimization. what optimizer are you using? adam, sgd? adam is generally a good starting point, but maybe you should try another optimizer for your problem. the learning rate is super critical too, is it too high, too low? if it's too high, it'll bounce around and never really converge to a minimum. too low and it will take too long to converge, and might even get stuck in a local minima. learning rate schedules can help a lot, lowering the learning rate during training. batch size is also important for the gradient descent. i find it often overlooked, smaller batch sizes, give more variability to the training updates, but the computation is faster. larger sizes are more computationally intensive, but can converge better. it's another balancing act.

consider adding regularization techniques to prevent overfitting. techniques like dropout or batch normalization help to prevent the model from memorizing the data instead of extracting general features. dropout basically turns off random neurons during training, which forces the network to learn more resilient representations. batch normalization helps to stabilize training and makes learning faster. i once had an experience where my model would just overfit and get 100% training accuracy and poor validation, a couple of dropout layers made all the difference, it actually started to generalize!

the loss function also has an impact, using the right loss is critical, especially if it is not a classification problem. for example, binary crossentropy for classification between 2 classes, categorical for more than 2, mse if it is a regression problem. each loss is different and suits different problems. make sure the selected loss matches the nature of the problem.

here is an example of a model with a regularization layer and a custom training loop, which i would recommend as a way to learn more about how the process actually works:

```python
import tensorflow as tf

# custom training
def train_step(model, images, labels, loss_fn, optimizer):
  with tf.GradientTape() as tape:
      predictions = model(images, training=True)
      loss = loss_fn(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  return loss

def create_regularized_cnn(input_shape, num_classes):
  model = tf.keras.models.Sequential([
      tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
      tf.keras.layers.MaxPooling2D((2, 2)),
      tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
      tf.keras.layers.MaxPooling2D((2, 2)),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dropout(0.5),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(num_classes, activation='softmax')
  ])
  return model

#usage:
input_shape = (256,256,3)
num_classes = 10

model = create_regularized_cnn(input_shape,num_classes)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_fn = tf.keras.losses.CategoricalCrossentropy()

# example dataset (replace with your own data)
images = tf.random.normal(shape=(64,256,256,3)) #example
labels = tf.random.uniform(shape=(64, num_classes), minval=0, maxval=1, dtype=tf.float32) # example

# training loop
epochs = 10
for epoch in range(epochs):
    loss = train_step(model, images, labels, loss_fn, optimizer)
    print(f"epoch: {epoch}  loss {loss}")

```

i know this feels like a lot of things to check, but honestly, it's just how it is with deep learning. it's a process of trying things out, monitoring the results, and adjusting your approach. it's not an exact science, it's more of an art form sometimes. oh, and one last thing: do not trust the training loss only, the validation loss is what truly matters for your network's performance on unseen data. training loss will always decrease but validation is the one that helps you know if your network is overfitting.

and finally, don't be afraid to try things out. that is the beauty of it, experiment, change parameters, try different stuff, maybe even grab some coffee. and if you hit a wall, it's better to ask the community for help (like you're doing now), you might be missing something very obvious that someone else spots. and, oh, one more thing, if all else fails, check if you plugged your monitor into the gpu rather than your motherboard, i have done that more than once. happens to the best of us.
