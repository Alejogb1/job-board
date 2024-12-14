---
title: "How to use a CVAE on a CIFAR10 Dataset?"
date: "2024-12-14"
id: "how-to-use-a-cvae-on-a-cifar10-dataset"
---

alright, so you’re looking to get a conditional variational autoencoder, a cvae, up and running on the cifar10 dataset. i’ve messed around with this kind of setup before, it's a fairly common challenge when you want more control over the generative process, not just random samples. basically, you want to guide the image generation based on the class labels, so we're not just spitting out random-looking "things". been there, done that, multiple times in fact. let me walk you through what i’ve found works well.

the core idea here is that we’re going to modify the standard vae architecture to include label information. the standard vae encodes input data, in this case images, into a latent space, then decodes that latent representation back to the image. the cvae, however, conditions both the encoder and decoder on the class label. this means the latent representation also has class information encoded within it. this gives you the power to sample from the latent space conditioned by a given class, so you get control over the output images being generated.

first, we’ll handle the cifar10 data. it’s a straightforward dataset available directly from tensorflow or pytorch, whichever you're more comfortable with. i lean towards tensorflow personally, but the general principles apply to both. the dataset provides 60,000 32x32 colour images spanning 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. i'm assuming you’re already familiar with basic image handling. i won't go deep into the basics of loading the data, preprocessing, since that’s a prerequisite for this sort of project, but i can cover the parts relevant to the cvae.

the trick is to make sure your label is embedded in the input to both the encoder and decoder. this can be achieved using one-hot encoding. each label becomes a vector where the class index is 1 and others are 0. we concatenate this with the images going into the encoder, and similarly, we input it as an additional signal to the decoder. this ensures that the model 'knows' which class to condition on at any time.

here's a snippet of how you might prepare the data with tensorflow, the one i like best:

```python
import tensorflow as tf
import numpy as np

def prepare_cifar10_data(batch_size=32):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # normalize pixel values to [0, 1] range
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    # one-hot encode the labels
    num_classes = 10
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    # create the datasets, include batching
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

    return train_dataset, test_dataset, num_classes

train_data, test_data, num_classes = prepare_cifar10_data()
print(f"number of classes: {num_classes}")
for images, labels in train_data.take(1):
    print("image batch shape:", images.shape)
    print("labels batch shape:", labels.shape)
```

this function takes care of loading the dataset, normalizing pixel values to between 0 and 1, converting the labels to one-hot vectors and creating a batched dataset, ready for consumption by the model. notice the `take(1)` in the print part of the snippet i added to inspect the tensors that we will be feeding to our network. the output of this print shows the shape of the image tensors, a batch of 32 images with 32x32 pixel shape and 3 channels (rgb) and the label tensors which are a batch of 32 one-hot encoded vectors each with 10 entries since we have 10 classes in cifar10 dataset.

next is the actual cvae architecture. we will define encoder and decoder models that integrate the label information. the encoder takes in the concatenated image and label vector, squashes this down to a lower-dimensional latent space, and produces the mean and the standard deviation of the latent distribution. the decoder then receives a sample from this latent space *and* the label vector, using both to generate an image. a simple network with convolutional and dense layers does the trick, as long as you take care about dimensionality matching and non-linearities.

here's a basic cvae implementation with tensorflow:

```python
import tensorflow as tf
from tensorflow.keras import layers

class CVAE(tf.keras.Model):
    def __init__(self, latent_dim, num_classes):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # encoder model
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(32, 32, 3 + num_classes)), # input shape: image + one-hot label
            layers.Conv2D(32, 3, activation='relu', padding='same'),
            layers.MaxPool2D((2, 2)),
            layers.Conv2D(64, 3, activation='relu', padding='same'),
            layers.MaxPool2D((2, 2)),
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dense(latent_dim * 2) # mean and log variance concatenated
        ])

        # decoder model
        self.decoder = tf.keras.Sequential([
            layers.Input(shape=(latent_dim + num_classes,)), # input shape: latent + one-hot label
            layers.Dense(256, activation='relu'),
            layers.Dense(8*8*64, activation='relu'),
            layers.Reshape((8, 8, 64)),
            layers.Conv2DTranspose(64, 3, activation='relu', padding='same', strides=2),
            layers.Conv2DTranspose(32, 3, activation='relu', padding='same', strides=2),
            layers.Conv2DTranspose(3, 3, activation='sigmoid', padding='same')
        ])

    def encode(self, x, label):
        x = tf.concat([x, tf.tile(tf.expand_dims(label, axis=[1, 2]), [1, 32, 32, 1])], axis=-1)
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * 0.5) + mean

    def decode(self, z, label):
        z = tf.concat([z, label], axis=1)
        return self.decoder(z)

    def call(self, x, label):
        mean, logvar = self.encode(x, label)
        z = self.reparameterize(mean, logvar)
        return self.decode(z, label), mean, logvar


    def generate_image(self, z, label):
        #this method does not generate new samples instead generates the image according to the label vector.
        return self.decode(z, label)
```

this code creates the basic structure. we use convolutional and dense layers, and we explicitly concatenate the one-hot labels in both the encoder's input and in the decoder's input. pay close attention to how we are handling dimensionality, and how the labels vectors are repeated in the correct shape so we can concatenate them with the feature map tensors. notice also the method `generate_image()` will be very useful when we want to generate images with a given label that we specify.

now for the loss function and training loop. the loss of the vae is made of two parts, the reconstruction loss which measures how good the image generated is to the original image and the kullback-leibler divergence which tries to enforce the distribution of the latent space to be gaussian.

```python
def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(-.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi), axis=raxis)

optimizer = tf.keras.optimizers.Adam(1e-3)

@tf.function
def train_step(model, x, label, optimizer):
    with tf.GradientTape() as tape:
        x_hat, mean, logvar = model(x, label)
        reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.keras.losses.binary_crossentropy(x, x_hat), axis=[1,2,3]))
        kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + logvar - tf.square(mean) - tf.exp(logvar), axis=1))
        loss = reconstruction_loss + kl_loss

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

latent_dim=128
num_epochs = 20
model = CVAE(latent_dim, num_classes)

for epoch in range(num_epochs):
    for batch, (images, labels) in enumerate(train_data):
        train_step(model, images, labels, optimizer)
        if batch % 100 == 0:
            print(f"epoch: {epoch}, batch: {batch}")
```

the training loop iterates through the dataset, computes the loss and optimizes the model using backpropagation. the kl divergence is the penalty term that helps the latent space become more similar to a normal gaussian distribution, which is good because this makes it easier to sample from it when we want to generate data.

when you're working with cvaes or vaes in general, don't expect the best image quality from a simple implementation. the images can be pretty blurry, and they may not perfectly match a class. that's part of what makes them generative models, and a different story is when you want high-quality and sharp generated images. but it gets the job done when you want to explore the capabilities of the latent space.

now, when you want to sample, it’s as easy as sampling a vector `z` from a gaussian distribution and feeding it to the decoder with a specific label. i can recommend some resources that will be very useful. in deep learning there are two papers which are a must read. the paper 'auto-encoding variational bayes' by kingma and welling (2013) is the foundational work on vaes. it's heavy on math, but it’s crucial to understand the theory behind it. and then ‘conditional generative adversarial nets’ by mirza and osindero (2014) on conditional generation, is also very important. if you want a deeper dive in the math for vaes, i would recommend the book ‘deep learning’ by goodfellow, bengio and courville, it’s a bible for deep learning concepts.

i remember my first time dealing with variational autoencoders, it was an experience. i was getting nan values and it turned out that i was not handling the numerical stability of the logarithm function which is part of the kl divergence. it's always some numerical issue, always... it can make you want to become a plumber instead of dealing with deep learning models... well, almost.

it's also a good practice to check how the latent space has been organized by performing some visualization. a very common way to accomplish that is to feed images from different classes into the encoder and plot the mean value of the resulting latent vector using for example tsne algorithm. this can give insight into whether the latent space has properly separated the various classes.

that’s pretty much the core of getting a cvae running on cifar10. it requires some tweaks and experimentation to get the results you are looking for. it is also a good idea to train it for more epochs or to try some architecture variations, but with this code you are good to go! give it a try.
