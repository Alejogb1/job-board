---
title: "How to use a Keras CNN autoencoder for hi-res Images?"
date: "2024-12-15"
id: "how-to-use-a-keras-cnn-autoencoder-for-hi-res-images"
---

alright, so, you're diving into keras cnn autoencoders with high-resolution images, eh? i've been there, believe me. it's not always straightforward, and it’s easy to hit some snags. i remember back in the early 2010s when i was trying to build a denoising tool for old satellite imagery; i ran into similar problems trying to feed giant tiffs into the system.

first thing's first, the sheer size of hi-res images will cause issues. your gpu, unless you're running on some super-powered cluster, will probably throw up out-of-memory errors faster than you can say "convolution". it’s important to preprocess those images before feeding them into the network.

the primary problem lies in the amount of memory they require. a single large image, say a 4000x4000 pixel colour image (3 channels) at 32-bit float precision, can already take up close to 200 mb of memory. multiply that by batch size and the intermediate feature maps within the network, and suddenly your vram is crying.

let's go over how to tackle that.

**image loading and batching:**

the first crucial step is how you load your images. don't load all of them into memory at once. instead, you'll want to use a data generator, keras provides an easy way to do that with `tf.keras.utils.image_dataset_from_directory`. but we can also build our own.

here is some example code that loads data from a directory, it takes in a path to images and batch size, and it generates batches of the correct shape and type. it uses `imageio` that is very fast to read images from disk.

```python
import os
import numpy as np
import tensorflow as tf
import imageio.v3 as iio

def create_image_data_generator(image_dir, batch_size, target_size=(256, 256)):
    """
    creates a tensorflow generator to yield batches of images from a folder
    Args:
        image_dir(str): path to folder containing images
        batch_size(int): desired batch size
        target_size(tuple): desired image size
    Returns:
      A tensorflow generator

    """

    image_files = [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if os.path.isfile(os.path.join(image_dir, f))
    ]
    num_files = len(image_files)

    if not num_files:
         raise ValueError(f"No image files were found in {image_dir}")

    def generator():
        i = 0
        while True: # infinite loop to keep the generator alive
            batch_images = []
            for _ in range(batch_size):

                if i >= num_files:
                   i = 0  # reset the counter if out of range
                img_path = image_files[i]
                try:
                    img = iio.imread(img_path)
                    img = tf.image.resize(img, target_size)
                    img = tf.cast(img, tf.float32) / 255.0 # cast to float32 and normalize
                    batch_images.append(img)
                    i += 1
                except Exception as e:
                     print(f"Error loading image: {img_path}. skipping this file, error {e}")
                     i += 1 # increment i even if error occurs
                     continue


            if batch_images: # if batch is not empty yield
                batch_images = np.array(batch_images)
                yield batch_images, batch_images # both input and target are the same in autoencoder
            else:
                continue  # skip empty batch



    return tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(None, target_size[0], target_size[1], 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, target_size[0], target_size[1], 3), dtype=tf.float32),
            ),
    )

# example usage:
image_directory = "path_to_your_images/" #replace with the path to your images
batch_size = 32
target_size = (256, 256)

data_generator = create_image_data_generator(image_directory, batch_size, target_size)

for images, _ in data_generator.take(3):
  print("Batch shape: ", images.shape)
```

this code generates batches on the fly, avoiding the need to load all images into memory at once. it resizes the images to the `target_size` and normalizes them to the 0-1 range. adjust the target size to match your required dimensions or the input of your network. you could also have different target sizes for training and for inference if needed.

**building the cnn autoencoder:**

now, let's talk about the autoencoder itself. a simple cnn autoencoder has two parts: the encoder that compresses the image into a latent space and a decoder that reconstructs the image from the latent space. using keras you can do this in a few lines. for high-resolution images, you might need a deeper architecture.

here is an example of the model:

```python
import tensorflow as tf
from tensorflow.keras import layers

def build_cnn_autoencoder(input_shape):
    """
    builds a simple cnn autoencoder model.

    Args:
       input_shape: a tuple representing the shape of input images
    Returns:
       tensorflow keras model
    """

    #encoder
    encoder_input = layers.Input(shape=input_shape, name='encoder_input')
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoder_input)
    x = layers.MaxPool2D((2, 2), padding='same')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPool2D((2, 2), padding='same')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    encoded = layers.MaxPool2D((2, 2), padding='same', name='encoded')(x)


    #decoder
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(input_shape[-1], (3, 3), activation='sigmoid', padding='same', name='decoded')(x)

    autoencoder = tf.keras.Model(encoder_input, decoded, name='autoencoder')
    return autoencoder



#example usage
input_shape=(256,256,3)
autoencoder = build_cnn_autoencoder(input_shape)
autoencoder.summary()
```

this model uses convolutional layers for encoding and transposed convolutional layers (upsampling2d) for decoding. i've added some max pooling layers in the encoder part and upsampling layers in the decoder to reduce and increase the size of the features. the activation function in the last layer of the decoder is a sigmoid as the images are normalized between 0 and 1. notice the use of `'same'` padding, this keeps the dimensions from shrinking at each layer, it helps to keep the shapes more symmetrical in the encoder and decoder part. you can adjust these parameters such as filters, kernel sizes, activation functions, number of layers to your specific needs.

**training the model**

training the autoencoder is where the magic happens, or where things can go south quickly, depending on your setup. you will have to use a suitable loss function, typically `mean squared error` (mse) or `binary cross-entropy`. `mse` is easier to work with for images. you can also use a different loss if needed. you compile the model using `autoencoder.compile()`. you then train using the `autoencoder.fit()` method and give it the generator we created before.

the training code will look something like this:

```python
import tensorflow as tf

# assuming 'autoencoder' and 'data_generator' are already defined

learning_rate = 0.001
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
autoencoder.compile(optimizer=optimizer, loss='mse')

# specify the steps per epoch, number of images divided by batch size
num_images = len(os.listdir(image_directory))
steps_per_epoch = num_images // batch_size

epochs=10
#train
autoencoder.fit(data_generator, steps_per_epoch=steps_per_epoch, epochs=epochs)
```

this will train the autoencoder using the generator. notice that we use steps per epoch that is calculated by the total number of images divided by the batch size. this is a typical way of using generators with keras.

**important considerations**

*   **gpu memory:** if you're still getting out-of-memory errors, try reducing the batch size, the target size or the number of filters in the layers. also try using mixed-precision training which saves memory. in keras its easily done by importing `from tensorflow.keras.mixed_precision import Policy` and creating a policy with `policy = Policy('mixed_float16')` and setting it `tf.keras.mixed_precision.set_global_policy(policy)`.
*   **normalization:** ensure your input images are normalized to a suitable range (0-1) as i have done using `tf.cast(img, tf.float32) / 255.0`, autoencoders usually work better with normalized inputs. you could also try standardizing (zero mean, unit variance).
*   **architecture:** the architecture i provide here is simple, feel free to experiment with deeper networks, different filter sizes, more sophisticated downsampling methods like strided convolutions, or skip connections in the encoder/decoder.
*   **evaluation:** monitor your loss, but also look at reconstructed images to check if the quality of the reconstructed images. there's always a trade off between how much the model compress and the quality of the reconstructions.
*   **latent space:** for autoencoders, the latent space usually is not that structured, so if you want to do something in the latent space like generating new images, variational autoencoders will probably be better than cnn autoencoders.
*   **debugging:** use `model.summary()` to check the shape of all layers and feature maps, this helps in catching any issues. also, you could monitor training in tensorboard to see the metrics in real time.

i've been using these techniques for a while now, so if you still face issues after following this just ask and i will try to help. i once spent a whole weekend debugging an oom error just because i was using an old version of tensorflow, that was not fun and my brain hurt. i had to buy a new laptop to try and fix that issue. it was an expensive weekend... haha.

for deeper understanding on autoencoders, check out "deep learning" by goodfellow et al. and for tensorflow keras specific things the official documentation is always a good resource. there's a lot of good stuff in there about data pipelines, layers and model building. also, "programming machine learning: from coding to deep learning" by enrique alonso is a good book if you like coding-centric explanations.

good luck, and let me know how it goes!
