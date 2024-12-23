---
title: "How can Keras Functional API be used with TFDS datasets?"
date: "2024-12-23"
id: "how-can-keras-functional-api-be-used-with-tfds-datasets"
---

Let's dive right into this; I've spent a good chunk of my career knee-deep in TensorFlow and Keras, and the integration with TensorFlow Datasets (TFDS) is a subject I've revisited numerous times, especially when scaling up model training. The beauty of the Keras Functional API paired with TFDS is how cleanly it handles data pipelines, particularly for complex input structures. Let me unpack that a bit, and then we can get to some concrete code examples.

The Functional API, in essence, provides a way to define your models as directed acyclic graphs of layers, which contrasts with the more sequential, linear approach of the Sequential API. This offers immense flexibility when your model needs multiple inputs, outputs, or when it involves shared layers. When you’re pulling datasets via TFDS, it's rarely just a homogenous tensor feeding directly into a dense network. You often encounter structured data: images paired with text, various modalities of sensor data, or even multiple related inputs like auxiliary regression targets. Here, the Functional API shines.

TFDS gives us datasets as `tf.data.Dataset` objects, which are highly efficient for handling large volumes of data. These datasets can be further manipulated using the `tf.data` API to batch, shuffle, pre-process, and generally optimize data loading. This is where the real integration happens. We are not simply passing raw data into our model; rather, we’re creating a robust and optimized pipeline feeding our functional model.

Now, the crucial point is how we structure this data within our `tf.data.Dataset` so it coheres with the input expected by our Functional model. Generally, for a simple case with a single input and single output, we would prepare our dataset to return tuples of (input_tensor, output_tensor). However, with the Functional API’s flexibility, we often need to structure this data as dictionaries. Think of it like this: each entry in our data pipeline, as provided by TFDS and processed by our `tf.data.Dataset`, needs to match up with the input layers that are defined in the model.

Let’s look at some working examples.

**Example 1: Single Input Image Classification**

This is the simplest case, but it demonstrates the basic principle. Suppose we are using a subset of the CIFAR10 dataset, fetched through TFDS:

```python
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, Model, Input

# Fetch the CIFAR10 dataset, let's use a small split for this example
(ds_train, ds_test), ds_info = tfds.load('cifar10', split=['train[:1000]', 'test[:200]'], with_info=True)

def preprocess_image(example):
    image = tf.image.convert_image_dtype(example['image'], dtype=tf.float32)
    image = tf.image.resize(image, [32, 32]) # Ensure all images are of the same shape
    label = example['label']
    return image, label

# Build the tf.data pipeline
BATCH_SIZE = 32
ds_train_prep = ds_train.map(preprocess_image).shuffle(100).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
ds_test_prep = ds_test.map(preprocess_image).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Define the functional model
inputs = Input(shape=(32, 32, 3), name='image_input')
x = layers.Conv2D(32, 3, activation='relu')(inputs)
x = layers.MaxPooling2D()(x)
x = layers.Flatten()(x)
outputs = layers.Dense(10, activation='softmax', name='output_probabilities')(x)

model = Model(inputs=inputs, outputs=outputs)

# Compile and train the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(ds_train_prep, epochs=5, validation_data = ds_test_prep)
```
Here, the `preprocess_image` function takes the image and label from the TFDS dataset. It converts and resizes the image, and then packages it with the label as the standard (input, output) tuple required by the Keras `fit` method. The functional API part constructs the model graph and is straightforward since it has one input and one output node.

**Example 2: Multiple Inputs - Text and Images**

Now let’s get a little more involved. Imagine using a dataset where both images and associated captions are inputs to our model. We'll synthesize a simplified version of a multi-modal scenario since we don’t have a straightforward multi-modal TFDS dataset for demonstration.

```python
import tensorflow as tf
from tensorflow.keras import layers, Model, Input

# Simulate a dataset using tf.data.Dataset
images = tf.random.normal(shape=(100, 64, 64, 3))
texts = tf.random.uniform(shape=(100, 10), minval=0, maxval=1000, dtype=tf.int32)  # Placeholder for word indices
labels = tf.random.uniform(shape=(100,), minval=0, maxval=10, dtype=tf.int32)

def create_dataset_dict(images, texts, labels):
  data = []
  for i in range(len(images)):
     data.append({'image': images[i], 'text': texts[i], 'label': labels[i]})
  ds = tf.data.Dataset.from_tensor_slices(data)
  return ds

ds_train_multimodal = create_dataset_dict(images, texts, labels).shuffle(100).batch(32).prefetch(tf.data.AUTOTUNE)


# Define the functional model
input_image = Input(shape=(64, 64, 3), name='image_input')
x = layers.Conv2D(32, 3, activation='relu')(input_image)
x = layers.MaxPooling2D()(x)
image_embedding = layers.Flatten()(x)

input_text = Input(shape=(10,), name='text_input')
y = layers.Embedding(input_dim=1000, output_dim=32)(input_text)
y = layers.LSTM(32)(y)
text_embedding = y

merged = layers.concatenate([image_embedding, text_embedding])

z = layers.Dense(64, activation = 'relu')(merged)
outputs = layers.Dense(10, activation='softmax', name = 'output_probabilities')(z)


model = Model(inputs={'image_input': input_image, 'text_input': input_text}, outputs=outputs)


# Compile and train the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

def prepare_multimodal(element):
    return {'image_input': element['image'], 'text_input': element['text']}, element['label']
ds_train_multimodal_prepared = ds_train_multimodal.map(prepare_multimodal)


model.fit(ds_train_multimodal_prepared, epochs=5)
```

Here, the critical change is that our dataset now returns a *dictionary* of inputs, matching the named inputs in our Functional model. In the `prepare_multimodal` mapping, we extract the `image` and `text` tensors from our dataset entry and put them into a dictionary, which will match the named `input` layers defined in the model. The outputs remain as a simple label tensor.

**Example 3: Multiple Outputs - Regression and Classification**

Finally, let’s consider a case where our model produces multiple outputs, say, a classification label *and* a regression value. This is common in tasks like pose estimation where you may need to predict both class labels and coordinates. We again synthesize some example data for clarity.

```python
import tensorflow as tf
from tensorflow.keras import layers, Model, Input

# Simulate a dataset using tf.data.Dataset
images = tf.random.normal(shape=(100, 64, 64, 3))
class_labels = tf.random.uniform(shape=(100,), minval=0, maxval=10, dtype=tf.int32)
regression_values = tf.random.normal(shape=(100, 2))

def create_dataset_multioutput(images, class_labels, regression_values):
  data = []
  for i in range(len(images)):
     data.append({'image': images[i], 'class_label': class_labels[i], 'regression_value': regression_values[i]})
  ds = tf.data.Dataset.from_tensor_slices(data)
  return ds

ds_train_multioutput = create_dataset_multioutput(images, class_labels, regression_values).shuffle(100).batch(32).prefetch(tf.data.AUTOTUNE)


# Define the functional model
inputs = Input(shape=(64, 64, 3), name='image_input')
x = layers.Conv2D(32, 3, activation='relu')(inputs)
x = layers.MaxPooling2D()(x)
x = layers.Flatten()(x)

class_output = layers.Dense(10, activation='softmax', name = 'class_probabilities')(x)
regression_output = layers.Dense(2, activation='linear', name = 'regression_coords')(x)

model = Model(inputs=inputs, outputs={'class_probabilities': class_output, 'regression_coords': regression_output})


# Compile and train the model
model.compile(optimizer='adam', loss={'class_probabilities': 'sparse_categorical_crossentropy', 'regression_coords': 'mse'},
            metrics={'class_probabilities': 'accuracy', 'regression_coords': 'mse'})
def prepare_multioutput(element):
    return element['image'], {'class_probabilities': element['class_label'], 'regression_coords': element['regression_value']}

ds_train_multioutput_prepared = ds_train_multioutput.map(prepare_multioutput)

model.fit(ds_train_multioutput_prepared, epochs=5)
```

Here, our data still comprises one input tensor (image), but the dataset now produces *dictionary* as its second element, mapping to multiple *output* layers of the model. Notice how we also specify the losses and metrics as dictionaries, aligning with each of the named output layers. The `prepare_multioutput` function constructs our target output in the dictionary structure.

In each case, the key takeaway is consistency: your `tf.data.Dataset` output structure must match the named inputs and outputs defined in your Functional model. The Functional API is extremely powerful for handling complex architectures and data input strategies that would be cumbersome with a simple sequential approach, and its compatibility with TFDS makes it ideal for working with massive datasets.

For more in-depth learning, I recommend looking at the official TensorFlow documentation on the Functional API, particularly the sections on multi-input and multi-output models. Also, the 'Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow' by Aurélien Géron has a very strong practical approach to using Keras and tensorflow in general. Finally, for a more theoretical understanding of data pipelines, I would recommend delving into papers on the `tf.data` API optimization strategies, but start with the Tensorflow guides as it is comprehensive. I've personally found these to be invaluable resources in my own work, and I hope they are helpful in yours too.
