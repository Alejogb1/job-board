---
title: "How can Keras Functional API be used with TFDS datasets?"
date: "2025-01-30"
id: "how-can-keras-functional-api-be-used-with"
---
The inherent flexibility of the Keras Functional API, combined with the structured data loading capabilities of TensorFlow Datasets (TFDS), allows for building complex machine learning models directly from curated datasets, streamlining experimentation and production workflows. I’ve used this pattern extensively in building sequence-to-sequence models for text summarization, and the integration, once understood, becomes quite intuitive. The key lies in understanding how TFDS produces `tf.data.Dataset` objects and how these objects seamlessly plug into the functional API.

At its core, the Keras Functional API empowers you to build models as directed acyclic graphs, where layers are functions that operate on tensors and the model is a mapping from an input tensor to an output tensor. The inputs for such a model are typically `tf.Tensor` instances, or more commonly in practice, batches of tensors. TFDS, in turn, yields `tf.data.Dataset` instances that, when iterated over, provide these tensor batches. Therefore, the crux of the matter is to ensure the shape and type of data provided by a TFDS dataset aligns with what a Keras Functional model expects as input.

The typical workflow involves three core steps: acquiring and preparing the TFDS dataset, defining the Keras model using the Functional API, and finally, connecting the dataset to the model for training, validation, or prediction.

Let’s begin with a conceptual illustration and then move on to some specific code. Imagine you are working with a toy TFDS dataset containing pairs of sequences – an input sequence of integers representing tokenized text, and a target sequence also of integers. The first step is to load this dataset and prepare it for the network. This preparation may include batching, shuffling, and possibly data transformations such as padding. Once we have the processed `tf.data.Dataset`, we can build a simple sequence-to-sequence model with the functional API, specifying the input layer to expect the shape of data produced by the dataset. Finally, we compile this model and train it using the prepared TFDS dataset.

The critical component in establishing this link is the `tf.data.Dataset`’s ability to provide batched data in a format that mirrors the model's expected input structure.

Now, let's look at specific examples:

**Example 1: A Simple Classification Task**

Suppose we have a TFDS dataset with image classifications. Let's simplify by assuming the images are 28x28 grayscale and the labels are integers. Here's how we’d proceed:

```python
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model

# 1. Load and Prepare TFDS Dataset
dataset, info = tfds.load('mnist', with_info=True, as_supervised=True)

def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

train_dataset = dataset['train'].map(preprocess).batch(32).shuffle(100)
val_dataset = dataset['test'].map(preprocess).batch(32)

# 2. Define the Model (Functional API)
input_shape = (28, 28, 1)  # Grayscale images
inputs = Input(shape=input_shape)
x = Conv2D(32, (3, 3), activation='relu')(inputs)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
x = Dense(10, activation='softmax')(x)
model = Model(inputs=inputs, outputs=x)

# 3. Train the Model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_dataset, epochs=10, validation_data=val_dataset)
```

In this example, `tfds.load('mnist', ...)` fetches the MNIST dataset as a `tf.data.Dataset`. We then define a `preprocess` function to normalize image pixel values, then batch and shuffle the training set, and batch the validation set. The model is constructed using the Keras Functional API: an `Input` layer specifies the expected input tensor shape, followed by convolutional, pooling, flattening, and dense layers. We compile the model and train it directly using the batched TFDS datasets. Critically, the `input_shape` in the `Input` layer aligns with the shape of the image data produced by the TFDS dataset after preprocessing, namely (28, 28, 1).  This alignment is essential for proper model execution.

**Example 2: Handling Text Data with Sequences**

Let’s consider a text classification scenario. We will assume the text is encoded into integer sequences. Here’s a modified example illustrating a different data format and model:

```python
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.layers import Input, Embedding, GlobalAveragePooling1D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. Load and Prepare TFDS Dataset (simplified; assumes existing tokenized data)
# Assuming a custom dataset is loaded with tokenized sentences and labels.
# Let's mock the dataset for demonstration. In reality, you would load data
# from TFDS and encode sentences to integers.
class MockTextDataset:
    def __init__(self, num_samples, vocab_size, max_len):
        self.num_samples = num_samples
        self.vocab_size = vocab_size
        self.max_len = max_len
    
    def _generate_sequence(self):
        return tf.random.uniform((self.max_len,), minval=0, maxval=self.vocab_size, dtype=tf.int32)
    
    def _generate_label(self):
        return tf.random.uniform((), minval=0, maxval=2, dtype=tf.int32) # binary labels

    def __len__(self):
      return self.num_samples

    def __getitem__(self, idx):
      return (self._generate_sequence(), self._generate_label())

def create_dataset(mock_dataset, batch_size, max_len):
    def pad_sequence(sequence, label):
        padded_sequence = pad_sequences([sequence.numpy()], maxlen=max_len, padding='post', dtype='int32')
        return padded_sequence[0], label

    dataset = tf.data.Dataset.from_generator(lambda: (mock_dataset[i] for i in range(len(mock_dataset))),
        output_types=(tf.int32, tf.int32)).map(pad_sequence)
    
    return dataset.batch(batch_size).shuffle(len(mock_dataset))

vocab_size = 10000
max_len = 50
num_samples = 1000
batch_size = 32
mock_data = MockTextDataset(num_samples, vocab_size, max_len)
train_dataset = create_dataset(mock_data, batch_size, max_len)

# 2. Define the Model (Functional API)
input_shape = (max_len,)  # Sequences of integers
inputs = Input(shape=input_shape)
x = Embedding(input_dim=vocab_size, output_dim=128)(inputs)
x = GlobalAveragePooling1D()(x)
x = Dense(1, activation='sigmoid')(x)  # Binary Classification
model = Model(inputs=inputs, outputs=x)

# 3. Train the Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_dataset, epochs=5)
```

Here, our input dataset consists of tokenized sentences which we padded. The `input_shape` of the `Input` layer reflects the shape of the padded sequences produced by `pad_sequences` inside the `create_dataset` function. The model now uses an embedding layer to translate the integer-encoded text sequences into dense vectors, followed by global average pooling and finally a dense layer for the binary classification task. Again, the alignment between the dataset's data shape after batching and the `input_shape` is paramount.

**Example 3: Handling Multiple Inputs**

Lastly, consider a scenario where you have multiple input sources. Perhaps you have both text and numerical data. Here's how we'd handle it:

```python
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Embedding, GlobalAveragePooling1D, Dense, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# 1. Mock TFDS Dataset
class MockMultipleInputDataset:
    def __init__(self, num_samples, vocab_size, max_len, numerical_dim):
        self.num_samples = num_samples
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.numerical_dim = numerical_dim

    def _generate_sequence(self):
        return tf.random.uniform((self.max_len,), minval=0, maxval=self.vocab_size, dtype=tf.int32)
    
    def _generate_numerical(self):
        return tf.random.normal((self.numerical_dim,), dtype=tf.float32)

    def _generate_label(self):
        return tf.random.uniform((), minval=0, maxval=2, dtype=tf.int32) # binary labels
    
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return (self._generate_sequence(), self._generate_numerical()), self._generate_label()

def create_multiple_input_dataset(mock_dataset, batch_size, max_len, numerical_dim):
    def process_data(sequence, numerical, label):
        padded_sequence = pad_sequences([sequence.numpy()], maxlen=max_len, padding='post', dtype='int32')
        return (padded_sequence[0], numerical), label

    dataset = tf.data.Dataset.from_generator(lambda: (mock_dataset[i] for i in range(len(mock_dataset))),
    output_types=((tf.int32, tf.float32), tf.int32)).map(process_data)
    
    return dataset.batch(batch_size).shuffle(len(mock_dataset))

vocab_size = 10000
max_len = 50
numerical_dim = 10
num_samples = 1000
batch_size = 32
mock_multiple_input = MockMultipleInputDataset(num_samples, vocab_size, max_len, numerical_dim)
train_dataset = create_multiple_input_dataset(mock_multiple_input, batch_size, max_len, numerical_dim)


# 2. Define the Model (Functional API)
text_input_shape = (max_len,)
numerical_input_shape = (numerical_dim,)

text_inputs = Input(shape=text_input_shape, name='text_input')
numerical_inputs = Input(shape=numerical_input_shape, name='numerical_input')


x_text = Embedding(input_dim=vocab_size, output_dim=128)(text_inputs)
x_text = GlobalAveragePooling1D()(x_text)
x_numerical = Dense(128, activation='relu')(numerical_inputs)

merged = concatenate([x_text, x_numerical])
x = Dense(1, activation='sigmoid')(merged)

model = Model(inputs=[text_inputs, numerical_inputs], outputs=x)

# 3. Train the Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_dataset, epochs=5)
```

In this final example, the data provided by the dataset is a tuple of two tensors – a padded sequence of integers and numerical data. We define two separate `Input` layers, one for each data stream, and then process them independently using an embedding and dense layer. We use `concatenate` to merge the output of these layers, before going into the final prediction layer. The critical aspect here is that the `input` argument of the `Model` class takes a *list* of `Input` layers corresponding to the structure of the data emitted by the dataset.

For further information, I recommend consulting these resources: The TensorFlow documentation on the Keras Functional API, the TensorFlow Datasets API documentation, and any comprehensive tutorial materials focused on integrating these two APIs.  Pay particular attention to sections on dataset preparation, custom dataset creation, and handling different data structures with the functional API. Careful examination of worked examples will be invaluable for your own projects.
