---
title: "How does Keras' `Sequence` class work?"
date: "2025-01-30"
id: "how-does-keras-sequence-class-work"
---
The core purpose of Keras’ `Sequence` class is to decouple data loading from model training, enabling efficient handling of large datasets that exceed available memory. I’ve spent considerable time optimizing deep learning pipelines and found this decoupling to be paramount for scalability. Essentially, `Sequence` acts as a Python generator, providing batches of data on demand, preventing the entire dataset from residing in RAM simultaneously. This contrasts sharply with loading all training data into memory upfront, a method that quickly becomes infeasible with sizeable datasets, especially images, audio, or large text corpora.

Here's how it works: you subclass `keras.utils.Sequence` and override three key methods: `__len__`, `__getitem__`, and optionally `on_epoch_end`. `__len__` defines the number of batches in your dataset, not the number of individual samples. This method is called by the Keras training loop to determine the total number of iterations per epoch. `__getitem__`, the workhorse of the class, takes an index (representing a batch number) and returns a tuple of (inputs, targets). The inputs and targets themselves are typically NumPy arrays or tensors. Crucially, `__getitem__` only loads the data required for that specific batch, thus saving memory. Finally, `on_epoch_end` is invoked at the end of each epoch, allowing for actions such as shuffling the data indices before the next epoch begins.

Let's illustrate with a concrete example of a `Sequence` class designed for image data:

```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import os

class ImageSequence(keras.utils.Sequence):

    def __init__(self, image_dir, labels, batch_size, image_size=(224, 224)):
        self.image_dir = image_dir
        self.labels = labels
        self.batch_size = batch_size
        self.image_size = image_size
        self.image_paths = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir)]
        self.indices = np.arange(len(self.image_paths))

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_images = []
        batch_targets = []

        for i in batch_indices:
            image_path = self.image_paths[i]
            img = Image.open(image_path)
            img = img.resize(self.image_size)
            img_array = np.array(img) / 255.0  # Normalization
            batch_images.append(img_array)
            filename = os.path.basename(image_path)
            #Assume a simplified naming convention label_imageid.jpg where label is a 0 or 1
            label = int(filename.split("_")[0])
            batch_targets.append(label)

        return np.array(batch_images), np.array(batch_targets)

    def on_epoch_end(self):
        np.random.shuffle(self.indices)
```

In this example, I've defined `ImageSequence` to read images from a directory. The `__init__` method initializes the image paths, labels, batch size, and other necessary parameters, also creating the sequence of file indices. `__len__` calculates how many batches there will be. `__getitem__` takes an index, determines which image files belong to the corresponding batch, loads, resizes, and normalizes the images, and then retrieves their corresponding labels. Crucially, it only loads the image files needed for the current batch. `on_epoch_end` shuffles the order of the images ensuring that the order of the training data changes with each new epoch.

Here's a second example, this time processing text data. Suppose we have sentences and corresponding labels:

```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


class TextSequence(keras.utils.Sequence):
    def __init__(self, sentences, labels, batch_size, max_len, vocab_size):
        self.sentences = sentences
        self.labels = labels
        self.batch_size = batch_size
        self.max_len = max_len
        self.tokenizer = Tokenizer(num_words=vocab_size, oov_token="<unk>")
        self.tokenizer.fit_on_texts(self.sentences)
        self.indices = np.arange(len(self.sentences))

    def __len__(self):
        return int(np.ceil(len(self.sentences) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_sentences = [self.sentences[i] for i in batch_indices]
        batch_labels = [self.labels[i] for i in batch_indices]

        tokens = self.tokenizer.texts_to_sequences(batch_sentences)
        padded_tokens = pad_sequences(tokens, maxlen=self.max_len, padding='post')
        batch_labels = np.array(batch_labels)

        return padded_tokens, batch_labels

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

```
In this example, `TextSequence` takes lists of sentences and labels as input. The `__init__` method initializes the tokenizer and fits it to the vocabulary present within the supplied sentences, and also creates the shuffled index list. `__getitem__` tokenizes the sentences in the current batch using the fitted tokenizer, pads the token sequences so they are of consistent length, and then retrieves corresponding labels. Again, the data is processed in batches to avoid memory overload. `on_epoch_end` is once more used to shuffle the indices of the data.

A crucial distinction here from the previous image example, is that we are applying a pre-processing step that is reliant on the entire dataset, namely the tokenizer fitting. This is a typical requirement, but it also is an example of something that can be computationally demanding and should be done carefully to ensure efficiency. In this case, the tokenization occurs just before the batches are created.

Finally, let’s look at a more complex example that returns three inputs and one target - a common occurrence in multi-modal model scenarios:
```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import os


class MultiInputSequence(keras.utils.Sequence):
    def __init__(self, image_dir, text_dir, labels, batch_size, image_size=(224, 224), max_text_length=50):
        self.image_dir = image_dir
        self.text_dir = text_dir
        self.labels = labels
        self.batch_size = batch_size
        self.image_size = image_size
        self.max_text_length = max_text_length
        self.image_paths = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir)]
        self.text_paths = [os.path.join(text_dir, filename) for filename in os.listdir(text_dir)]
        self.indices = np.arange(len(self.image_paths))

        assert len(self.image_paths) == len(self.text_paths), "Image and text dataset lengths do not match"

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_images = []
        batch_texts = []
        batch_targets = []

        for i in batch_indices:
            image_path = self.image_paths[i]
            text_path = self.text_paths[i]

            img = Image.open(image_path)
            img = img.resize(self.image_size)
            img_array = np.array(img) / 255.0
            batch_images.append(img_array)

            with open(text_path, 'r') as f:
                text = f.read()
            words = text.split()[:self.max_text_length]
            padded_text = ' '.join(words).ljust(self.max_text_length)
            batch_texts.append(np.array([ord(c) for c in padded_text]))

            filename = os.path.basename(image_path)
            label = int(filename.split("_")[0]) #assume same as before
            batch_targets.append(label)


        return [np.array(batch_images), np.array(batch_texts), np.random.rand(len(batch_indices), 100)], np.array(batch_targets)

    def on_epoch_end(self):
        np.random.shuffle(self.indices)
```
This `MultiInputSequence` class demonstrates handling heterogeneous data types for multi-modal models. The `__getitem__` method now loads images and text files in the batch, performing preprocessing on both, creating padded arrays of text with the specified `max_text_length`, and concatenates the outputs into an appropriate multi-input format, along with a random third input. As can be seen, this approach can accommodate more complex data structures without significant alterations to the core `Sequence` class.

When I consider these examples within the context of my own experiences, one benefit that is not immediately obvious is how this architecture naturally lends itself to distributed training. This allows each worker to load only a partition of the dataset rather than the whole thing.

To further explore, I would suggest examining the official Keras documentation, which includes details on the base `Sequence` class, as well as advanced topics such as data augmentation strategies that can be incorporated within the custom `Sequence` classes. Further investigation into how these strategies can be used in conjunction with the TensorFlow Data API for highly optimized data loading would be beneficial, alongside exploration of custom subclassing to create prefetching for even greater efficiency in data ingestion, specifically during large-scale, distributed training operations. Studying these topics allows for a deeper understanding of data pipelines.
