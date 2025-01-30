---
title: "How can custom generators be used with `zip` for model training in Python?"
date: "2025-01-30"
id: "how-can-custom-generators-be-used-with-zip"
---
The inherent memory limitations of loading large datasets into RAM necessitate the use of generators, particularly when training complex models. This approach is especially relevant when dealing with multiple input sources that require parallel processing and alignment during training. The `zip` function in Python, coupled with custom generators, offers an elegant solution for such scenarios. I’ve frequently employed this technique in projects dealing with multi-modal data, such as image-text pairings or time series data with associated metadata.

The core idea revolves around creating generators that yield sequences of training data, and then using `zip` to combine these sequences into tuples suitable for model input. Instead of loading the entire dataset into memory, each generator fetches and preprocesses data on-the-fly, providing a stream of training samples. The `zip` function then takes these streams and produces an iterable that yields corresponding data from each generator at each step. This creates a combined, on-demand stream, effectively mitigating memory overloads and providing a streamlined way to handle complex data pipelines during model training. The primary benefit is that data is only loaded into memory when it is actually needed, dramatically reducing RAM requirements when dealing with large datasets.

To illustrate, consider the scenario of training a model on image and corresponding text data. I might have two separate data sources, images stored on disk and text data in a JSON file. Instead of loading both into memory, I’d create two generators; one for images and one for text. These generators would individually manage loading and preprocessing of their respective data. The `zip` function would then be used to pair each image with its corresponding text, yielding a tuple ready for model training.

Here’s the first code example, demonstrating a very basic setup:

```python
import os
import json
from typing import Generator, Tuple
from PIL import Image
import numpy as np

def image_generator(image_dir: str) -> Generator[np.ndarray, None, None]:
    """
    Generator to yield images from a directory.

    Args:
        image_dir: Path to the directory containing image files.

    Yields:
        numpy.ndarray: A single image as a numpy array.
    """
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_dir, filename)
            try:
                 img = Image.open(image_path).convert('RGB')
                 img_array = np.array(img)
                 # Example preprocessing: Resizing
                 img_array = np.resize(img_array,(64,64,3)) # assuming consistent resizing
                 yield img_array
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue

def text_generator(text_file: str) -> Generator[str, None, None]:
    """
    Generator to yield text from a JSON file.

    Args:
        text_file: Path to the JSON file.

    Yields:
        str: A single text string.
    """
    with open(text_file, 'r') as f:
        data = json.load(f)
        for item in data:
            if 'text' in item:
                yield item['text']

def data_generator(image_dir: str, text_file: str) -> Generator[Tuple[np.ndarray, str], None, None]:
   """
   Combines image and text generators using zip.

   Args:
        image_dir: Path to the directory containing image files.
        text_file: Path to the JSON file.

    Yields:
        Tuple[np.ndarray, str]: A tuple containing an image array and a text string.
   """
   image_gen = image_generator(image_dir)
   text_gen = text_generator(text_file)
   yield from zip(image_gen, text_gen)


# Example usage:
# This part is assumed to be defined and configured by the user, or within another module.
# image_directory = 'path/to/images'
# json_file = 'path/to/text.json'
# for image, text in data_generator(image_directory, json_file):
#       print(f"Image shape: {image.shape}, Text sample: {text[:20]}...")
#       # Training step: feed image and text to model
#      # break #for illustrative purposes only.
```

In this first example, `image_generator` yields preprocessed image arrays from a specified directory, while `text_generator` yields text strings from a JSON file, assuming a consistent format where each item has a 'text' key. The `data_generator` uses `zip` to combine these two generator outputs, yielding tuples of (image array, text string). The example usage section demonstrates the core concept of using these generators in training. Error handling using a `try-except` block is integrated into the image processing to address potential file errors without halting training.

The key is the use of `yield from zip(image_gen, text_gen)` in `data_generator`. This leverages Python's generator delegation syntax to produce a new generator which yields the combined output of the individual generators via zip. The `zip` function will produce a tuple at each iteration, taking one element from each of its arguments until the shortest argument is exhausted. The code snippet demonstrates the core functionality, where the training loop is implicitly called by the example use case.

However, in practical situations, data loading is often more complex and requires additional transformations. Furthermore, a naive generator could be inefficient if there are dependencies between the elements. Therefore, preprocessing can be extended. The following code expands the previous example by introducing a class-based generator that encapsulates the data loading and some additional preprocessing steps:

```python
import os
import json
from typing import Generator, Tuple, List
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

class ImageDataTextGenerator:
    """
    A class-based generator for image and text data.
    """
    def __init__(self, image_dir: str, text_file: str, batch_size: int = 32, test_size: float = 0.2, seed: int = 42):
      """
        Initialize the generator.

        Args:
            image_dir: Path to the directory containing image files.
            text_file: Path to the JSON file.
            batch_size: The number of samples to include in each batch.
            test_size: The fraction of data to use for testing.
            seed: Random seed for shuffling.
      """
      self.image_dir = image_dir
      self.text_file = text_file
      self.batch_size = batch_size
      self.image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
      with open(text_file, 'r') as f:
          self.texts = [item['text'] for item in json.load(f) if 'text' in item]
      # Split the data for a more realistic use case
      self.image_files_train, self.image_files_test, self.texts_train, self.texts_test = train_test_split(
          self.image_files, self.texts, test_size=test_size, random_state=seed
      )
      self.training_indices = list(range(len(self.image_files_train)))

    def _load_image(self, image_path: str) -> np.ndarray:
      """
        Load and preprocess a single image.
        Args:
            image_path: Path to the image file.
        Returns:
            numpy.ndarray: A single preprocessed image array.
      """
      try:
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)
        img_array = np.resize(img_array,(64,64,3)) # Assuming consistent resizing
        return img_array
      except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None #Handle errors gracefully, and allow dataset to continue

    def _text_preprocess(self, text: str) -> str:
      """
        Preprocess a single text string.
        Args:
            text: Input text string.
        Returns:
            str: A preprocessed text string.
      """
      #Placeholder - Example tokenization.
      return text.lower().strip()

    def _batch_generator(self, image_files: List[str], texts: List[str]) -> Generator[Tuple[List[np.ndarray], List[str]], None, None]:
        """
        Generate batches of image and text data.

        Args:
            image_files: List of paths to the image files.
            texts: List of the text strings.

        Yields:
            Tuple[List[np.ndarray], List[str]]: A tuple containing batches of image arrays and text strings.
        """
        i = 0
        while i < len(image_files):
            batch_images = []
            batch_texts = []
            for j in range(self.batch_size):
                if i + j < len(image_files):
                  img = self._load_image(image_files[i + j])
                  text = self._text_preprocess(texts[i+j])
                  if img is not None: #avoid appending null images
                    batch_images.append(img)
                    batch_texts.append(text)

            if len(batch_images) > 0:
                yield batch_images, batch_texts
            i += self.batch_size

    def train_generator(self) -> Generator[Tuple[List[np.ndarray], List[str]], None, None]:
       """
       Returns the training data generator
       """
       yield from self._batch_generator(self.image_files_train, self.texts_train)

    def test_generator(self) -> Generator[Tuple[List[np.ndarray], List[str]], None, None]:
        """
        Returns the test data generator.
        """
        yield from self._batch_generator(self.image_files_test, self.texts_test)



# Example usage:
# image_directory = 'path/to/images'
# json_file = 'path/to/text.json'
# generator = ImageDataTextGenerator(image_directory, json_file, batch_size=32, test_size = 0.2)

# print("Training Data")
# for images, texts in generator.train_generator():
#     print(f"Image batch shape: {np.array(images).shape}, text samples: {[t[:10] for t in texts]}")
#    # Train model on batch

# print("Test Data")
# for images, texts in generator.test_generator():
#   print(f"Image batch shape: {np.array(images).shape}, text samples: {[t[:10] for t in texts]}")
#   # Evaluate model on batch
```

This improved version encapsulates the generator logic within a class (`ImageDataTextGenerator`), enhancing organization and adding batching functionality, along with data splitting. It introduces private methods for loading and preprocessing each type of data. Splitting into training and test sets facilitates proper evaluation of the model. Additionally, it demonstrates an approach to preprocessing texts using a simple tokenizer.

For more complex scenarios involving multiple input data types, one might need to build a composite data structure, and a custom `collate_fn`. The function below shows such an approach, where we might have data of varying types that we want to pass to different parts of our model.

```python
import os
import json
from typing import Generator, Tuple, List, Any
from PIL import Image
import numpy as np

class MultiModalDataGenerator:
    """
    A class-based generator for multimodal data (images, text and numeric data).
    """
    def __init__(self, image_dir: str, text_file: str, numeric_file: str, batch_size: int = 32):
        """
          Initialize the generator.

        Args:
            image_dir: Path to the directory containing image files.
            text_file: Path to the JSON file.
            numeric_file: Path to the numeric data (assumed csv).
            batch_size: The number of samples to include in each batch.
      """
        self.image_dir = image_dir
        self.text_file = text_file
        self.numeric_file = numeric_file
        self.batch_size = batch_size
        self.image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        with open(text_file, 'r') as f:
            self.texts = [item['text'] for item in json.load(f) if 'text' in item]
        with open(numeric_file, 'r') as f:
          self.numerics = [list(map(float,line.strip().split(','))) for line in f.readlines()]

    def _load_image(self, image_path: str) -> np.ndarray:
        """
        Load and preprocess a single image.
        Args:
            image_path: Path to the image file.
        Returns:
            numpy.ndarray: A single preprocessed image array.
      """
        try:
            img = Image.open(image_path).convert('RGB')
            img_array = np.array(img)
            img_array = np.resize(img_array,(64,64,3))
            return img_array
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None #Handle errors gracefully, and allow dataset to continue

    def _text_preprocess(self, text: str) -> str:
      """
        Preprocess a single text string.
        Args:
            text: Input text string.
        Returns:
            str: A preprocessed text string.
      """
      return text.lower().strip()

    def _numeric_preprocess(self, data: List[float]) -> np.ndarray:
        """
        Preprocess numeric data.
        Args:
            data: Input numeric data as a list of floats
        Returns:
            numpy.ndarray: A preprocessed numeric data array.
      """
        return np.array(data)

    def _data_generator(self) -> Generator[Tuple[Any, Any, Any], None, None]:
       """
       Combines all generator outputs, yielding a tuple of image, text and numeric data.

       Yields:
           Tuple[Any, Any, Any]: A tuple containing image, text and numeric data.
       """

       i = 0
       while i < len(self.image_files):
            batch_images = []
            batch_texts = []
            batch_numerics = []
            for j in range(self.batch_size):
                if i + j < len(self.image_files):
                  img = self._load_image(self.image_files[i+j])
                  text = self._text_preprocess(self.texts[i+j])
                  numeric = self._numeric_preprocess(self.numerics[i+j])
                  if img is not None:
                     batch_images.append(img)
                     batch_texts.append(text)
                     batch_numerics.append(numeric)

            if len(batch_images) > 0:
              yield batch_images, batch_texts, np.array(batch_numerics)
            i += self.batch_size

    def generator(self) -> Generator[Tuple[Any, Any, Any], None, None]:
      """
      Returns the data generator.
      """
      yield from self._data_generator()

# Example usage:
# image_directory = 'path/to/images'
# json_file = 'path/to/text.json'
# numeric_file = 'path/to/numeric_data.csv'
# generator = MultiModalDataGenerator(image_directory, json_file, numeric_file, batch_size=32)

# for images, texts, numerics in generator.generator():
#   print(f"Image batch shape: {np.array(images).shape}, text samples: {[t[:10] for t in texts]}, numeric shape: {numerics.shape}")
# # Train on combined batches
```

This last example expands on the concepts by showing the flexibility of how custom generators can be used to generate any combination of data types. The `_data_generator` yields a tuple of a list of images, a list of texts, and the numeric data. These can then be handled by a custom `collate_fn` (not shown) and sent to the model.

For further study of techniques for advanced data loading, I suggest reviewing the documentation on Python generators and iterators, as well as exploring the `tf.data` API (Tensorflow) and PyTorch's `DataLoader`, as these libraries are focused on efficient model training with large data sets. These offer highly optimized mechanisms for data pipelining, shuffling, and batching that often improve upon the techniques described here. Additionally, literature on data engineering and machine learning best practices would be beneficial. Focusing on optimizing the loading and preprocessing stages of model development can result in large gains in training efficiency.
