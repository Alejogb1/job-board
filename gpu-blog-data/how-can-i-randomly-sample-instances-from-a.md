---
title: "How can I randomly sample instances from a TensorFlow dataset?"
date: "2025-01-30"
id: "how-can-i-randomly-sample-instances-from-a"
---
TensorFlow Datasets, when structured for machine learning tasks, often require random subsampling for model training, validation, or testing. A naive approach of converting the entire dataset to a list and then performing Python-based random selection proves inefficient, particularly for large datasets exceeding system memory. Leveraging TensorFlow's built-in shuffling and sampling functionalities provides a robust and performant solution.

I've spent a significant portion of my time fine-tuning large language models, and frequently dealt with massive datasets. Shuffling data before training to avoid any ordering bias is critical, but during the debugging or experimentation phase I often need to select random batches or individual samples. The key here isn't about external randomisation as it exists in Python, but using TensorFlow’s internal tools. The most straightforward way to implement random sampling involves three primary components within the TensorFlow Dataset API: the `shuffle()`, `skip()`, and `take()` transformations. These methods allow controlled, randomized access to data without loading it entirely into memory.

The fundamental mechanism is this: `shuffle()` randomizes the order of the elements within the dataset, with a buffer size dictating the scope of randomization; a higher buffer typically leads to more thorough shuffling, but also higher memory usage. Following the shuffle, `skip()` can advance to a random point within the now-shuffled dataset, allowing us to start a sample at an arbitrary position. Finally, `take()` limits the dataset to a specified number of elements, effectively returning a sample of the desired size.

Let's consider a practical scenario. Imagine a dataset consisting of text and labels for a classification problem. I needed to randomly select a small, representative set for debugging a custom loss function. Here's the first example illustrating how to obtain a random sample of five data instances:

```python
import tensorflow as tf

def create_dummy_dataset(num_elements):
    data = [ (f"Text {i}", i % 2) for i in range(num_elements)]
    return tf.data.Dataset.from_tensor_slices(data)

dataset = create_dummy_dataset(1000)  # Generate a sample dataset

# Parameters
BUFFER_SIZE = 1000 # A buffer size equal to or larger than the dataset size ensures full shuffling
SAMPLE_SIZE = 5

shuffled_dataset = dataset.shuffle(buffer_size=BUFFER_SIZE)
random_sample = shuffled_dataset.take(SAMPLE_SIZE)

for text, label in random_sample.as_numpy_iterator():
    print(f"Text: {text.decode('utf-8')}, Label: {label}")
```

In this code, a dummy dataset is created using `tf.data.Dataset.from_tensor_slices()`. Then, `dataset.shuffle(BUFFER_SIZE)` randomizes the order of the elements using a buffer equal to the number of elements; the larger the buffer, the more random the shuffle. Finally, the `take(SAMPLE_SIZE)` method selects the first five elements from the shuffled dataset. This approach avoids loading the entire dataset into memory, thus remaining efficient even with larger data volumes. It also guarantees that all five selected elements are from different parts of the shuffled sequence.

For a more targeted scenario, I often found myself needing to draw multiple, non-overlapping samples from a dataset, resembling a random train/test split with a fixed random state. Let’s say I need a small validation set of size 10 from a large dataset. A direct `take()` with a constant random seed won’t produce a different sample; therefore, an explicit, variable starting point (controlled via the `skip()` method) is necessary. Below is a modified implementation that extracts disjoint samples:

```python
import tensorflow as tf

def create_dummy_dataset(num_elements):
    data = [ (f"Text {i}", i % 2) for i in range(num_elements)]
    return tf.data.Dataset.from_tensor_slices(data)

dataset = create_dummy_dataset(1000)  # Generate a sample dataset

# Parameters
BUFFER_SIZE = 1000
SAMPLE_SIZE = 10
RANDOM_SEED = 42

shuffled_dataset = dataset.shuffle(buffer_size=BUFFER_SIZE, seed = RANDOM_SEED)

# Get the total number of elements
num_elements = tf.data.experimental.cardinality(dataset).numpy()

#Calculate how many elements to skip, use the sample index to get multiple different samples
sample_index = 50 #this would be determined in practice by a loop of some kind
start_index = (sample_index * SAMPLE_SIZE) % num_elements

random_sample = shuffled_dataset.skip(start_index).take(SAMPLE_SIZE)

for text, label in random_sample.as_numpy_iterator():
    print(f"Text: {text.decode('utf-8')}, Label: {label}")
```

Here, after shuffling using a specific seed, `start_index` determines the beginning of the sample. Crucially, `skip()` allows us to start at a specific element based on the index provided, ensuring that repeated executions using different `sample_index` yield distinct, non-overlapping samples. The modulus operation (`% num_elements`) guarantees that the index stays within the dataset boundaries, even when the sample index is very high. This provides a stable and reproducible sampling process, controlled by the specified `RANDOM_SEED` and `sample_index`.

Finally, for practical debugging of model training, I often wanted to visualise one sample batch. Given that TensorFlow datasets produce batch, I often had to take just one element out of the dataset which was not already batched. Below is the code which accomplishes just that; this is slightly less common, but was critical in some model debugging phases.

```python
import tensorflow as tf
import numpy as np

def create_dummy_dataset(num_elements):
    data = [ (f"Text {i}", i % 2) for i in range(num_elements)]
    return tf.data.Dataset.from_tensor_slices(data)

dataset = create_dummy_dataset(1000)  # Generate a sample dataset

# Parameters
BUFFER_SIZE = 1000
BATCH_SIZE = 32
RANDOM_SEED = 42
sample_index = 10

shuffled_dataset = dataset.shuffle(buffer_size=BUFFER_SIZE, seed = RANDOM_SEED)

# Get the total number of elements
num_elements = tf.data.experimental.cardinality(dataset).numpy()

#Apply batching and create batch iterator
batched_dataset = shuffled_dataset.batch(BATCH_SIZE)
batch_iterator = iter(batched_dataset)

#Compute sample_batch for each sample
sample_batch = (sample_index * BATCH_SIZE) % num_elements
start_index = (sample_index * BATCH_SIZE) // num_elements

#Use skip to skip to the correct batch and then take one batch
random_batch = batched_dataset.skip(start_index).take(1)

#Extract the first element of the batch
for batch_text, batch_label in random_batch.as_numpy_iterator():
    print("The batch sample is:")
    for text, label in zip(batch_text, batch_label):
        print(f"Text: {text.decode('utf-8')}, Label: {label}")
```

Here, following the shuffling, the dataset is batched. Instead of `take` and `skip`, the same logical operation is applied on batches. This code first calculates which batch contains the element we want, and then skips to that batch; this is not something that can be done with the non-batched dataset. Finally, we print each element of the batch. While the example demonstrates selecting a single batch, the `skip` and `take` approach can be extended to select multiple batches.

In practical application, these techniques are crucial for controlled data access. I often combine shuffling with stratified sampling techniques (not demonstrated here, but readily found in other modules like `tf.data.experimental.sample_from_datasets`) when class imbalances exist within my datasets.

For further exploration, the official TensorFlow documentation is essential. The modules on data loading and pre-processing, especially regarding `tf.data.Dataset` transformations, provide detailed examples. Also, academic papers on machine learning best practices often address the importance of proper data handling, which includes randomisation as a critical step in model development. I found that following community posts on machine learning forums (such as Stack Overflow) was extremely useful, as users often present different sampling scenarios, which provide different ways of conceptualising and understanding the underlying concepts. The TensorFlow documentation also includes code examples that are readily available for experimentation.
