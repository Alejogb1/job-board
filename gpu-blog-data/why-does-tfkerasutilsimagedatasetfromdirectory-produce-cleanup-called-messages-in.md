---
title: "Why does `tf.keras.utils.image_dataset_from_directory` produce 'Cleanup called...' messages in a notebook?"
date: "2025-01-30"
id: "why-does-tfkerasutilsimagedatasetfromdirectory-produce-cleanup-called-messages-in"
---
The `tf.keras.utils.image_dataset_from_directory` function, while convenient for loading image datasets, often triggers "Cleanup called..." messages within Jupyter notebooks or similar interactive environments.  This isn't indicative of a bug within TensorFlow itself, but rather a consequence of the underlying data handling and garbage collection processes interacting with the dynamic nature of notebook execution.  My experience working on large-scale image classification projects has highlighted this behavior repeatedly, specifically when dealing with substantial datasets or limited system resources.

The core issue stems from the way `image_dataset_from_directory` manages file handles and temporary objects. The function efficiently batches and preprocesses images from a directory structure, but this process involves creating and potentially discarding numerous intermediate objects – temporary files, memory buffers, and potentially even subprocesses depending on the image format and preprocessing steps.  When these objects are no longer referenced by the Python interpreter, the garbage collector (GC) is triggered to reclaim memory.  This garbage collection process is what produces the "Cleanup called..." messages.  These messages are informational and not necessarily indicative of an error; they simply report the GC's activity in releasing resources occupied by objects that are no longer in use.

However, the frequency and visibility of these messages can be influenced by several factors.  A larger dataset necessitates more temporary objects and thus, more frequent garbage collection.  Limited RAM can exacerbate this, forcing the GC to intervene more aggressively to free up memory.  Finally, the Jupyter notebook environment itself has a somewhat less predictable memory management behaviour compared to a standard Python script execution, potentially leading to more noticeable GC activity.

Let's examine this through code examples.  Each example demonstrates a scenario that might trigger these messages, along with strategies to mitigate the issue.  These examples assume familiarity with TensorFlow and Keras.

**Example 1: Large Dataset, Limited Resources**

```python
import tensorflow as tf

IMG_HEIGHT, IMG_WIDTH = 128, 128
BATCH_SIZE = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
    '/path/to/large/dataset',
    labels='inferred',
    label_mode='binary',
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    interpolation='nearest',
    batch_size=BATCH_SIZE,
    shuffle=True
)

for images, labels in train_ds:
    # Process each batch here
    pass
```

This example is straightforward, loading a large dataset.  The message frequency is directly related to the dataset size and the system's available memory.  On systems with limited RAM, the frequent GC calls, as the data is loaded and processed batch-wise, will result in noticeable "Cleanup called..." messages.  A solution is to reduce `BATCH_SIZE` to process smaller chunks of data at a time, limiting the memory footprint of each iteration.


**Example 2:  Complex Preprocessing**

```python
import tensorflow as tf

IMG_HEIGHT, IMG_WIDTH = 256, 256
BATCH_SIZE = 16

train_ds = tf.keras.utils.image_dataset_from_directory(
    '/path/to/dataset',
    labels='inferred',
    label_mode='categorical',
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    interpolation='bicubic',
    batch_size=BATCH_SIZE,
    shuffle=True,
    preprocessing_function=lambda img: tf.image.adjust_brightness(img, 0.5)
)

for images, labels in train_ds:
    #Further preprocessing might be done here
    pass
```

Here, we introduce a custom `preprocessing_function`.  Complex preprocessing can increase the number of temporary objects created during image manipulation.  The resulting increased memory usage leads to more frequent GC activity.  The solution involves simplifying preprocessing or optimizing the function for efficiency, reducing the memory overhead per image.


**Example 3: Memory-Efficient Loading**

```python
import tensorflow as tf

IMG_HEIGHT, IMG_WIDTH = 64, 64
BATCH_SIZE = 64

train_ds = tf.keras.utils.image_dataset_from_directory(
    '/path/to/dataset',
    labels='inferred',
    label_mode='int',
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    interpolation='nearest',
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42
)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)

for images, labels in train_ds:
    # Training loop here
    pass
```

This example introduces `cache()` and `prefetch()`.  `cache()` loads the entire dataset into memory, reducing disk I/O during training but increasing memory consumption. `prefetch()` prepares the next batch while the current batch is being processed, improving performance and potentially reducing GC calls by smoothing out the memory usage pattern. Note the reduced image size and increased `BATCH_SIZE` to offset the added memory used by `cache()`.  Careful consideration of memory usage is crucial when employing `cache()`.  For very large datasets, it might still lead to frequent GC activity.


In summary, the "Cleanup called..." messages from `image_dataset_from_directory` primarily reflect the garbage collector's work in managing memory resources during dataset processing.  The frequency of these messages isn't inherently problematic but can be influenced by dataset size, system resources, and preprocessing complexity.  Strategic adjustment of parameters like `BATCH_SIZE`, careful consideration of preprocessing steps, and effective use of TensorFlow's data handling utilities like `cache()` and `prefetch()` can minimize their occurrence and improve the overall efficiency and stability of your notebook's execution.


**Resource Recommendations:**

*   TensorFlow documentation: Focus on the `tf.data` API for advanced data handling techniques.
*   Python's `gc` module documentation: Understand the garbage collection mechanism in Python.
*   A book on efficient Python programming: Explore memory management best practices.
