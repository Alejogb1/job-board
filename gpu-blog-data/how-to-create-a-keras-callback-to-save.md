---
title: "How to create a Keras callback to save batch predictions and targets during training?"
date: "2025-01-30"
id: "how-to-create-a-keras-callback-to-save"
---
Saving batch predictions and targets during Keras model training requires careful consideration of memory management and efficient data handling.  My experience working on large-scale image classification projects highlighted the limitations of simply appending predictions to a list within a callback; this approach rapidly consumes memory, especially with extensive datasets and complex models.  Instead, a more robust solution involves leveraging NumPy's memory-mapped arrays or a suitable database for persistent storage.  This ensures that prediction and target data are written to disk incrementally, preventing memory exhaustion and allowing for post-training analysis without loading the entire dataset into RAM.

**1. Clear Explanation**

The core functionality of a Keras callback revolves around the `on_epoch_end` or `on_batch_end` method.  We'll focus on `on_batch_end` for finer-grained control. Within this method, we access the batch predictions and targets directly from the Keras `model` object. However, directly appending these to a Python list is inefficient and prone to `MemoryError` exceptions.  To avoid this, we will use a memory-mapped file (using NumPy's `memmap`) to store the data. This allows us to write data in chunks, only loading the necessary portions into memory at any given time.

The process involves pre-allocating a memory-mapped array of the appropriate size before training begins.  This size is determined by the total number of batches and the batch size. During each batch, the predictions and targets are written to their respective sections of the memory-mapped array. After training completes, the data can be easily accessed from the memory-mapped file, for example, to generate confusion matrices or to perform detailed error analysis on specific batches.  Alternative approaches, such as using a database (like SQLite), provide similar advantages but with potentially higher overhead for smaller datasets.


**2. Code Examples with Commentary**

**Example 1: Using NumPy's `memmap`**

This example demonstrates saving predictions and targets to a memory-mapped file using NumPy.  This approach is ideal for datasets that fit comfortably within the available disk space, but might still exceed available RAM.


```python
import numpy as np
from tensorflow import keras

class BatchPredictionSaver(keras.callbacks.Callback):
    def __init__(self, filepath_predictions, filepath_targets, batch_size, num_classes, num_batches):
        super(BatchPredictionSaver, self).__init__()
        self.predictions_mmap = np.memmap(filepath_predictions, dtype='float32', mode='w+', shape=(num_batches, batch_size, num_classes))
        self.targets_mmap = np.memmap(filepath_targets, dtype='int32', mode='w+', shape=(num_batches, batch_size))
        self.batch_size = batch_size
        self.batch_counter = 0

    def on_batch_end(self, batch, logs=None):
        predictions = self.model.predict(self.model.input)  # Assuming functional API; adjust if using Sequential
        targets = self.model.get_layer('target_layer').output # Assumes a defined target layer; adapt as needed
        self.predictions_mmap[self.batch_counter] = predictions
        self.targets_mmap[self.batch_counter] = targets
        self.batch_counter += 1

    def on_train_end(self, logs=None):
        self.predictions_mmap.flush()
        self.targets_mmap.flush()
        del self.predictions_mmap
        del self.targets_mmap


# Example usage:
batch_size = 32
num_classes = 10
num_batches = 1000 #Example number, adjust as needed

filepath_predictions = 'predictions.dat'
filepath_targets = 'targets.dat'


callback = BatchPredictionSaver(filepath_predictions, filepath_targets, batch_size, num_classes, num_batches)

model = keras.Model(...) #Your Keras model definition here.  Make sure to define the target layer appropriately.
model.compile(...) # Your model compilation here.

model.fit(..., callbacks=[callback], ...)


```

**Commentary:** This code defines a custom callback that uses `memmap` to store predictions and targets.  Crucially,  `on_train_end` ensures that all data is written to disk before the callback is deactivated, and the `memmap` objects are explicitly deleted to release resources.  The placeholder comments highlight the necessary model definition and compilation steps, which should be adapted to the user's specific model architecture and data. Note the necessity of knowing `num_batches` beforehand.  This can be calculated from the dataset size and batch size. The assumption of a functional API and the existence of 'target_layer' are important - adapt this to match your specific model.

**Example 2: Handling Variable Batch Sizes**

The previous example assumes a constant batch size. To handle variable batch sizes, dynamic allocation is needed.


```python
import numpy as np
from tensorflow import keras

class VariableBatchPredictionSaver(keras.callbacks.Callback):
    def __init__(self, filepath_predictions, filepath_targets):
        super(VariableBatchPredictionSaver, self).__init__()
        self.filepath_predictions = filepath_predictions
        self.filepath_targets = filepath_targets
        self.predictions = []
        self.targets = []

    def on_batch_end(self, batch, logs=None):
        predictions = self.model.predict(self.model.input)
        targets = self.model.get_layer('target_layer').output
        self.predictions.append(predictions)
        self.targets.append(targets)

    def on_epoch_end(self, epoch, logs=None):
        predictions_np = np.concatenate(self.predictions, axis=0)
        targets_np = np.concatenate(self.targets, axis=0)
        np.save(self.filepath_predictions + f'_{epoch}.npy', predictions_np)
        np.save(self.filepath_targets + f'_{epoch}.npy', targets_np)
        self.predictions = []
        self.targets = []

```

**Commentary:** This example uses lists to accumulate predictions and targets during each epoch. At the end of each epoch,  the accumulated data is concatenated into NumPy arrays and saved to disk using `np.save`.  This approach avoids pre-allocation but sacrifices the memory efficiency of `memmap` for handling variable batch sizes.  The files are saved per epoch, which might be preferable in certain scenarios, especially if an epoch's predictions are large enough to cause issues.  The `f'_{epoch}.npy'` suffix ensures that files from different epochs don't overwrite each other.  This approach requires more disk space than `memmap`.



**Example 3:  Illustrative SQLite Integration (Conceptual)**

While full SQLite integration is beyond a concise example, the conceptual approach is presented.

```python
import sqlite3
from tensorflow import keras

class SQLiteBatchPredictionSaver(keras.callbacks.Callback):
    def __init__(self, db_path, table_name):
        super(SQLiteBatchPredictionSaver, self).__init__()
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute(f'''CREATE TABLE IF NOT EXISTS {table_name} (
                                    batch_id INTEGER,
                                    prediction BLOB,
                                    target BLOB
                                )''')
        self.conn.commit()
        self.batch_id_counter = 0


    def on_batch_end(self, batch, logs=None):
        predictions = self.model.predict(self.model.input)
        targets = self.model.get_layer('target_layer').output
        prediction_bytes = predictions.tobytes()
        target_bytes = targets.tobytes()
        self.cursor.execute(f"INSERT INTO predictions_targets VALUES (?, ?, ?)", (self.batch_id_counter, prediction_bytes, target_bytes))
        self.conn.commit()
        self.batch_id_counter += 1


    def on_train_end(self, logs=None):
        self.conn.close()

```

**Commentary:** This illustrates a database approach.  Predictions and targets are stored as `BLOB` fields.  This requires serialization of NumPy arrays into bytes. The `batch_id` ensures unique identification of each batch.  This approach is preferable for very large datasets where memory mapping might still be insufficient or when concurrent access to the data is required.  Remember to handle potential exceptions during database operations.  Appropriate error handling and transaction management should be included in a production-ready implementation.


**3. Resource Recommendations**

*   NumPy documentation on memory-mapped arrays.
*   SQLite documentation for Python integration.
*   The Keras documentation on custom callbacks.
*   A comprehensive guide on Python database interaction.
*   A textbook on data structures and algorithms for efficient data management.


These resources provide the necessary background knowledge and practical guidance for implementing and optimizing the proposed solutions.  Consider the trade-offs between memory efficiency, disk space usage, and processing overhead when choosing the most appropriate approach for your specific needs.  Remember to adapt these examples to your specific model architecture and data characteristics.  Thorough testing and validation are essential to ensure the correctness and robustness of your implementation.
