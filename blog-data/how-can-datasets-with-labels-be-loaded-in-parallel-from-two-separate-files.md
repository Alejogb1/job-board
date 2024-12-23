---
title: "How can datasets with labels be loaded in parallel from two separate files?"
date: "2024-12-23"
id: "how-can-datasets-with-labels-be-loaded-in-parallel-from-two-separate-files"
---

Alright, let's tackle this. I've seen this specific challenge crop up more often than one might expect, particularly when dealing with large datasets split across different storage mediums or systems for practical reasons. It’s not always a straightforward case of simply loading everything into memory; performance and resource management become critical. The task of loading labelled data in parallel from two files, which essentially boils down to coordinating the reading of feature data and associated labels, requires a bit more careful planning than sequential loading.

The general approach involves asynchronous or multithreaded loading of data from both files simultaneously and then ensuring the correct pairing of data and labels. The core idea is to avoid the bottleneck that occurs when loading each file sequentially. Specifically, we'll need to use parallel processing libraries (or language-level constructs if available) to accomplish this, and the implementation will depend heavily on the specifics of the programming environment being used. Let me walk you through how I’ve previously approached this.

Firstly, let's consider a scenario where your feature data is stored in a file called `features.csv` and your labels are in a corresponding `labels.csv`. Assume that each row in `features.csv` maps directly to the same row in `labels.csv`. This is crucial; without this correspondence, we have a bigger data integrity issue to resolve before even attempting to parallel load.

Now, if we're dealing with Python, we can leverage libraries like `concurrent.futures` for thread or process based parallelism. Here's a basic example using threads, suitable for I/O-bound operations where threads generally outperform processes because the global interpreter lock (GIL) doesn't become as significant of a constraint with I/O as it does with CPU-intensive tasks.

```python
import csv
import concurrent.futures

def load_data_from_csv(filename):
    """Loads data from a csv file."""
    data = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header
        for row in reader:
            data.append(row) # Appending rows as lists of strings for simplicity
    return data

def parallel_load_data(features_file, labels_file):
    """Loads features and labels in parallel and pairs them."""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        feature_future = executor.submit(load_data_from_csv, features_file)
        labels_future = executor.submit(load_data_from_csv, labels_file)

        features_data = feature_future.result()
        labels_data = labels_future.result()

        # Assuming a one-to-one correspondence between rows
        if len(features_data) != len(labels_data):
            raise ValueError("Mismatch in the number of data points between feature and label files.")

        paired_data = list(zip(features_data, labels_data))
        return paired_data

if __name__ == "__main__":
  # create dummy csv files
  with open("features.csv", 'w', newline='') as f_feat, open("labels.csv", 'w', newline='') as f_label:
      writer_feat = csv.writer(f_feat)
      writer_label = csv.writer(f_label)
      writer_feat.writerow(["feat1","feat2", "feat3"])
      writer_label.writerow(["label"])
      for i in range(10):
         writer_feat.writerow([str(i),str(i*2),str(i*3)])
         writer_label.writerow([str(i%2)])

  feature_file = 'features.csv'
  label_file = 'labels.csv'
  paired_results = parallel_load_data(feature_file, label_file)

  for features, labels in paired_results:
      print(f"Features: {features}, Label: {labels}")
```

In this snippet, `ThreadPoolExecutor` is used to dispatch the file loading function for both feature and label files concurrently. We use `submit` to invoke `load_data_from_csv` on both files and obtain the results via `result()`, which will block until the result of the loading is available. This simple approach effectively parallelizes file reads and it is quite scalable in practice, as I have previously implemented in similar scenarios involving large text datasets.

Now, let's consider a more complex case using memory-mapped files, which can be beneficial if your dataset is too large to fit entirely in memory. Memory mapping can provide a seemingly file-like interface while actually operating on the underlying memory. Here, we’ll illustrate with a very simple case where we are using raw binary files, since a structured file like CSV introduces complexities best addressed with other libraries. For this, we will be using `mmap` in Python, which directly provides the memory mapping functionality.

```python
import mmap
import os
import concurrent.futures
import struct


def load_from_mmap(filename, item_size):
    """Loads data from a memory mapped file."""
    data = []
    with open(filename, 'rb') as file:
        with mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            for offset in range(0, len(mm), item_size):
                item_bytes = mm[offset : offset + item_size]
                # Unpack as integers based on a fixed item size of 4 bytes
                item_val = struct.unpack('i',item_bytes)[0]
                data.append(item_val)
    return data

def parallel_load_mmap(features_file, labels_file):
    """Loads memory mapped files and pairs them."""
    item_size = 4 # Assuming an integer as the primitive data type for example
    with concurrent.futures.ThreadPoolExecutor() as executor:
        feature_future = executor.submit(load_from_mmap, features_file, item_size)
        labels_future = executor.submit(load_from_mmap, labels_file, item_size)

        features_data = feature_future.result()
        labels_data = labels_future.result()
        if len(features_data) != len(labels_data):
            raise ValueError("Mismatch in the number of data points between feature and label files.")

        paired_data = list(zip(features_data, labels_data))
        return paired_data

if __name__ == '__main__':

    num_items = 10
    item_size = 4 # integer for example

    # create dummy binary files
    with open("features_bin.bin", "wb") as f_feat, open("labels_bin.bin", "wb") as f_label:
        for i in range(num_items):
           f_feat.write(struct.pack('i', i*100)) # writing integers to binary file
           f_label.write(struct.pack('i', i%2))

    feature_file = 'features_bin.bin'
    label_file = 'labels_bin.bin'

    paired_results = parallel_load_mmap(feature_file, label_file)

    for features, labels in paired_results:
         print(f"Features: {features}, Label: {labels}")
```

Here, instead of reading from CSV files, we're reading raw binary files, and we utilize memory mapping via `mmap` to access the file contents. This approach can be significantly faster than loading large files directly into memory, especially if the data size is larger than available ram. The `struct.unpack` is used to interpret the binary data, with 'i' indicating the format is integer; you would have to adjust that based on the actual binary data format.

Lastly, let's consider an approach if you had to use processes instead of threads. This is crucial in CPU-bound situations where, again, the Global Interpreter Lock of Python would make threads inefficient. In this situation, the process based `ProcessPoolExecutor` is an appropriate choice. The core logic remains very similar.

```python
import csv
import concurrent.futures
import os

def load_data_from_csv_process(filename):
    """Loads data from a csv file."""
    data = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader) # Skip the header
        for row in reader:
            data.append(row)  # Appending rows as list of strings
    return data

def parallel_load_data_process(features_file, labels_file):
    """Loads features and labels in parallel using processes and pairs them."""
    with concurrent.futures.ProcessPoolExecutor() as executor:
        feature_future = executor.submit(load_data_from_csv_process, features_file)
        labels_future = executor.submit(load_data_from_csv_process, labels_file)

        features_data = feature_future.result()
        labels_data = labels_future.result()
        if len(features_data) != len(labels_data):
            raise ValueError("Mismatch in the number of data points between feature and label files.")
        paired_data = list(zip(features_data, labels_data))
        return paired_data

if __name__ == "__main__":
  # create dummy csv files
  with open("features.csv", 'w', newline='') as f_feat, open("labels.csv", 'w', newline='') as f_label:
      writer_feat = csv.writer(f_feat)
      writer_label = csv.writer(f_label)
      writer_feat.writerow(["feat1","feat2", "feat3"])
      writer_label.writerow(["label"])
      for i in range(10):
         writer_feat.writerow([str(i),str(i*2),str(i*3)])
         writer_label.writerow([str(i%2)])

  feature_file = 'features.csv'
  label_file = 'labels.csv'

  paired_results = parallel_load_data_process(feature_file, label_file)

  for features, labels in paired_results:
      print(f"Features: {features}, Label: {labels}")
```
The changes are in the `parallel_load_data_process` function using the `ProcessPoolExecutor`, which spawns processes instead of threads. This approach is preferred if the loading function is CPU-bound since it bypasses the limitations of the GIL, although in most file loading tasks, the majority of the time spent is waiting for i/o. The rest of the code follows the same logic as the initial example. This shows that the core concept can be utilized even if the underlying method for parallelism is different.

To further delve into these concepts, I highly recommend delving into "Operating System Concepts" by Abraham Silberschatz, Peter Baer Galvin, and Greg Gagne, for a more in-depth understanding of thread and process management at the OS level. For python specifics, check the official `concurrent.futures` module documentation in python.org and any online materials related to memory mapped files via the `mmap` library. Finally, for understanding the intricacies of different parallel techniques, reading the book "Parallel Programming" by Thomas Rauber and Gudula Rünger is very informative.

In conclusion, parallel loading from separate feature and label files can drastically speed up your data loading process, especially with larger datasets. The specific implementation will depend on your language, data format, and operational constraints, but the overarching principle is to parallelize I/O and ensure correct pairing of data and labels, and based on my past projects, there are usually no unsolvable challenges here. The provided examples should give a solid starting point.
