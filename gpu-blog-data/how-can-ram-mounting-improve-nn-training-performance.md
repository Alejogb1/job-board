---
title: "How can RAM mounting improve NN training performance in Google Colab?"
date: "2025-01-30"
id: "how-can-ram-mounting-improve-nn-training-performance"
---
Google Colab environments, by default, utilize cloud-backed virtual machines with limited disk I/O bandwidth. Consequently, repeatedly accessing datasets stored on Google Drive or the Colab instance’s persistent disk during neural network training can become a significant bottleneck, substantially hindering performance. Mounting a portion of the system's RAM as a virtual disk, a technique known as RAM mounting, offers a solution by creating a high-speed storage location readily accessible to the training process, mitigating this I/O bottleneck and speeding up data loading. This approach leverages the much faster access speeds of RAM compared to persistent disk or network storage.

In essence, RAM mounting transforms a portion of the volatile system memory into a temporary file system. This virtual disk operates entirely within the RAM, enabling extremely rapid read and write operations. During neural network training, datasets can be initially transferred to this RAM disk, and subsequent data loading for training epochs can then be performed from this fast location, avoiding the considerably slower access to traditional disk-based storage. This is especially beneficial when training with relatively large datasets that can fit within the available RAM, where data loading significantly impacts training speed.

To illustrate the benefits, I recall a past project involving image classification on a substantial dataset of high-resolution satellite imagery. Initially, training directly from mounted Google Drive folders led to significant delays as the model waited for image data to be read and preprocessed during each training step. The iterative reading and preprocessing became the dominant performance bottleneck. Introducing RAM mounting markedly reduced training time, revealing the direct impact of disk I/O on training speed. Let me now illustrate how to implement this with Python code examples.

**Code Example 1: Basic RAM Mounting and Data Transfer**

The following code demonstrates how to mount a RAM disk and copy a sample dataset. Here, I use a size of 8 GB for the RAM disk, as typically available in a Google Colab Pro environment:

```python
import os
import subprocess

def mount_ramdisk(size_gb=8):
    ramdisk_path = "/mnt/ramdisk"
    if not os.path.exists(ramdisk_path):
       subprocess.run(["mkdir", ramdisk_path], check=True) # creates directory
    subprocess.run(["mount", "-t", "tmpfs", "-o", f"size={size_gb}G", "tmpfs", ramdisk_path], check=True)
    print(f"RAM disk mounted at {ramdisk_path} with size {size_gb}GB.")
    return ramdisk_path

def copy_data_to_ramdisk(data_path, ramdisk_path):
    if not os.path.exists(data_path):
       raise FileNotFoundError(f"Data path does not exist: {data_path}")
    subprocess.run(["cp", "-r", data_path, ramdisk_path], check=True)
    print(f"Data copied from {data_path} to {ramdisk_path}")

if __name__ == '__main__':
    ramdisk_path = mount_ramdisk() # this step mounts the ramdisk
    data_path = "/content/sample_data" # replace with your actual dataset location in Google colab
    if os.path.exists(data_path):
      copy_data_to_ramdisk(data_path, ramdisk_path)
    else:
       print(f"sample_data path does not exist: {data_path}") # this path is just for testing
       # you would usually place a mounted google drive folder here.

```

This code first defines a function `mount_ramdisk` which uses the `subprocess` module to create a directory `/mnt/ramdisk` and then mounts a temporary filesystem (`tmpfs`) of the specified size (8 GB in this case) to that directory. This effectively creates the RAM disk. Then, it defines `copy_data_to_ramdisk` which copies the data from `data_path` to `ramdisk_path`. The `if __name__ == '__main__'` block shows how to call these methods. The `/content/sample_data` path is used for testing here, in a real application, you would typically replace this with the path of your mounted google drive dataset folder. The `check=True` parameter ensures an error is raised if a command fails, assisting in debugging. The output from this step confirms the RAM disk has been created and the data has been transferred. You'd replace "/content/sample_data" with the path to your actual data directory in Google Colab, for example: `/content/drive/MyDrive/your_dataset`.

**Code Example 2: Using a RAM Disk with TensorFlow Data Loading**

This example demonstrates how to incorporate the RAM disk into a TensorFlow data loading pipeline:

```python
import tensorflow as tf
import os
import subprocess
import numpy as np

# Assume ramdisk is already mounted from previous example.
ramdisk_path = "/mnt/ramdisk"

def prepare_dataset_for_training(ramdisk_path, batch_size=32):

    # Assuming your dataset consists of numerical files, create an example dummy path for loading
    example_dataset_folder = os.path.join(ramdisk_path, 'sample_data')
    # use this dummy folder to create dummy numerical data
    example_data_files = []
    if os.path.exists(example_dataset_folder):
      for i in range(10):
        example_data_file_path = os.path.join(example_dataset_folder, f"sample_data_{i}.npy")
        # generate dummy numpy array, size and shape does not matter for this example
        dummy_data = np.random.rand(10,10)
        np.save(example_data_file_path, dummy_data)
        example_data_files.append(example_data_file_path)
    else:
       print(f"sample_data path does not exist: {example_dataset_folder}")
       return None

    if example_data_files:
      def load_and_preprocess(file_path):
         # Here, read data from each file_path into a tensor,
         # this is an example you would replace with your real file loading and preprocessing
         tensor = tf.convert_to_tensor(np.load(file_path), dtype=tf.float32)
         # add your preprocessing steps here if needed
         return tensor
      # use the dataset API to create a dataset
      dataset = tf.data.Dataset.from_tensor_slices(example_data_files)
      dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
      dataset = dataset.batch(batch_size)
      dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
      return dataset
    return None

if __name__ == '__main__':
    # replace with the location of the sample data folder
    dataset = prepare_dataset_for_training(ramdisk_path)

    if dataset is not None:
        for batch in dataset.take(2):
            print("Batch shape: ", batch.shape)
    else:
        print("Failed to initialize dataset")

```

Here, I've created an artificial dataset in a subfolder named 'sample_data' on the RAM disk, and each file contains a randomly generated NumPy array. This mimics a scenario where you have multiple data files, such as image files or feature files. The `prepare_dataset_for_training` function loads the file paths, reads data from the files in parallel using TensorFlow's dataset API, then preprocesses it (in this dummy example by simply converting to tensor), batches the data, and prefetches it for optimal performance. This ensures that the model is constantly fed data without waiting for it to be loaded from the slower storage. The code includes comments on how this dummy example can be changed to fit your specific data format and preprocessing requirements.

**Code Example 3: Cleanup - Unmounting the RAM Disk**

Properly unmounting the RAM disk after use is important. This code shows the function to do so:

```python
import subprocess

def unmount_ramdisk(ramdisk_path):
    subprocess.run(["umount", ramdisk_path], check=True)
    print(f"RAM disk unmounted at {ramdisk_path}")

if __name__ == '__main__':
    ramdisk_path = "/mnt/ramdisk"
    unmount_ramdisk(ramdisk_path)
```
This is straightforward: it utilizes the `umount` command via `subprocess` to remove the mounted RAM disk, preventing potential issues with subsequent mounting or resource usage. It's good practice to always explicitly unmount at the end of a script or session that uses this setup.

It is also worth noting that the RAM disk is volatile, meaning that its contents are lost when the Colab instance is terminated. Therefore, it's imperative to use the RAM disk for caching data that can be easily reloaded from the original source, rather than storing anything that needs to be persistent. In addition, you should always make sure the RAM usage stays within the constraints of your Colab environment. If your data is too large to fit in RAM, using generators or other techniques to load subsets of the data is essential to avoid crashes.

To further explore this, consider reviewing the official documentation on Linux file systems, and also examining material on TensorFlow's data API, particularly `tf.data.Dataset`, to enhance understanding of how to manage large datasets efficiently within a training environment. Moreover, researching techniques on preprocessing large datasets efficiently with Python libraries can greatly improve the overall pipeline speed.
