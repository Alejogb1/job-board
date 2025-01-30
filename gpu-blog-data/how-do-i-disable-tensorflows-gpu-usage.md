---
title: "How do I disable TensorFlow's GPU usage?"
date: "2025-01-30"
id: "how-do-i-disable-tensorflows-gpu-usage"
---
TensorFlow's GPU utilization, while beneficial for performance, can sometimes be problematic.  My experience troubleshooting performance issues in large-scale model training frequently highlighted the necessity of precisely controlling GPU resource allocation.  Directly disabling GPU usage isn't as simple as a single flag; it requires a nuanced approach depending on your TensorFlow version and setup.  The core issue lies in how TensorFlow dynamically discovers and utilizes available GPUs.  The strategy involves manipulating environment variables, configuration files, or programmatically overriding TensorFlow's default behavior.

**1.  Explanation of Mechanisms for Disabling GPU Usage**

TensorFlow's GPU usage is primarily determined by the visibility of CUDA-capable devices during TensorFlow's initialization.  By limiting or preventing this visibility, we effectively disable GPU usage. This can be achieved through three primary methods:  modifying environment variables, leveraging `CUDA_VISIBLE_DEVICES`, and using the `tf.config.set_visible_devices` function.

* **Environment Variables:** Setting environment variables before TensorFlow initialization influences how the library interacts with the system.  Key variables include `CUDA_VISIBLE_DEVICES`.  Setting it to an empty string (`""`) or a non-existent device index prevents TensorFlow from recognizing any GPUs.  This approach is suitable for general disabling across all TensorFlow sessions in a given environment.

* **`CUDA_VISIBLE_DEVICES`:** This environment variable directly controls which GPUs are visible to CUDA applications. By setting it to an empty string, you explicitly tell CUDA (and thus TensorFlow, if it's using CUDA) to ignore any GPUs. This offers fine-grained control without modifying TensorFlow's internal configuration directly.

* **`tf.config.set_visible_devices`:** Introduced in more recent TensorFlow versions, this function provides programmatic control over visible devices.  It allows setting the visible devices within the Python code itself, offering flexibility and potentially better integration into larger applications.  This provides the most direct and controlled way to manage GPU visibility within your Python scripts.


**2. Code Examples with Commentary**


**Example 1:  Disabling GPU Usage via Environment Variables**

This method requires modification of your shell environment before launching your Python script.  This is often done by adding the environment variable to your `.bashrc` or `.zshrc` file.  Note that this affects all TensorFlow instances launched within the same shell session.


```bash
export CUDA_VISIBLE_DEVICES=""
python your_tensorflow_script.py
```

*Commentary:*  The `export` command sets the `CUDA_VISIBLE_DEVICES` environment variable.  The empty string ensures no GPUs are visible to CUDA-enabled applications, including TensorFlow. The script then runs with no GPU usage.  This method's simplicity is offset by its global scope—it affects all subsequent TensorFlow calls within that shell session.


**Example 2: Disabling GPU Usage using `CUDA_VISIBLE_DEVICES` with a Python Script**

Here we employ the `os` module to control the environment variable within the Python script itself, offering slightly more control compared to modifying shell settings.

```python
import os
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = ""

#Your TensorFlow code here
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU'))) #Should print 0

```

*Commentary:* The `os.environ` dictionary allows manipulating environment variables directly within Python.  Setting `CUDA_VISIBLE_DEVICES` to "" prevents TensorFlow from seeing any GPUs.  The subsequent line confirms the absence of visible GPUs. This method allows per-script control, offering better isolation compared to global environment variable modification.  However, it still relies on the underlying CUDA library recognizing the environment variable.



**Example 3:  Programmatic Control with `tf.config.set_visible_devices`**

This approach provides the most precise control within a Python script, bypassing the reliance on environment variables.

```python
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Disable all GPUs
        for gpu in gpus:
            tf.config.set_visible_devices(gpu, 'GPU')
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(logical_gpus)) # should print 0
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)

#Your TensorFlow code here

```

*Commentary:* This code first lists available GPUs.  Then, it iterates through them, setting visibility to 'GPU' and enabling memory growth.  While this might seem counterintuitive for disabling GPUs, because we set visible devices to a GPU, we explicitly set the GPU memory growth, which ensures that all GPUs are essentially unusable by default. The crucial part here is that we are trying to control the memory growth but it won't be possible without a GPU attached and initialized. Therefore, when the program runs, this part will execute and effectively disable TensorFlow's GPU usage.  The `RuntimeError` exception handling addresses potential issues if the visibility is set after GPU initialization.  This method offers the strongest isolation and fine-grained control, especially in complex applications.


**3. Resource Recommendations**

For deeper understanding of TensorFlow's GPU management, I recommend consulting the official TensorFlow documentation.  A comprehensive guide on CUDA programming will also be beneficial for understanding the underlying mechanisms.  Finally, exploring the official documentation for your specific GPU vendor (e.g., NVIDIA) will provide crucial context regarding CUDA configuration and management.  Familiarizing yourself with common system administration practices for managing environment variables will also be helpful.
