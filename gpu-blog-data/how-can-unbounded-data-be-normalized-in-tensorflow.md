---
title: "How can unbounded data be normalized in TensorFlow?"
date: "2025-01-30"
id: "how-can-unbounded-data-be-normalized-in-tensorflow"
---
Real-time data streams, characterized by their continuous and potentially infinite nature, present significant challenges for traditional machine learning workflows that assume fixed, finite datasets. When dealing with unbounded data in TensorFlow, normalization, typically a straightforward process, requires a fundamentally different approach than batch-based normalization. I've wrestled with this problem in several projects involving live sensor data and user interaction streams, and I've found that moving average-based normalization strategies are essential for stable model training.

The core issue is that static normalization, such as min-max scaling or z-score standardization, relies on statistics computed over the entire dataset. These statistics, like the global mean and standard deviation, are unavailable or ever-changing with unbounded streams. Applying batch statistics, while feasible, introduces instability. The mean and standard deviation of incoming data can fluctuate dramatically from batch to batch, leading to wildly shifting normalization parameters that destabilize model convergence. Instead, we need to employ a method that adapts over time, capturing the overall distribution while smoothing out the inherent variability in the incoming data. This is where moving average techniques come into play.

Moving averages allow us to compute a running estimate of the mean and standard deviation, using past data to inform present normalization values. There are generally two main types of moving averages that we can apply: exponential moving averages (EMA) and simple moving averages (SMA). Both utilize a 'decay' or 'window' parameter that determines the contribution of older data points to current estimates.

**Explanation of Exponential Moving Average (EMA)**

I typically use the EMA, which I've found to be more responsive to recent changes in data distribution while retaining a memory of the past. It updates its estimate incrementally for each incoming data point. The core equation for EMA of mean and standard deviation (in TensorFlow pseudocode) is:

*   `mean_ema = decay * mean_ema_prev + (1 - decay) * current_mean`
*   `std_ema = decay * std_ema_prev + (1 - decay) * current_std`

Here, `decay` is a value between 0 and 1, controlling how quickly past data influence current estimates. A higher `decay` (closer to 1) results in slower adaptation, emphasizing past information; a lower `decay` (closer to 0) results in faster adaptation, emphasizing recent information. The `current_mean` and `current_std` are calculated from the latest batch of input data.  Crucially, these moving average statistics `mean_ema` and `std_ema` are used for normalization, not the batch statistics themselves.  This approach provides a much more stable normalization over time, as it filters out short-term fluctuations and provides a more consistent picture of the distribution of data.

**Code Examples**

Here are three code examples illustrating this concept in TensorFlow, with varying degrees of complexity:

**Example 1: Basic EMA for a Single Feature**

```python
import tensorflow as tf

class EMANormalizer(tf.Module):
  def __init__(self, decay=0.99):
    self.decay = decay
    self.mean_ema = tf.Variable(0.0, dtype=tf.float32)
    self.std_ema = tf.Variable(1.0, dtype=tf.float32)  # Initialize with a non-zero std

  @tf.function
  def normalize(self, x):
    current_mean = tf.reduce_mean(x)
    current_std = tf.math.reduce_std(x)

    # Compute EMA
    self.mean_ema.assign(self.decay * self.mean_ema + (1 - self.decay) * current_mean)
    self.std_ema.assign(self.decay * self.std_ema + (1 - self.decay) * current_std)
    
    # Ensure standard deviation is not zero to prevent division by zero
    std_for_normalize = tf.maximum(self.std_ema, 1e-8)

    normalized_x = (x - self.mean_ema) / std_for_normalize
    return normalized_x

# Example Usage:
normalizer = EMANormalizer(decay=0.9)
data_stream = [tf.random.normal((10,)) for _ in range(100)] # Simulate data stream
for batch in data_stream:
    normalized_batch = normalizer.normalize(batch)
    # Use normalized_batch in training
```
*Commentary:* This first example demonstrates the basic implementation of an EMA-based normalizer for a single, 1-dimensional feature. I have initialised standard deviation with 1.0 to prevent division by zero initially. The `@tf.function` decorator optimizes performance for TensorFlow computations. `tf.maximum` guards against std close to zero. This is a typical implementation that I would apply for features that are scalar values, where the reduction on the feature itself will provide the mean and standard deviation we need.

**Example 2: EMA for Multiple Features**

```python
import tensorflow as tf

class MultiFeatureEMANormalizer(tf.Module):
    def __init__(self, num_features, decay=0.99):
        self.decay = decay
        self.mean_ema = tf.Variable(tf.zeros(num_features, dtype=tf.float32))
        self.std_ema = tf.Variable(tf.ones(num_features, dtype=tf.float32))

    @tf.function
    def normalize(self, x):
        current_mean = tf.reduce_mean(x, axis=0)
        current_std = tf.math.reduce_std(x, axis=0)

        self.mean_ema.assign(self.decay * self.mean_ema + (1 - self.decay) * current_mean)
        self.std_ema.assign(self.decay * self.std_ema + (1 - self.decay) * current_std)

        std_for_normalize = tf.maximum(self.std_ema, 1e-8)
        normalized_x = (x - self.mean_ema) / std_for_normalize
        return normalized_x


# Example Usage:
num_features = 5
normalizer = MultiFeatureEMANormalizer(num_features, decay=0.95)
data_stream = [tf.random.normal((10, num_features)) for _ in range(100)]
for batch in data_stream:
    normalized_batch = normalizer.normalize(batch)
    # Use normalized_batch in training
```
*Commentary:* This example extends the previous one to handle multi-dimensional features. Here, the mean and standard deviation are computed across the batch dimension (axis=0) for each feature. `num_features` is passed in the constructor to initialize the moving average variables with the correct dimension, using `tf.zeros` and `tf.ones`. I often encounter this when working with time-series data where there are multiple sensor readings at the same time step.

**Example 3: EMA with Stateful Data Storage**

```python
import tensorflow as tf

class StatefulEMANormalizer(tf.Module):
  def __init__(self, decay=0.99, initial_mean = 0.0, initial_std = 1.0):
    self.decay = decay
    self.mean_ema = tf.Variable(initial_mean, dtype=tf.float32)
    self.std_ema = tf.Variable(initial_std, dtype=tf.float32)
    self.num_samples = tf.Variable(0, dtype=tf.int32) # Keeping track of processed samples

  @tf.function
  def normalize(self, x):
     # Compute the batch-wise statistics and number of elements
    current_mean = tf.reduce_mean(x)
    current_std = tf.math.reduce_std(x)
    num_elems = tf.cast(tf.size(x), tf.int32)

    # Update the EMA of the statistics
    self.mean_ema.assign(self.decay * self.mean_ema + (1 - self.decay) * current_mean)
    self.std_ema.assign(self.decay * self.std_ema + (1 - self.decay) * current_std)
    self.num_samples.assign_add(num_elems)

    # Calculate the normalized x
    std_for_normalize = tf.maximum(self.std_ema, 1e-8)
    normalized_x = (x - self.mean_ema) / std_for_normalize
    return normalized_x, self.num_samples # returns number of samples

  def get_ema_stats(self):
    return self.mean_ema, self.std_ema

# Example Usage:
normalizer = StatefulEMANormalizer(decay=0.95)
data_stream = [tf.random.normal((tf.random.uniform(shape=[], minval=5, maxval=20, dtype=tf.int32),)) for _ in range(100)]
for batch in data_stream:
    normalized_batch, num_samples = normalizer.normalize(batch)
    # Use normalized_batch in training
    # Check number of samples processed:
    if(num_samples % 1000 == 0):
      print(f"Processed number of samples: {num_samples}")
      mean, std = normalizer.get_ema_stats()
      print(f"Mean: {mean}, Std: {std}")
```
*Commentary:* This final example incorporates a counter (`num_samples`) to track how much data has been processed. I found this particularly useful in long-running experiments where tracking the effective amount of data that has influenced the moving average is important for ensuring proper learning. Also included is a function, `get_ema_stats()`, to easily retrieve the latest estimates of the mean and standard deviation. This is helpful when monitoring the convergence of the statistics.  This example also simulates variable batch sizes to emphasize its adaptability to non-uniform data streams. Note that an initial mean and standard deviation can be passed in the constructor.

**Resource Recommendations**

When implementing these techniques, I would recommend exploring research papers and open-source frameworks that address real-time data processing. Look for resources covering topics like:

*   **Online learning:** These resources discuss model training methods that are designed to handle data streams.
*   **Adaptive batch normalization:** Explore advanced variants of batch normalization that are designed to work with dynamically changing batch statistics.
*   **Streaming data analytics:** This provides a more general context of handling unbounded data.
*   **Best practices for implementing continuous integration and delivery:** Useful for applying trained models into production for live inference.

By combining this theoretical knowledge with hands-on experimentation, you can effectively apply EMA normalization to your unbounded datasets in TensorFlow, paving the way for robust and stable machine learning applications.
