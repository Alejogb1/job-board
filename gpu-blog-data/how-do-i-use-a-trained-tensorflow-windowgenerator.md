---
title: "How do I use a trained TensorFlow WindowGenerator for time series prediction?"
date: "2025-01-30"
id: "how-do-i-use-a-trained-tensorflow-windowgenerator"
---
Time series forecasting with TensorFlow's `WindowGenerator` involves a careful dance of data preparation and model feeding. Having spent the last few years building predictive maintenance systems, I’ve found that understanding the nuances of how this class structures input for time-based models is paramount. The `WindowGenerator` doesn't directly perform prediction itself; instead, it's a pre-processing tool that transforms raw time series data into manageable input windows for models like Recurrent Neural Networks (RNNs) or Transformers. The core idea is to create sequences of past data (`inputs`) and their corresponding future values (`labels`) for training.

The `WindowGenerator` essentially crafts these input-label pairs from a single time series. You don't directly use a pre-trained model on the initial dataset, instead you use your existing time series data and the generator is a part of how you format data to fit the trained model. I will illustrate how a generator takes your dataset and produces these time-windows, formatted for use in training and subsequent prediction with a model. Here’s a breakdown:

**1. Window Generation Logic**

The `WindowGenerator` primarily works around the concepts of:

*   **`input_width`**: The length of the input time window. This is the number of historical data points the model will see.
*   **`label_width`**: The length of the prediction window, i.e., how many steps into the future you want to predict.
*   **`shift`**: How many time steps separate the end of the input window and the start of the label window. A `shift` of 1 is typical for single-step forecasting.

Given a time series of length *N*, the generator creates several windows. Each window contains `input_width` time steps of historical data (your inputs), and the model will be trained to predict the `label_width` values based on that input. This is the essence of moving data from raw timestamps to training/prediction sequences. Crucially, these windows are overlapping. A key consideration here is that you would not necessarily create new windows for every timestamp. If the number of input steps are larger than the shift value, you will have overlapping sequences of values. The `WindowGenerator` encapsulates the process of generating these overlapping sequences in a performant and controlled manner.

**2. Using `WindowGenerator`**

The common workflow is to create a `WindowGenerator` instance, provide it with your data, and use it to iterate over generated windows which can then be provided to TensorFlow models for training.

*   **Instantiation:** You initialize a `WindowGenerator` with your specific `input_width`, `label_width`, and `shift`.
*   **Data Assignment:** You pass the `WindowGenerator` either a NumPy array or a TensorFlow Dataset. In my experience, using TensorFlow datasets directly is usually more efficient as it avoids creating large in-memory arrays.
*   **Data Transformation**: The generator then splits and reorganizes data into input-label windows using TensorFlow operations.
*   **Model Consumption:** This generated data is then compatible for use with TensorFlow Keras models such as RNNs, Transformers, and even dense neural networks. This is not a "training function" or "model function" in itself but facilitates getting the data into the model efficiently.

**3. Code Examples**

Let's illustrate with a few examples using synthetic time series data. Imagine I had sensor readings from a factory floor over a series of timestamps. We will generate this as a NumPy array.

**Example 1: Basic Window Creation**

```python
import numpy as np
import tensorflow as tf

# Create sample time series data (N=100)
data = np.arange(100, dtype=np.float32)

# Define window parameters
input_width = 10
label_width = 5
shift = 1

# Instantiate the WindowGenerator
window_gen = tf.keras.preprocessing.sequence.TimeseriesGenerator(
    data,
    data,
    length=input_width,
    sampling_rate=1,
    stride=1,
    start_index=0,
    end_index=None,
    shuffle=False,
    reverse=False,
    batch_size=1,
    label_start_index=input_width + shift,
    label_end_index=input_width + shift + label_width
)

# Get a single batch
x,y = window_gen[0]

print("Input Sequence:")
print(x)

print("Label Sequence:")
print(y)
```

In this example, I use `TimeseriesGenerator`, which allows generating sequences for models directly from the data. Here, `length` is `input_width`, and the label sequences are taken from the indices defined by `label_start_index` and `label_end_index`. Notice the input and output sequences are generated directly. This generator uses the same parameters as the `WindowGenerator`, but provides training sequences directly. We are grabbing the first sequence via index `[0]`. I've intentionally kept batch_size at one to highlight individual data samples. A key takeaway is that this is not the model training function, but only the function for preparing the training data as a `tf.Dataset`

**Example 2: Multi-step Forecast**

```python
import numpy as np
import tensorflow as tf

data = np.arange(100, dtype=np.float32)

input_width = 10
label_width = 5
shift = 5

window_gen = tf.keras.preprocessing.sequence.TimeseriesGenerator(
    data,
    data,
    length=input_width,
    sampling_rate=1,
    stride=1,
    start_index=0,
    end_index=None,
    shuffle=False,
    reverse=False,
    batch_size=3,
    label_start_index=input_width + shift,
    label_end_index=input_width + shift + label_width
)

# Iterate through a batch of data
for x, y in window_gen:
    print("Input Sequence Shape:", x.shape)
    print("Label Sequence Shape:", y.shape)
    break
```

Here, `shift` is set to 5, meaning the label window starts 5 time steps after the input window ends. The label window is still of length 5. The `batch_size` is now set to 3, so each iteration now yields a batch of 3 input-label pairs, each correctly generated based on the defined parameters. This is a common use case for generating batches that can be used in the training loop of a model. Note, these batches are not shuffled by default. This may need to be changed for specific uses.

**Example 3: Using a Model (Conceptual)**

While `WindowGenerator` doesn't train models, here’s how one might conceptually feed the data into one. This uses the same generator as in Example 2:

```python
import numpy as np
import tensorflow as tf

data = np.arange(100, dtype=np.float32)

input_width = 10
label_width = 5
shift = 5

window_gen = tf.keras.preprocessing.sequence.TimeseriesGenerator(
    data,
    data,
    length=input_width,
    sampling_rate=1,
    stride=1,
    start_index=0,
    end_index=None,
    shuffle=False,
    reverse=False,
    batch_size=3,
    label_start_index=input_width + shift,
    label_end_index=input_width + shift + label_width
)

# Assume you have a trained model called 'model'
model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(32, input_shape=(input_width, 1)),
    tf.keras.layers.Dense(label_width)
])

model.compile(optimizer='adam', loss='mse')

# Example training loop
for x_batch, y_batch in window_gen:
    # Need to add a final dimension to input data (batch_size, time_steps, features)
    x_batch = tf.expand_dims(x_batch, -1)
    model.fit(x_batch, y_batch, verbose=0) # 'verbose' is not used here

# Example prediction using the last window from the generator (assuming there is still a batch remaining in the generator
# you can replace this with a new set of raw inputs)
x_batch, _ = window_gen[-1]
x_batch = tf.expand_dims(x_batch, -1)
predictions = model.predict(x_batch)
print("Predictions:", predictions)

```

I’ve created a simple LSTM network that matches the shape of the data. Notice I need to add a feature dimension (dimension 2, the last dimension) using `tf.expand_dims`, which allows our training data to match the `input_shape` of the layer. The training loop now takes data from the `TimeseriesGenerator` object. The final part shows the conceptual way you might grab a set of data and use the trained model to perform a prediction with the trained model.

**4. Post-Training Prediction**

After training, you use the same `WindowGenerator` to create windows from new (or unseen) time series data for predictions. Importantly, the `input_width` and the `shift` parameters must be identical to when the generator was created for training. You would generate windows from your new dataset, and feed these directly into the `.predict()` method of your trained TensorFlow model.

**Resource Recommendations**

For a deeper understanding of time series analysis, I recommend exploring materials on signal processing and control theory. Textbooks covering these areas will provide a valuable theoretical framework. Additionally, consulting the official TensorFlow documentation on `tf.data` and `tf.keras.preprocessing.sequence` will allow for further understanding and tuning of the provided solutions. These will help improve your approach to time series data processing using tools like the `TimeseriesGenerator` and TensorFlow models.
