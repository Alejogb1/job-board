---
title: "What data adapter can handle TimeseriesGenerator input?"
date: "2025-01-30"
id: "what-data-adapter-can-handle-timeseriesgenerator-input"
---
Specifically, how can you utilize this adapter within a Keras model, and what considerations should one make when preprocessing data for time series analysis?
\
`TimeseriesGenerator` in Keras presents a structured way to prepare time series data for sequential model training; however, it is not directly compatible with standard model inputs. The adapter that effectively bridges this gap is the `tf.data.Dataset` object, particularly when generated using `tf.data.Dataset.from_generator`. This allows the generator’s output, which yields batches of time series sequences and their corresponding targets, to be directly consumed by a Keras model.

Having implemented several time series forecasting projects, I've found that understanding this interface is fundamental. When we instantiate a `TimeseriesGenerator`, it does not produce a data structure directly consumed by Keras `fit()` method. Instead, it generates data on demand when iterated over. The `tf.data.Dataset` encapsulates data and provides functionalities for batching, shuffling, and preprocessing, which become crucial for efficient training. The crucial element here is the conversion of the `TimeseriesGenerator` into a `tf.data.Dataset` which effectively serves as the bridge.

Here's how to incorporate it:

First, we define our `TimeseriesGenerator`. Let's assume we have time series data, `data`, with a defined sequence length `length`, and a sampling stride of one. We also have a `batch_size` parameter.

```python
import numpy as np
from tensorflow import keras

# Sample time series data (replace with actual data)
data = np.arange(100, dtype=float)
length = 10
batch_size = 3

# Initialise TimeseriesGenerator
generator = keras.preprocessing.sequence.TimeseriesGenerator(
    data, data, length=length, sampling_rate=1, batch_size=batch_size
)
```

Here, the `data` variable holds a simple sequence. `length` dictates the size of the input window and the generator will transform data into batches each of a specific length. Crucially, because this is configured to be a prediction problem where the past values are used to predict the future, input data *is* the target data for the generator.

The next step requires using a custom function, `generator_wrapper` in conjunction with  `tf.data.Dataset.from_generator`, to adapt the generator for Keras:

```python
import tensorflow as tf

def generator_wrapper(gen):
    while True:
        yield next(gen)

dataset = tf.data.Dataset.from_generator(
    generator_wrapper,
    args=[generator],
    output_signature=(
        tf.TensorSpec(shape=(None, length, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 1), dtype=tf.float32)
    )
)
```

The `generator_wrapper` function serves as an iterator, ensuring that the generator's data is continually available to the dataset without exhaustion. The `output_signature` defines the shapes and types of the generated data, explicitly declaring a three-dimensional input tensor with a variable number of batches, sequence length equal to the window size, and a channel dimension of one, representing the single time series. The target output is a two-dimensional tensor with a variable number of batches and a single prediction value. We assume the `data` has to be in a single channel (converted to this form later) hence the shape `(None, length, 1)`.  Without these, the dataset object is unable to effectively manage the data stream for training.

Finally, this `dataset` object can be used to train our Keras model.  I will now create a very simple Keras model:

```python
model = keras.models.Sequential([
    keras.layers.LSTM(32, activation='relu', input_shape=(length, 1)),
    keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')
```

Here, we are using a simple LSTM architecture suitable for time series data. The `input_shape` parameter aligns with the input shape declared within our `output_signature` of the dataset.  Now, we can train using our created `dataset` object.

```python
model.fit(dataset, epochs=5, steps_per_epoch=len(generator))

```

The `steps_per_epoch` is calculated based on the total number of batches the generator produces within one complete cycle over the original data.

Preprocessing is critical in time series analysis. Based on my experience, the first step almost always involves reshaping and scaling the data. The initial data must be reshaped into a suitable form to align with how `TimeseriesGenerator` processes it, usually a two-dimensional structure, with each observation being a feature. Then, this is reshaped during the dataset conversion as we did earlier. For example, single channel data of 1 dimension needs to become 2 dimensional before being transformed by the `TimeseriesGenerator` with the final dimension reshaped as a single channel of `(None, length, 1)`. This reshaping can be handled prior to the generator itself, using numpy’s reshape function.

Next is scaling. Applying techniques such as standardization or min-max scaling is usually essential. Standardizing data (mean zero, unit variance) is often my go-to technique, calculated on the training data to avoid information leakage from the test set. However, min-max scaling may suit some use cases where data is tightly bound in a specific range. If there is seasonality involved, other transformations might be applicable. For example, applying a logarithmic function or a difference can help remove non-stationarity.

Moreover, dealing with missing values is critical. I tend to use linear interpolation for short gaps, or forward or backward fill if interpolation is not suitable. If large gaps are present, consider dropping these regions as they may introduce noise. The decision depends on the nature of the data and the specific task. Feature engineering can dramatically improve model performance. This involves, for example, creating lagged features (previous time steps), rolling statistics (mean, standard deviation) over a time window, and other domain-specific transformations.

One needs to carefully consider the use of training, validation, and testing splits. Random splits are not usually suitable for time series; a temporal split (e.g., using past data for training, future data for validation and testing) is required. The temporal split needs to be done prior to data processing such as scaling, so that you do not expose the model to information from the testing set during training and validation. Data leakage can lead to over-optimistic performance on validation sets and poor generalization on unseen testing data.

The choice of the window size (`length`) is also important. It should be long enough to capture sufficient historical information for prediction, but not too long that the model struggles to learn patterns, or the computational cost becomes prohibitive. This hyperparameter needs to be tuned empirically, sometimes in conjunction with the choice of model architecture.

Finally, careful attention to error metrics is important. Root mean squared error is usually what I use when assessing regression metrics. However, metrics such as mean absolute percentage error might be preferable in other cases, especially where it is important to gauge proportional errors instead of absolute differences. The nature of the error needs to be assessed to choose the right metric to evaluate model quality.

In terms of further learning, I recommend exploring “Forecasting: Principles and Practice” by Hyndman and Athanasopoulos. This book offers a wide view on time series analysis and is available online for free. For a deeper dive into deep learning aspects, I suggest “Deep Learning with Python” by Chollet. Also, the Keras documentation and Tensorflow website are invaluable resources for staying updated on the latest practices in the framework.
