---
title: "How can I train an LSTM on multiple time series datasets?"
date: "2025-01-30"
id: "how-can-i-train-an-lstm-on-multiple"
---
Training an LSTM (Long Short-Term Memory) network on multiple time series datasets presents a unique challenge because the data's statistical characteristics, temporal patterns, and even scales might differ significantly across datasets. Directly concatenating these disparate series and feeding them into a single LSTM is often suboptimal, leading to poor generalization and model instability. The key is to address the heterogeneity by either adapting the input data or the model architecture, or a combination of both.

I've personally grappled with this while developing a predictive maintenance system for various industrial machines, each exhibiting unique operational patterns recorded via sensor data. A naive approach of combining these datasets resulted in a model that struggled to generalize and predict accurately for any particular machine. Based on this experience, I've found that effective approaches often revolve around feature engineering, data normalization, and potentially using an ensemble of LSTMs or adapting the network architecture itself.

A primary consideration is how the diverse time series are structured. Are they all of the same length? If not, we need to adopt a method to align them. Common techniques here include zero-padding the shorter series or truncating the longer ones. Padding with zeros, while straightforward, might introduce artificial patterns that can skew the training process. Truncation, on the other hand, risks discarding valuable data. A more refined approach involves dynamically partitioning the data, creating subsequences of fixed length, an approach commonly used in natural language processing with recurrent networks.

Another crucial aspect is feature scaling. Each dataset might have features measured on different scales. For example, temperature readings might range between -20 and 50 degrees Celsius, while pressure readings could range between 0 and 1000 PSI. Feeding these directly to the LSTM without normalization is problematic. Features with large magnitude values will dominate the optimization process, preventing the model from learning patterns from the features with smaller values. Techniques like standardization (subtracting the mean and dividing by the standard deviation for each feature in each dataset) or min-max scaling (linearly transforming values between 0 and 1) are essential before feeding the data into the LSTM. I found that standardization was particularly effective when dealing with sensor data having varying units and ranges.

Now, let us examine approaches with code examples. Assume the data is already organized as a list of time series, where each time series is a NumPy array representing a sequence of observations with potentially multiple features. Each time series has length `time_steps` and number of features `num_features`.

**Example 1: Standardized Data with Fixed-Length Sequences**

This example shows how to standardize each time series and create subsequences suitable for LSTM input using TensorFlow and Keras.

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import StandardScaler

def prepare_timeseries_data(time_series_data, time_steps):
    """Preprocesses time series data.

    Args:
        time_series_data: A list of numpy arrays, each representing a time series.
        time_steps: The length of each subsequence to create.

    Returns:
        A tuple: (concatenated input data, list of number of sequences per time series)
    """
    scaled_data = []
    num_sequences_per_series = []

    for series in time_series_data:
        # 1. Standardize
        scaler = StandardScaler()
        scaled_series = scaler.fit_transform(series)
        scaled_data.append(scaled_series)

        # 2. Create subsequences
        data_generator = TimeseriesGenerator(scaled_series, scaled_series,
                                             length=time_steps, batch_size=1,
                                             sampling_rate=1, stride=1)
        num_sequences_per_series.append(len(data_generator))
    
    # 3. Concatenate all sequences, maintaining sequence order
    concatenated_input = np.concatenate([gen[0] for gen in 
                                       [TimeseriesGenerator(scaled_data[i], scaled_data[i], length=time_steps, batch_size=1) 
                                       for i in range(len(scaled_data))]], axis=0)
    return concatenated_input, num_sequences_per_series

# Example Usage
num_time_series = 3
time_steps = 10
num_features = 5

time_series_data = [np.random.rand(100, num_features) for _ in range(num_time_series)]
concatenated_input, num_sequences_per_series = prepare_timeseries_data(time_series_data, time_steps)

print(f"Shape of concatenated data: {concatenated_input.shape}")
print(f"Number of sequences per series: {num_sequences_per_series}")
```

This function first standardizes each time series using `StandardScaler`. Then, it utilizes `TimeseriesGenerator` to create subsequences of length `time_steps`. Finally, it concatenates all subsequences and returns both the concatenated data and the number of subsequences from each original time series. Note how we preserve the order when concatenating, which is essential for correct model training.

**Example 2: LSTM Network Definition**

Once the data is prepared as in Example 1, an LSTM network can be trained with the data. Here's an example of a simple LSTM network:

```python
def create_lstm_model(time_steps, num_features, hidden_units):
    """Creates a simple LSTM model.

    Args:
        time_steps: The length of input sequence.
        num_features: Number of input features.
        hidden_units: Number of hidden units in the LSTM layer.

    Returns:
        A compiled keras model.
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(hidden_units, activation='tanh', input_shape=(time_steps, num_features)),
        tf.keras.layers.Dense(num_features)  # Output size should match num_features if predicting the next step in the series
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Example Usage
hidden_units = 64
model = create_lstm_model(time_steps, num_features, hidden_units)
# Assume 'concatenated_input' and 'concatenated_target' exist from previous code.
target = concatenated_input[:,:,:]
model.fit(concatenated_input, target, epochs=10, verbose=0) #Train the model
print("LSTM model trained successfully")
```

This defines a basic LSTM model with one LSTM layer followed by a dense layer. The `input_shape` is determined by the `time_steps` and `num_features` in the dataset. It’s crucial that the output layer dimensions match the prediction target which in this case we assume to be the same as the input at the next step, making it a time-series prediction task.

**Example 3: Custom Loss Function Addressing Data Source**

In some cases, it's useful to inject awareness of the data's source into the training process. If the datasets have significantly different characteristics, it might be beneficial to emphasize that through the loss function. This can be accomplished by weighting the loss based on the dataset origin using the sequence index:

```python
def custom_weighted_loss(num_sequences_per_series):
    """Creates a custom loss function with dataset weights.

    Args:
        num_sequences_per_series: A list of the number of sequences per time series.

    Returns:
        A function that computes the weighted MSE.
    """
    weights = []
    for num_seq in num_sequences_per_series:
        weights.extend([1/num_seq for _ in range(num_seq)])
    weights = np.array(weights,dtype=np.float32)
    
    def weighted_mse(y_true, y_pred):
        loss = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
        weighted_loss = loss * weights
        return tf.reduce_mean(weighted_loss)
    return weighted_mse

# Example Usage
custom_loss = custom_weighted_loss(num_sequences_per_series)
model.compile(optimizer='adam', loss=custom_loss)
# Assume 'concatenated_input' and 'concatenated_target' exist from previous code.
model.fit(concatenated_input, target, epochs=10, verbose=0)
print("LSTM model trained with custom loss")
```

This approach creates a custom loss function that weights the MSE of each sequence by the inverse of how many sequences were drawn from the original time series. The goal is to prevent a single high-volume time-series from dominating the loss calculation. Each series has equal weight in the loss during training. In real-world applications, the weights should be carefully designed to address the specific characteristics of the time series.

In addition to these code-oriented strategies, one can explore more complex techniques. Attention mechanisms within LSTMs can allow the network to prioritize certain parts of the sequence, which may be particularly useful in time-series having varying levels of temporal significance. Transfer learning could also be employed where an LSTM trained on a very large time-series dataset is then fine-tuned on the data of the smaller datasets. Furthermore, an ensemble of models, each trained on a different set of time series data, could improve overall predictive performance by leveraging the diversity of the models. I've found the key is iterative experimentation, beginning with simple approaches before moving to advanced ones.

For further study, textbooks on time-series analysis, machine learning, and deep learning, along with research papers in these fields can be highly informative. Specifically, look for papers on multi-variate time-series forecasting and recurrent neural networks. Additionally, tutorials and blog posts that cover practical aspects of Keras and Tensorflow are invaluable resources. Finally, gaining a solid theoretical foundation in signal processing can provide a more profound understanding of the underlying nature of time-series data.
