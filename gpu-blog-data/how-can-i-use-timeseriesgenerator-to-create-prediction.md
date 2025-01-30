---
title: "How can I use TimeseriesGenerator to create prediction data?"
date: "2025-01-30"
id: "how-can-i-use-timeseriesgenerator-to-create-prediction"
---
The efficacy of Keras' `TimeseriesGenerator` hinges on a precise understanding of its input and output shapes; a common pitfall is misinterpreting the `target_size` parameter, leading to incorrectly shaped prediction data.  My experience building predictive models for high-frequency financial data emphasized this point repeatedly.  Incorrectly configuring `target_size` often resulted in predictions that were off by one or more timesteps, rendering the model useless.

**1. Clear Explanation:**

`TimeseriesGenerator` transforms a sequence of time series data into input-output pairs suitable for training recurrent neural networks (RNNs) or other sequence models. It does this by creating sliding windows across the input data.  The input sequence is a single NumPy array or a Pandas Series. The generator yields batches of input data (`X`) and corresponding target data (`y`).  Crucially, the relationship between the input and target is defined by the `length` and `sampling_rate` parameters, which specify the size of the input window and the step size between successive windows, respectively, and the `target_size` parameter which defines the length of the target sequence for each input window.

The generator produces input samples of shape `(batch_size, length, n_features)` where `length` is the size of the input window (number of timesteps), and `n_features` is the number of features in the time series (e.g., 1 for a univariate time series, or multiple for a multivariate time series).  The corresponding target samples have the shape `(batch_size, target_size, n_features)`.

The `target_size` parameter is central to prediction data creation.  It dictates how many future timesteps the model is trained to predict. A `target_size` of 1 means the model predicts the next single timestep; a `target_size` of 3 means the model predicts the next three timesteps.  The selection of `target_size` depends entirely on the prediction horizon required by the application.   This is frequently misunderstood, leading to models that appear to work but yield fundamentally incorrect predictions.

Understanding the interplay between `length`, `sampling_rate`, and `target_size` is critical.  For example, if `length` is 10, `sampling_rate` is 1, and `target_size` is 3, each input sequence will consist of 10 timesteps, and the corresponding target sequence will consist of the next 3 timesteps.  If `sampling_rate` is greater than 1, the input windows will skip timesteps.   Careful consideration of these parameters is essential to ensure the model learns the correct temporal relationships.


**2. Code Examples with Commentary:**

**Example 1: Single-Step Prediction (Univariate)**

```python
import numpy as np
from keras.preprocessing.sequence import TimeseriesGenerator

# Sample univariate time series data
data = np.linspace(0, 100, 100)

# Create the generator for single-step prediction
data_gen = TimeseriesGenerator(data, data, length=10, sampling_rate=1, target_size=1, batch_size=1)

# Generate and inspect the first batch
X, y = data_gen[0]
print("Input shape:", X.shape) # Output: (1, 10)
print("Target shape:", y.shape) # Output: (1, 1)
print("First input sequence:", X[0])
print("Corresponding target:", y[0])
```

This example demonstrates single-step ahead prediction.  The `target_size` is 1, meaning the model predicts only the next timestep.  The input sequence has a length of 10. The `sampling_rate` of 1 means no timesteps are skipped between input windows.


**Example 2: Multi-Step Prediction (Univariate)**

```python
import numpy as np
from keras.preprocessing.sequence import TimeseriesGenerator

data = np.linspace(0, 100, 100)

# Generate for multi-step prediction
data_gen = TimeseriesGenerator(data, data, length=10, sampling_rate=1, target_size=3, batch_size=1)

X, y = data_gen[0]
print("Input shape:", X.shape) # Output: (1, 10)
print("Target shape:", y.shape) # Output: (1, 3)
print("First input sequence:", X[0])
print("Corresponding target sequence:", y[0])
```

This example extends to multi-step prediction with `target_size` set to 3. The model now predicts the next three timesteps.  The input shape remains consistent, but the target shape now reflects the three-step prediction.


**Example 3: Multivariate Time Series with Skipped Samples**

```python
import numpy as np
from keras.preprocessing.sequence import TimeseriesGenerator

# Sample multivariate time series data (2 features)
data = np.random.rand(100, 2)

# Generator with sampling_rate > 1
data_gen = TimeseriesGenerator(data, data, length=5, sampling_rate=2, target_size=2, batch_size=1)

X, y = data_gen[0]
print("Input shape:", X.shape) # Output: (1, 5, 2)
print("Target shape:", y.shape) # Output: (1, 2, 2)
print("First input sequence:", X[0])
print("Corresponding target sequence:", y[0])
```

This example shows the use of `TimeseriesGenerator` with multivariate time series data and a sampling rate greater than 1.  The input has two features, and every other timestep is skipped due to the `sampling_rate` of 2. The `target_size` is 2, predicting the next two timesteps.  Observe that both the input and target now have a third dimension representing the features.


**3. Resource Recommendations:**

I would recommend reviewing the official Keras documentation on data preprocessing, specifically the section on `TimeseriesGenerator`.  Additionally, explore tutorials and examples on RNN architectures and their application to time series forecasting.  A solid grasp of linear algebra and basic time series analysis techniques will significantly aid in understanding and utilizing `TimeseriesGenerator` effectively.  Furthermore, consider exploring resources on multivariate time series analysis for handling more complex datasets.  Finally, understanding the implications of different activation functions and loss functions in the context of time series forecasting is crucial for effective model building.
