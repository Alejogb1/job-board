---
title: "What input formats are supported by KerasRegressor from scikeras?"
date: "2025-01-30"
id: "what-input-formats-are-supported-by-kerasregressor-from"
---
The core limitation of `scikeras.KerasRegressor` regarding input formats stems from its reliance on the underlying Keras model's input shape expectations.  It doesn't intrinsically support a wider range of input formats than what a standard Keras model can handle.  My experience building and deploying various regression models using `scikeras` in production environments has highlighted this crucial dependency.  Therefore, understanding the input expectations of your Keras model is paramount before considering `scikeras` integration.

**1.  Clear Explanation of Input Format Handling:**

`scikeras.KerasRegressor` acts as a wrapper, allowing seamless integration of Keras models within the scikit-learn ecosystem.  This means the input format compatibility is determined solely by the input layer definition of your custom Keras model.  The `scikeras` wrapper doesn't perform any pre-processing or transformation of the input data beyond what you explicitly define within your Keras model architecture.  This is a key distinction; `scikeras` doesn't introduce any new input handling mechanisms.

The most common input format supported, implicitly, is a NumPy array.  This aligns directly with the fundamental data structure Keras models expect.  Specifically, the shape of this array must precisely match the `input_shape` parameter declared when defining your model's input layer.  For example, if your input layer is defined as `Input(shape=(10,))`, then the input data should be a 2D NumPy array with shape (n_samples, 10), where `n_samples` represents the number of data points in your dataset.  Similarly, for images, the input array would need to have dimensions conforming to the expected height, width, and channels (e.g., (n_samples, height, width, channels)).

Beyond NumPy arrays, you can technically use Pandas DataFrames as input, but this requires careful consideration.  `scikeras` doesn't handle DataFrame conversion automatically. You must explicitly extract the relevant numerical features from your DataFrame and convert them into a NumPy array before passing them to the `fit` method.  Attempting to directly feed a DataFrame, without proper preprocessing to extract and shape the numerical features correctly, will lead to a `ValueError` indicating shape mismatch.  In my experience, this is a common source of errors for newcomers to `scikeras`.

Another important nuance is categorical features.  These must be pre-processed externally, typically using one-hot encoding or other suitable techniques, before being fed into the model.  The Keras model itself, through layers like `Embedding` or by the design of your input layer, will interpret these appropriately. `scikeras` provides no inherent categorical data handling; it relies on the preprocessing steps completed *before* feeding the data into the model.


**2. Code Examples with Commentary:**

**Example 1:  Simple Linear Regression with NumPy Array Input:**

```python
import numpy as np
from tensorflow import keras
from scikeras.wrappers import KerasRegressor

def create_model():
    model = keras.Sequential([
        keras.layers.Dense(1, input_shape=(10,), activation='linear')
    ])
    model.compile(loss='mse', optimizer='adam')
    return model

model = KerasRegressor(model=create_model, epochs=100, verbose=0)
X = np.random.rand(100, 10)
y = np.random.rand(100, 1)
model.fit(X, y)
```

This example demonstrates a basic linear regression model.  The input `X` is a NumPy array of shape (100, 10), matching the `input_shape` defined in `create_model`. The output `y` is a (100, 1) array.  This is the most straightforward and commonly used input format.

**Example 2:  Image Regression with NumPy Array Input:**

```python
import numpy as np
from tensorflow import keras
from scikeras.wrappers import KerasRegressor

def create_model():
  model = keras.Sequential([
      keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
      keras.layers.MaxPooling2D((2, 2)),
      keras.layers.Flatten(),
      keras.layers.Dense(1)
  ])
  model.compile(loss='mse', optimizer='adam')
  return model

model = KerasRegressor(model=create_model, epochs=10, verbose=0)
X = np.random.rand(100, 28, 28, 1) # 100 images, 28x28 pixels, 1 channel
y = np.random.rand(100, 1)
model.fit(X, y)
```

This example shows image regression.  The input `X` is a 4D NumPy array representing 100 grayscale images of size 28x28 pixels. The `input_shape` in the Keras model matches this format.


**Example 3:  Pre-processed DataFrame Input:**

```python
import numpy as np
import pandas as pd
from tensorflow import keras
from scikeras.wrappers import KerasRegressor

def create_model():
    model = keras.Sequential([
        keras.layers.Dense(10, input_shape=(5,), activation='relu'),
        keras.layers.Dense(1, activation='linear')
    ])
    model.compile(loss='mse', optimizer='adam')
    return model

df = pd.DataFrame({'feature1': np.random.rand(100),
                   'feature2': np.random.rand(100),
                   'feature3': np.random.rand(100),
                   'feature4': np.random.rand(100),
                   'feature5': np.random.rand(100),
                   'target': np.random.rand(100)})

X = df[['feature1', 'feature2', 'feature3', 'feature4', 'feature5']].values
y = df['target'].values.reshape(-1, 1)  # Reshape for single output

model = KerasRegressor(model=create_model, epochs=100, verbose=0)
model.fit(X, y)
```

Here, we use a Pandas DataFrame.  Crucially, we extract the features into a NumPy array `X` and reshape the target variable `y` before fitting the model.  Directly feeding the DataFrame would fail due to incompatibility.


**3. Resource Recommendations:**

The official scikit-learn documentation, the Keras documentation, and a comprehensive textbook on machine learning with Python are invaluable resources for deepening your understanding of these libraries and their interactions.  Focusing on chapters dedicated to neural networks and model building within the machine learning textbook will provide a strong foundational context for using `scikeras` effectively.  Pay close attention to sections on data preprocessing and model input/output considerations.  Consulting the Keras documentation for detailed information on layer types and input shape definitions is also essential for successful model creation and integration with `scikeras`.  Finally, thoroughly examining the scikit-learn documentation to understand its estimators and model evaluation techniques will streamline the process of using `scikeras` within a scikit-learn workflow.
