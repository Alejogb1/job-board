---
title: "Why is mean absolute percentage error so high during simple regression training with tf.keras?"
date: "2025-01-30"
id: "why-is-mean-absolute-percentage-error-so-high"
---
High mean absolute percentage error (MAPE) during simple regression training with `tf.keras` frequently stems from the presence of outliers or near-zero values in the target variable.  My experience debugging similar issues across numerous projects, particularly those involving financial time series and sensor data, reveals this as a predominant factor.  The MAPE calculation, being highly sensitive to these data points, magnifies their influence, leading to inflated error metrics even when the model shows reasonable performance on the majority of the data.

Let's clarify the underlying mechanism. The MAPE is calculated as the average of the absolute percentage differences between predicted and actual values.  Formally:

MAPE = (1/n) * Σ<sub>i=1 to n</sub> |(y<sub>i</sub> - ŷ<sub>i</sub>) / y<sub>i</sub>| * 100

Where:

* `n` is the number of data points.
* `y<sub>i</sub>` is the actual value.
* `ŷ<sub>i</sub>` is the predicted value.

Notice the denominator: `y<sub>i</sub>`.  When `y<sub>i</sub>` is close to zero or zero itself, the fraction becomes arbitrarily large or undefined, respectively, dramatically increasing the MAPE. Similarly, outliers, significantly deviating from the majority of data points, contribute disproportionately to the overall error.  A single extreme outlier can inflate the MAPE to an unacceptable level, obscuring the model's performance on the bulk of the data.

Addressing this requires a multi-pronged approach involving data preprocessing, model selection and evaluation metric adjustments.


**1. Data Preprocessing:**

The most direct solution is to handle the problematic data points. This can involve several techniques:

* **Outlier Removal:** Identifying and removing or replacing outliers using methods such as the Interquartile Range (IQR) method or Z-score based outlier detection.  Extreme values significantly above or below the median can be removed, replaced by the median, or capped at a pre-defined percentile.  Caution must be exercised here to avoid data leakage and ensure the chosen method aligns with the domain expertise.

* **Log Transformation:** Applying a logarithmic transformation to the target variable (y<sub>i</sub>) can often mitigate the influence of outliers and near-zero values. This compresses the scale of the data, reducing the impact of extreme values on the MAPE.  Remember to back-transform predictions for meaningful interpretation.

* **Winsorizing:** This technique replaces extreme values with less extreme values – typically the values at specific percentiles.  For instance, you might replace values below the 5th percentile and above the 95th percentile with the values at those percentiles themselves.

**2. Model Selection and Regularization:**

While data preprocessing is critical, model choice also plays a significant role.

* **Robust Regression Models:**  Consider using robust regression models that are less sensitive to outliers.  These techniques, often based on minimizing the influence of outliers during parameter estimation, can produce more stable results in the presence of contaminated data.  Examples include Huber regression or quantile regression.  These are not directly available within the `tf.keras` API but can be implemented using other libraries and integrated within a custom training loop.

* **Regularization Techniques:** Incorporating regularization techniques like L1 or L2 regularization within your `tf.keras` model can help prevent overfitting and reduce the model's sensitivity to outliers.  This is achieved by adding penalty terms to the loss function, discouraging the model from fitting the noise in the data, including outliers.


**3. Alternative Evaluation Metrics:**

Finally, reconsider relying solely on MAPE.  Its sensitivity to outliers limits its usefulness in these situations.  Explore alternatives:

* **Root Mean Squared Error (RMSE):** Less sensitive to outliers than MAPE, providing a more stable measure of overall model performance.

* **Mean Absolute Error (MAE):**  Another robust alternative that focuses on the average absolute difference between predicted and actual values, ignoring the percentage aspect which causes issues with near-zero values.


**Code Examples:**

**Example 1: Data Preprocessing with Log Transformation**

```python
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Sample data (replace with your data)
X = np.random.rand(100, 1)
y = np.exp(np.random.rand(100)) #Introducing near-zero values

# Log transformation
y_log = np.log(y + 1e-9) #Adding a small constant to avoid log(0)

#Scaling for better model convergence
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_x.fit_transform(X)
y_log_scaled = scaler_y.fit_transform(y_log.reshape(-1,1))


# Build and train your keras model (example with a simple dense layer)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_scaled, y_log_scaled, epochs=100, verbose=0)

#Prediction and backtransformation
y_pred_scaled = model.predict(X_scaled)
y_pred = np.exp(scaler_y.inverse_transform(y_pred_scaled)) - 1e-9

#Evaluate using RMSE or MAE
```

This example demonstrates how to apply a log transformation to handle near-zero values.  The addition of `1e-9` prevents errors when taking the logarithm of zero.  Note the use of `MinMaxScaler` to ensure stable model training.


**Example 2: Outlier Removal with IQR**

```python
import numpy as np
import tensorflow as tf
from scipy.stats import iqr

# Sample data (replace with your data)
X = np.random.rand(100, 1)
y = np.concatenate((np.random.rand(90), np.random.rand(10)*100)) #Introduce outliers


# IQR method for outlier detection and removal

Q1 = np.percentile(y, 25)
Q3 = np.percentile(y, 75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

y_filtered = y[(y >= lower_bound) & (y <= upper_bound)]
X_filtered = X[(y >= lower_bound) & (y <= upper_bound)]

#Build and train your keras model
# ... (same as Example 1, but using X_filtered and y_filtered)

```
This snippet shows how to identify and remove outliers using the Interquartile Range method.  Data points outside the defined bounds are eliminated before model training.


**Example 3: Using RMSE as an Evaluation Metric**

```python
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error

# ... (Model building and training as in previous examples) ...

y_pred = model.predict(X) # or X_scaled, X_filtered depending on your preprocessing

rmse = np.sqrt(mean_squared_error(y, y_pred)) #or appropriate y variable

print(f"RMSE: {rmse}")
```

This example showcases how to calculate RMSE.  Replace `y` and `y_pred` with the appropriately processed data.


**Resource Recommendations:**

*  Books on statistical modeling and machine learning.
*  Documentation for TensorFlow/Keras.
*  Textbooks focusing on time series analysis (if applicable).
*  Research papers on robust regression techniques.
*  Articles dedicated to handling outliers in regression analysis.


By systematically addressing data issues, carefully selecting models, and employing appropriate evaluation metrics, you can significantly improve the accuracy and interpretability of your simple regression models trained with `tf.keras`, even in the presence of challenging data. Remember that the best approach will depend on the specific characteristics of your data and the goals of your analysis.
