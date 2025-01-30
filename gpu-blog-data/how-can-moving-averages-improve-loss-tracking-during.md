---
title: "How can moving averages improve loss tracking during Keras training?"
date: "2025-01-30"
id: "how-can-moving-averages-improve-loss-tracking-during"
---
The inherent volatility of loss values during neural network training, particularly in the early stages or with small batch sizes, can make it difficult to discern genuine trends in performance. Specifically, noisy loss curves can obscure whether a model is converging effectively, leading to suboptimal hyperparameter tuning or premature stopping.

I've consistently observed that raw loss values, especially those calculated on a per-batch basis, jump erratically. These fluctuations, while containing underlying information, are difficult to interpret at a glance. The moving average (MA) offers a smoothing mechanism, providing a more stable and interpretable representation of training progress. By averaging loss values over a defined window, the method suppresses high-frequency noise, allowing for a clearer visualization of the overall loss trajectory. This improved clarity directly impacts my ability to make informed decisions about learning rate adjustments, batch size modifications, and even architecture refinements.

The core idea behind implementing a moving average for loss tracking is to maintain a running average of recent loss values. Each new loss value is incorporated, and the oldest value within the averaging window is discarded, thereby updating the average in a continuous manner. The size of this window, often called the 'window size' or 'span,' determines the degree of smoothing applied. A smaller window will track changes more rapidly but will be more susceptible to noise, whereas a larger window will produce a smoother line but may lag behind actual performance changes. Selecting an appropriate window size is crucial and depends on the specific problem and the variability of the loss.

The moving average can be implemented in several ways. One common method is the Simple Moving Average (SMA), where each loss value within the window is weighted equally. Another is the Exponential Moving Average (EMA), which gives more weight to recent observations. EMA tends to adapt more quickly to new trends than SMA but is sensitive to the choice of the decay factor. I often prefer EMA due to its responsiveness, but SMA works sufficiently well in many scenarios. The following examples explore how these implementations can be integrated into a Keras training loop.

**Example 1: Simple Moving Average**

```python
import tensorflow as tf
import numpy as np

class SimpleMovingAverage:
    def __init__(self, window_size):
        self.window_size = window_size
        self.values = []
        self.average = 0

    def update(self, value):
        self.values.append(value)
        if len(self.values) > self.window_size:
            self.values.pop(0)
        self.average = np.mean(self.values)

    def get_average(self):
        return self.average

# Example usage within Keras training
model = tf.keras.models.Sequential([tf.keras.layers.Dense(1)])
optimizer = tf.keras.optimizers.Adam()
loss_fn = tf.keras.losses.MeanSquaredError()

# Dummy data
X = tf.random.normal((100, 1))
y = tf.random.normal((100, 1))

# Initialize SMA for loss
loss_ma = SimpleMovingAverage(window_size=20)

for epoch in range(10):
    for i in range(0, len(X), 32):
        batch_x = X[i:i+32]
        batch_y = y[i:i+32]

        with tf.GradientTape() as tape:
            y_pred = model(batch_x)
            loss = loss_fn(batch_y, y_pred)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        loss_ma.update(loss.numpy())
        print(f"Epoch: {epoch}, Batch: {i//32}, Loss: {loss.numpy():.4f}, MA Loss: {loss_ma.get_average():.4f}")

```

This first example provides a clear instantiation of the SMA. The `SimpleMovingAverage` class maintains a list of recent loss values. The `update` method adds a new loss value to the list, removes the oldest value if the list exceeds the window size, and calculates the average. In the training loop, we create an instance of this class, update it with each batch loss, and print both the raw loss and the smoothed version. The output shows how the smoothed loss fluctuates less than the raw loss, revealing the overall trend more clearly.

**Example 2: Exponential Moving Average**

```python
class ExponentialMovingAverage:
    def __init__(self, decay=0.9):
        self.decay = decay
        self.average = None

    def update(self, value):
        if self.average is None:
          self.average = value
        else:
            self.average = self.decay * self.average + (1 - self.decay) * value

    def get_average(self):
        return self.average


# Example usage within Keras training
model = tf.keras.models.Sequential([tf.keras.layers.Dense(1)])
optimizer = tf.keras.optimizers.Adam()
loss_fn = tf.keras.losses.MeanSquaredError()

# Dummy data
X = tf.random.normal((100, 1))
y = tf.random.normal((100, 1))

# Initialize EMA for loss
loss_ema = ExponentialMovingAverage(decay=0.9)

for epoch in range(10):
    for i in range(0, len(X), 32):
        batch_x = X[i:i+32]
        batch_y = y[i:i+32]

        with tf.GradientTape() as tape:
            y_pred = model(batch_x)
            loss = loss_fn(batch_y, y_pred)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        loss_ema.update(loss.numpy())
        print(f"Epoch: {epoch}, Batch: {i//32}, Loss: {loss.numpy():.4f}, EMA Loss: {loss_ema.get_average():.4f}")
```

The second example implements an EMA. The `ExponentialMovingAverage` class stores a decay factor, controlling the weighting of new values relative to old averages. The `update` method calculates the EMA value, incorporating the new batch loss with the previously stored average. As in the SMA example, the smoothed loss presents a less volatile picture of the training progress. Here, the EMA, with the chosen decay factor, should react more quickly than SMA to recent changes in the loss function. This responsiveness can be a benefit during specific training stages.

**Example 3: Implementing EMA with a Lambda Callback**

```python
import tensorflow as tf
import numpy as np

class ExponentialMovingAverageCallback(tf.keras.callbacks.Callback):
    def __init__(self, decay=0.9):
        super().__init__()
        self.decay = decay
        self.loss_ema = None

    def on_batch_end(self, batch, logs=None):
        if logs is not None and 'loss' in logs:
            loss = logs['loss']
            if self.loss_ema is None:
              self.loss_ema = loss
            else:
                self.loss_ema = self.decay * self.loss_ema + (1 - self.decay) * loss
            logs['loss_ema'] = self.loss_ema

# Example usage within Keras training
model = tf.keras.models.Sequential([tf.keras.layers.Dense(1)])
optimizer = tf.keras.optimizers.Adam()
loss_fn = tf.keras.losses.MeanSquaredError()

# Dummy data
X = tf.random.normal((100, 1))
y = tf.random.normal((100, 1))

# Initialize EMA callback
ema_callback = ExponentialMovingAverageCallback(decay=0.9)

model.compile(optimizer=optimizer, loss=loss_fn)

model.fit(X, y, epochs=10, batch_size=32, callbacks=[ema_callback], verbose = 1)


```

This final example encapsulates the EMA logic inside a Keras Callback. A callback allows for more structured integration during model training. The `ExponentialMovingAverageCallback` class computes and stores the EMA within the training logs which can later be accessed in a more organized and reproducible manner. This approach avoids manual integration of the MA calculation inside the training loop and makes the MA available to Tensorboard.  The `on_batch_end` method computes the EMA of loss and adds it to the dictionary of logs, named `loss_ema`. Consequently, the fit operation generates both `loss` and `loss_ema`.

From my practical experiences, I've identified some beneficial resources for further study. Books on time series analysis and forecasting offer foundational knowledge about moving averages, providing a deeper theoretical understanding. Additionally, exploring articles that investigate the statistical properties of moving averages will enhance comprehension of their behavior and the impact of different window sizes or decay rates. Finally, code repositories focused on model monitoring and evaluation within TensorFlow or PyTorch typically contain relevant implementations, demonstrating the practical application of moving averages in a broader context. Understanding how loss is tracked via MA, allows for better parameter tuning and more insightful convergence metrics.
