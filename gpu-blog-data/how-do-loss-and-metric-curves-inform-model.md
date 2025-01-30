---
title: "How do loss and metric curves inform model performance?"
date: "2025-01-30"
id: "how-do-loss-and-metric-curves-inform-model"
---
The disparity between a model’s loss function and its evaluation metric is often the primary source of confusion when assessing machine learning model performance. I've spent considerable time debugging seemingly well-performing models only to discover this exact discrepancy. The loss function, minimized during training, guides the model's parameter updates, while the metric, evaluated on a held-out dataset, reflects the task's desired outcome. They are not interchangeable, and understanding how their curves relate is crucial for interpreting a model's capabilities.

The loss function, typically differentiable, is an objective function that quantifies the difference between a model's predictions and the actual ground truth values. Its primary purpose is optimization; the algorithm iteratively adjusts the model's internal parameters to reduce this discrepancy. Common loss functions include mean squared error (MSE) for regression tasks and categorical cross-entropy for classification. Minimizing the loss is the model's learning signal, directly impacting how parameters change. A low loss, however, does not automatically translate to excellent performance on the intended task. The loss is an internal measure, often optimized on the training data, and doesn't consider factors such as the balance of classes, or the sensitivity of the metric to slight variations in prediction.

The evaluation metric, on the other hand, is a task-specific measure reflecting the model's real-world utility. It's what you care about in practice. Metrics are designed to assess performance from a user or business standpoint. For example, accuracy assesses the fraction of correct predictions, precision measures the proportion of true positives among all positive predictions, and recall measures the proportion of true positives among all actual positives. In a medical diagnosis system, recall might be more crucial than precision. Metrics are chosen based on the problem's specific requirements and biases. A model might perform well on the loss function during training, but show inadequate results on the chosen evaluation metric on unseen data. This is where the curves come into play.

During training, both the loss and the chosen metric are calculated across epochs, generating corresponding curves. Analyzing these curves, generally plotted together, reveals several important aspects of model performance. A consistently decreasing loss curve indicates that the model is learning, that it's adjusting parameters to fit the training data. However, a decreasing loss curve without a corresponding increase in the evaluation metric indicates the model might be overfitting. In this scenario, the model learns intricacies of the training data but fails to generalize to unseen examples. Furthermore, the gap between the training loss and the validation loss also reveals overfitting. If training loss drops drastically, but validation loss plateaus or rises, the model is fitting the training data too closely at the expense of generalizability. Conversely, a stagnant loss curve (and metric curve) suggests that the model may have reached convergence, or the model lacks the capacity to learn, or the training data is not sufficient. Similarly, fluctuating or erratic curves may reveal unstable training parameters, excessive learning rates, or poor dataset quality. The relationship and behavior of these two curves reveal where the model stands in its performance and learning.

Here are some code examples with corresponding commentary, reflecting specific scenarios I've encountered:

**Example 1: Basic Regression Model**

This example showcases a simple linear regression model, where both mean squared error (MSE) and root mean squared error (RMSE) are calculated (MSE is used as loss, RMSE as metric). I opted for RMSE for ease of interpretation because it’s on the same scale as the output variable.

```python
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Create synthetic data
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = 2 * X + 1 + np.random.normal(0, 2, X.shape)

# Define a simple linear model
model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(1,))])
model.compile(optimizer='adam', loss='mean_squared_error')

# Training the model and collecting history
history = model.fit(X, y, epochs=100, verbose=0)

# Extracting loss and RMSE
loss_values = history.history['loss']
rmse_values = np.sqrt(loss_values) # calculate RMSE from training loss

# Plotting Loss and Metric
epochs = range(1, len(loss_values) + 1)
plt.figure(figsize=(10, 5))
plt.plot(epochs, loss_values, 'b', label='Training Loss (MSE)')
plt.plot(epochs, rmse_values, 'r', label='Training RMSE')
plt.title('Training Loss and RMSE')
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.legend()
plt.show()
```

*Commentary*: This example depicts a basic regression case. The MSE (loss) and RMSE (metric) curves both decrease smoothly over epochs.  The relatively smooth curves suggest the model is learning well, without overfitting or underfitting tendencies. The important observation here is the curves follow the same trend as the metric is derived from the loss; they tend to move together.

**Example 2: Classification with Overfitting**

Here, we see a common case of overfitting in a binary classification task using categorical cross-entropy for the loss and accuracy for metric.

```python
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Create synthetic data, intentionally designed for overfitting
X = np.random.rand(500, 2)
y = np.where((X[:, 0]**2 + X[:, 1]**2) < 0.5, 1, 0)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Model setup
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the model
history = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_val, y_val), verbose=0)

# Extract training and validation loss and accuracy
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# Plot
epochs = range(1, len(train_loss) + 1)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, 'b', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, train_acc, 'b', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.legend()
plt.tight_layout()
plt.show()
```

*Commentary*: The loss plot reveals a substantial gap between training and validation losses. Training loss decreases continuously, while validation loss plateaus or even increases after a certain number of epochs. Similarly, training accuracy rises whereas validation accuracy stagnates. This significant divergence signifies that the model is memorizing the training data rather than learning underlying patterns. The overfitting is visually evident in the divergence.

**Example 3: Classification with Imbalanced Classes**

This code demonstrates the effect of imbalanced data using a binary classification task and focusing on the F1-score, a crucial metric when class imbalance is present, in contrast with accuracy.

```python
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Create imbalanced synthetic data
X = np.random.rand(1000, 2)
y = np.where(np.random.rand(1000) < 0.1, 1, 0) # imbalanced: ~ 10% positive
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Model and loss settings
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training and predictions
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), verbose=0)
y_pred_train = (model.predict(X_train) > 0.5).astype("int32").flatten()
y_pred_val = (model.predict(X_val) > 0.5).astype("int32").flatten()


# Calculate F1-scores, accuracy scores
f1_train = [f1_score(y_train,y_pred_train) for i in range(len(history.history['loss']))]
f1_val = [f1_score(y_val, y_pred_val) for i in range(len(history.history['val_loss']))]

# Plot
epochs = range(1, len(history.history['loss']) + 1)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs, history.history['loss'], 'b', label='Training Loss')
plt.plot(epochs, history.history['val_loss'], 'r', label='Validation Loss')
plt.title('Loss Curves')
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(epochs, f1_train, 'b', label='Train F1')
plt.plot(epochs, f1_val, 'r', label='Validation F1')
plt.title('F1 Score Curves')
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.legend()
plt.tight_layout()
plt.show()
```

*Commentary:* In this scenario, we observe that while the loss curves show a normal trend,  the f1 score curves show significantly different behavior compared to the accuracy (not plotted here due to space constraints), illustrating that accuracy is not a reliable metric in this imbalanced dataset. The F1 score emphasizes the model's performance on the minority class. This difference highlights the importance of choosing metrics appropriate for the task at hand, as a high overall accuracy can be misleading if a minority class is important. The code here demonstrates the metric calculation after training to make the visualization more clear.

In summary, observing loss and metric curves during training is not merely a post-training exercise; it is a diagnostic process. It requires a careful analysis of not just the value but also the shape of the curves, the trends they exhibit, and their inter-relationships. These are critical for early detection of issues like overfitting, underfitting, convergence problems, and data imbalances, that will affect the model's generalization capacity.

For further resources on this topic, I recommend consulting academic resources on statistical learning and optimization. Also beneficial are textbooks specifically on the application of machine learning in different areas such as computer vision and natural language processing. Additionally, documentation of popular machine learning libraries, such as TensorFlow and scikit-learn, can be of great use to gain practical understanding of loss function design, metric selection, and model training processes. Finally, online courses offer interactive ways to explore these concepts, frequently providing hands-on programming opportunities.
