---
title: "Why is accuracy different when using `evaluate()` vs `predict()`?"
date: "2024-12-16"
id: "why-is-accuracy-different-when-using-evaluate-vs-predict"
---

Okay, let's unpack the nuanced difference between accuracy when employing `evaluate()` versus `predict()` in the context of machine learning models – a topic I've seen trip up many newcomers and even some seasoned practitioners over the years. Believe me, I've spent more late nights debugging this than I care to remember. Specifically, I recall a project a few years back where our initial model seemed fantastic based on evaluation metrics, but fell apart when we put it into production, a situation primarily due to this very distinction.

The fundamental reason why accuracy often differs between `evaluate()` and `predict()` lies in the context of their usage and the data they operate on. `evaluate()`, typically used in the model training or validation phase, calculates metrics like accuracy (or precision, recall, f1-score, etc.) on labeled data – meaning, data where we *know* the correct output. Conversely, `predict()` is employed on unlabeled data, the kind you’d see in a real-world application where the 'ground truth' is unknown. While this may seem obvious, the implications are substantial. Let’s explore why that matters, technically.

When you call `evaluate()`, you are generally feeding in your testing or validation dataset. This dataset includes *both* the input features and the corresponding target variables (or labels). The model internally performs a prediction using the input features, compares its prediction against the actual labels, and uses this information to compute an evaluation metric like accuracy. Think of it as a carefully orchestrated test where the answers are readily available to the examiner.

The calculation is straightforward:

`Accuracy = (Number of Correct Predictions) / (Total Number of Predictions)`

However, when you use `predict()`, you're usually dealing with new, unseen data without labels. The model's purpose at this point is to *infer* the target variable. You are no longer comparing the predictions to the ground truth when you call `predict()`. Instead, it’s purely the model's output. Therefore, you can't calculate accuracy with `predict()` directly, at least not without a separate dataset of corresponding labeled examples. So what you have after the predict function is simply predictions of your target variable given the input feature without any accuracy comparison.

Here’s a critical point: the perceived difference in accuracy between the two functions isn't a reflection of the model behaving differently during prediction. Rather, it highlights a *misunderstanding* about what each function is measuring and when to measure.

Let's get into some concrete examples using Python and TensorFlow, where this distinction becomes clearer.

**Example 1: Simple Classification with Labeled Data and Evaluation**

```python
import tensorflow as tf
import numpy as np

# Simulate labeled data for testing
x_test = np.random.rand(100, 10).astype(np.float32)
y_test = np.random.randint(0, 2, 100).astype(np.int32) # Binary classification

# Create a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Evaluate with test data, providing labels, calculating accuracy
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Evaluation Accuracy: {accuracy:.4f}")

# Predict on the same test data. Predictions only, no accuracy calculation here
predictions = model.predict(x_test)
print(f"Predictions from predict() function: {predictions[:5]}") # Show just the first 5 predicted probabilities
```

In this snippet, `evaluate()` is provided both the input features (`x_test`) and their corresponding labels (`y_test`), allowing the accuracy to be computed. In contrast, the `predict()` call on `x_test` gives us probabilities for each class, but without the ground truth, there is no direct way to calculate an accuracy metric. If you were to *incorrectly* attempt to calculate accuracy on the output of a `predict()` call, you are making a mistake by assuming that you have ground truth to compare to, when that’s not the case.

**Example 2: Impact of Data Distribution Shifts**

Another aspect I’ve noticed in practical projects is that even with *seemingly* similar data, subtle differences between training, evaluation, and real-world data can cause discrepancies between the evaluation accuracy and the accuracy perceived after deployment, where you are making predictions.

```python
import tensorflow as tf
import numpy as np

# Simulate training data
x_train = np.random.rand(500, 5).astype(np.float32) * 2  # Scale up data
y_train = np.random.randint(0, 3, 500).astype(np.int32)

# Create test data, with a different distribution
x_test = np.random.rand(200, 5).astype(np.float32)
y_test = np.random.randint(0, 3, 200).astype(np.int32)

# Create out of sample prediction data
x_predict = np.random.rand(200,5).astype(np.float32) * 0.5 # Scale it down

# Create a basic classifier
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, verbose=0)

# Evaluate accuracy on test data
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test set Evaluation Accuracy: {accuracy:.4f}")

# Predict on out-of-sample data without labels for future performance estimate
predictions = model.predict(x_predict)
print(f"Sample Predictions (first 5) for Prediction set:{predictions[:5]}")
```

Here, the distribution of `x_train` and `x_test` are different, and the distribution of `x_predict` is different again. The accuracy is only calculated on the test set data which has the labels. The prediction set gives you probabilities, but not accuracy. In a real world use case, this would be your out-of-sample data for production. If there's a distribution shift, as shown here, the evaluation accuracy may not accurately reflect the model's performance once deployed, where you might see different (likely lower) accuracy when real-world labels become available down the line and can be compared to the predictions. This difference isn't due to `predict()` *itself*, but because real-world data may have very different characteristics than your training set, a concept often called *distribution drift*.

**Example 3: Proper Use of Validation Sets and Monitoring**

Often, the key issue is improper validation setups. In my experience, the *best* way to anticipate a model’s actual performance is by having a well-defined validation set, representative of what you expect in real-world scenarios.

```python
import tensorflow as tf
import numpy as np

# Simulate a larger, more complex dataset
x_all = np.random.rand(1000, 20).astype(np.float32)
y_all = np.random.randint(0, 5, 1000).astype(np.int32) # 5-class classification


# Split the data into training and validation sets
split_index = int(0.8 * len(x_all))
x_train, x_val = x_all[:split_index], x_all[split_index:]
y_train, y_val = y_all[:split_index], y_all[split_index:]

# Out of sample prediction data set
x_predict = np.random.rand(200, 20).astype(np.float32) # Generate new unseen data


# Create a slightly deeper model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(20,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train and validate in each epoch
history = model.fit(x_train, y_train, epochs=10, verbose=0, validation_data=(x_val,y_val))


# Evaluate on validation set (which should be close to real world data if done right)
loss, accuracy = model.evaluate(x_val, y_val, verbose=0)
print(f"Validation set accuracy after training : {accuracy:.4f}")

# Predict on real world data and show first five predictions
predictions = model.predict(x_predict)
print(f"Sample Predictions (first 5) for Prediction set:{predictions[:5]}")


# Access the history object to understand what was happening during training
print(f"Training accuracy history over epochs: {history.history['accuracy']}")
print(f"Validation accuracy history over epochs: {history.history['val_accuracy']}")
```

Here, we use a validation set during training and evaluate our model on it. The validation accuracy, monitored during training, gives a better idea of the model's generalization capability than simply observing the training accuracy, or just testing once on held out test set data. The `predict` call is then used on a different out of sample set to generate predictions.

For further reading to solidify these concepts, I would suggest the book "Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow" by Aurélien Géron, which covers this distinction clearly, and also the paper "Hidden Technical Debt in Machine Learning Systems" by Sculley et al. which discusses the broader challenges of maintaining reliable real-world machine learning systems. Understanding the implications of these concepts is not just important, it's crucial for making robust predictions with machine learning. This difference isn’t merely semantics, it’s a fundamental component of responsible model deployment and evaluation.
