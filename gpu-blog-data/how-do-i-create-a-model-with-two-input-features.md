---
title: "How do I create a model with two input features?"
date: "2025-01-26"
id: "how-do-i-create-a-model-with-two-input-features"
---

When constructing a machine learning model, handling two input features requires careful consideration of the model's architecture and the nature of the features themselves.  My experience in developing predictive models for resource allocation has shown that even seemingly simple two-feature models can benefit from various approaches, depending on the underlying data patterns and the problem's complexity. It's not a monolithic process; the "how" is highly dependent on the target outcome.

**Clear Explanation:**

At its core, a two-input feature model is about mapping a two-dimensional input space to an output space, whether that output is a continuous value (regression) or a discrete category (classification). These inputs, often represented as columns in a dataset, can be of various types: numeric, categorical, ordinal, or even transformed versions of other features. The choice of model is heavily influenced by the characteristics of these features.

Consider, for example, a model predicting website loading times using two input features: server load (numeric) and time of day (categorical, perhaps bucketed into morning, afternoon, and evening). Here, one might opt for a linear model if the relationship between server load and loading time is approximately linear and if the time of day effect can be captured through categorical encoding. However, if interactions between time of day and server load are crucial—perhaps servers are more sensitive to high load during peak hours—a model capable of capturing these interactions would be more appropriate. A simple linear model would struggle with non-linear or interaction effects.

In practice, working with two input features involves several key steps: data preprocessing, model selection, training, evaluation, and potentially, tuning. Preprocessing often entails scaling numeric features to a similar range to prevent features with larger magnitudes from dominating the model's training. If one or both features are categorical, encoding them into numerical representations is required. Common encoding techniques include one-hot encoding, ordinal encoding (for ranked categories), and embeddings (for high-cardinality categorical features). One-hot encoding converts each category into a separate feature, creating a potentially sparse input vector. Ordinal encoding converts categories into a numerical sequence, implying a specific order between them. Embeddings learn a dense, low-dimensional vector representation for each category, capturing more complex semantic relationships. Feature selection, while less critical with only two inputs, should still be considered when assessing whether each feature contributes meaningfully to model performance.

Model selection hinges on the nature of the relationship between input features and output. Linear regression is suitable if the relationship is approximately linear and if no complex interactions are expected. For classification tasks, logistic regression could be used. More complex models, such as decision trees, random forests, gradient boosting machines, and neural networks, become pertinent when the underlying relationship is non-linear or involves interactions. These models offer increased flexibility but also increased complexity and risk of overfitting. Finally, model tuning, involving the selection of appropriate hyperparameters, is important in maximizing a model's predictive capability on unseen data. This often requires careful consideration of validation error to avoid overfitting. Cross-validation techniques are generally employed to assess generalization performance.

**Code Examples with Commentary:**

The following examples demonstrate the creation of models using Python and libraries like scikit-learn and numpy.

**Example 1: Linear Regression**

This example illustrates how to perform a simple linear regression with two numeric input features.

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Sample data (replace with your actual data)
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
y = np.array([3, 5, 7, 9, 11]) # Example target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
predictions = model.predict(X_test)

print("Predictions:", predictions)
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
```

*   **Explanation:**  This snippet generates a simple dataset with two features (X) and a corresponding target (y). We then split the data for proper training and evaluation. A LinearRegression model is initialized and fit to the training data using `model.fit`. The trained model can then be used to predict targets from new unseen data with `model.predict`. The model's learned coefficients for each feature and the intercept term are also outputted for insight into the linear relationship. Note that the example utilizes a small and synthetic dataset; in practice, larger and real datasets are generally needed.

**Example 2: Logistic Regression with One-Hot Encoding**

This example shows how to use logistic regression with one numeric feature and one categorical feature. The categorical feature is one-hot encoded.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Sample Data
X = np.array([[25, 'A'], [30, 'B'], [35, 'A'], [40, 'C'], [45, 'B']])
y = np.array([0, 1, 0, 1, 0]) # Binary target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(), [1]), # onehot encode index 1 (categorical feature)
        ('passthrough', 'passthrough', [0]) # pass index 0 through as is (numeric feature)
    ])


# Apply preprocessing to train and test data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Create logistic regression model
model = LogisticRegression()

# Train the model
model.fit(X_train_processed, y_train)

# Make predictions
predictions = model.predict(X_test_processed)

print("Predictions:", predictions)
```

*   **Explanation:** This example shows the utility of `ColumnTransformer` and `OneHotEncoder`. Before fitting the model, the data is preprocessed. The ColumnTransformer applies the OneHotEncoder to the second column of the data (the categorical feature, indexed by 1), and uses passthrough on the first column (the numeric feature, indexed by 0). It transforms the data into suitable input format for Logistic Regression. The process ensures that the model receives proper input with numerical representations of both features. The logistic regression model can then predict the binary target. This illustrates how to handle mixed feature types in scikit-learn.

**Example 3: Simple Neural Network**

This final example demonstrates a basic neural network using `tensorflow` that is structured to handle two input features.

```python
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Sample data
X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0]])
y = np.array([[3.0], [5.0], [7.0], [9.0], [11.0]])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(10, activation='relu', input_shape=(2,)),  # Input layer with 2 features
  tf.keras.layers.Dense(1)  # Output layer (regression)
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=100, verbose = 0)

# Make predictions
predictions = model.predict(X_test)
print("Predictions:", predictions)
```

*   **Explanation:** Here, a simple neural network is defined with `tensorflow`. This model accepts 2 inputs (indicated with the `input_shape=(2,)` parameter) which corresponds to the two input features. The hidden layer uses a 'relu' activation. The final layer contains only 1 output neuron, suitable for regression. Note that both features are treated as numeric; if there are categorical inputs, appropriate embeddings or one-hot encodings would be needed to handle them as numeric data. The model is trained with sample input and targets using the Adam optimizer and mean squared error (MSE) as a loss function. The model is then tested and predictions printed.

**Resource Recommendations:**

For a deeper understanding of model development, the following books and online resources are useful:

*   **"Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow"** by Aurélien Géron offers a practical guide to machine learning concepts and their implementation in Python.
*   **"Python Data Science Handbook"** by Jake VanderPlas provides comprehensive coverage of data manipulation, analysis, and machine learning libraries in Python.
*   **The scikit-learn documentation** is an invaluable resource for learning the intricacies of various machine learning algorithms and preprocessing techniques.
*   **TensorFlow official documentation and tutorials** offer in-depth guides to building and deploying deep learning models.
*  **Numerous online courses on platforms such as Coursera and edX** cover a wide range of machine learning topics, including feature engineering and model selection, in a structured learning environment.
