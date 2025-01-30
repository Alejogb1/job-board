---
title: "What is the default category assignment during classification model training?"
date: "2025-01-30"
id: "what-is-the-default-category-assignment-during-classification"
---
In my experience developing machine learning models for a variety of classification tasks, I've observed that a 'default' category assignment, as you've phrased it, is not a universal or implicit property inherent in the training process itself. Rather, the concept of a 'default' category emerges primarily during the application of a trained model to new, unseen data, and it stems from specific choices made in model implementation, not training. During training, labels are explicitly provided for each training instance. The model learns to map features to these pre-defined categories. There's no implicit assumption of what to do if the model encounters data it hasn't seen during training; this decision comes *after* the training phase, when making predictions on unseen data. If no specific mechanism is implemented, such as setting a threshold or using a probability based rule, then the class with highest prediction score is chosen. This becomes, by effect, the 'default' because that will always be the output, given any input, if you take the argmax.

The crux of the matter lies in the fact that classification models, particularly probabilistic ones like logistic regression or neural networks, output a probability distribution over all possible classes for a given input. These probabilities represent the model's belief that the instance belongs to each class. The 'default' behavior occurs when a decision is made based on these probabilities. Most commonly, the class with the highest predicted probability is chosen as the prediction. Consequently, if no explicit logic for rejecting or categorizing ambiguous cases is put in place, this maximum probability class always becomes the default. It's not a category the model 'learns' to associate by default, but rather a consequence of the argmax or the decision rule. The model itself has no notion of 'default' during the learning process. The choice happens when the model attempts to classify and needs to finalize a prediction.

Consider a binary classification problem. I worked on a project classifying emails as either "spam" or "not spam." The training data consisted of emails labeled accordingly. The model, a simple logistic regression, learned a decision boundary in the feature space that separated, to the best of its ability, spam from not-spam emails. When applying this trained model to a new email, the model outputs the probability of being "spam." The other probability is computed as the probability of "not spam," which is one minus the spam probability. If we use a decision rule that selects the class which probability greater than 0.5, then the behavior will be as we intended. If we take the argmax, then we choose the class with highest probability. In the absence of any specific rule, if the model predicts that the email has a 0.49 chance of being spam, this then becomes not spam. If the model has been poorly trained or is confused, then the prediction may not be reliable. However, with no other guidance, that outcome becomes the prediction of the model and therefore, in a sense, the 'default.' A similar scenario unfolds for multi-class problems: the class with the highest probability gets selected when you use a argmax decision rule.

This 'default' isn't a category learned *during training*; it's a consequence of how we choose to translate the model's probabilistic output into a final class assignment. This is an important difference. In order to create more reliable predictions, we might take action during deployment and inference to mitigate undesirable 'default' outcomes.

Here are some code examples using Python and common machine learning libraries to demonstrate these concepts. I will present three examples.

**Example 1: Basic Logistic Regression**

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Sample data (replace with real data)
X = np.array([[1, 2], [2, 3], [3, 4], [1, 1], [2, 2], [3, 3]])
y = np.array([0, 0, 0, 1, 1, 1]) # 0: 'Not Spam', 1: 'Spam'

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions (using the implicit 'default' of highest probability)
predictions = model.predict(X_test)

# Calculate accuracy of predictions.
accuracy = accuracy_score(y_test, predictions)
print(f"Test set accuracy is {accuracy}")
print(f"Test predictions are {predictions}")


# Predict using probability
probabilities = model.predict_proba(X_test)

print(f"Predicted probabilities are {probabilities}")
```

In this example, I train a simple logistic regression model. The `model.predict()` function implicitly applies the "argmax" decision rule, selecting the class with the highest probability as the prediction. The `model.predict_proba()` provides the probabilities for the various classes, which is useful for implementing a prediction rule more complex than the argmax, if desired. There is no 'default' category during the training itself; the `model.fit()` is only learning parameters that allow to separate the classes and assign the correct labels when predicting.

**Example 2: Implementing a Threshold**

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Sample data (replace with real data)
X = np.array([[1, 2], [2, 3], [3, 4], [1, 1], [2, 2], [3, 3]])
y = np.array([0, 0, 0, 1, 1, 1]) # 0: 'Not Spam', 1: 'Spam'

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Get class probabilities
probabilities = model.predict_proba(X_test)

# Implement a threshold based rule
threshold = 0.7
predictions = []

for probs in probabilities:
    if probs[1] > threshold:
        predictions.append(1)
    elif probs[0] > threshold:
         predictions.append(0)
    else:
        predictions.append(-1)

print(f"Predictions: {predictions}")
# Calculate accuracy of predictions
predictions = np.array(predictions)
y_test = np.array(y_test)
mask = predictions != -1
if mask.sum() != 0:
    accuracy = accuracy_score(y_test[mask], predictions[mask])
    print(f"Test set accuracy is {accuracy}")
```

Here, I illustrate how to move beyond the implicit 'default' behavior and choose predictions based on a decision rule with a threshold of 0.7. If neither probability exceeds the threshold, a value of `-1` is appended which is not a valid category; therefore I have implemented an outlier class, which prevents the model from making a prediction it isn't comfortable with. This provides a prediction more informative than simply assigning it to the most likely outcome. The accuracy is calculated based only on the samples where we made a prediction.

**Example 3: Multi-class Classification with a Custom "Unknown" Class**

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Sample multi-class data (replace with real data)
X = np.array([[1, 2], [2, 3], [3, 4], [1, 1], [2, 2], [3, 3], [4,5], [5,6]])
y = np.array([0, 0, 1, 1, 2, 2, 1, 0])  # Categories 0, 1, and 2

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a multi-class logistic regression model
model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
model.fit(X_train, y_train)

# Get class probabilities
probabilities = model.predict_proba(X_test)

# Add an 'unknown' category if max probability is below a threshold.
threshold = 0.5
predictions = []

for probs in probabilities:
    if max(probs) < threshold:
        predictions.append(-1) # Unknown class is -1
    else:
        predictions.append(np.argmax(probs))


print(f"Predictions: {predictions}")
predictions = np.array(predictions)
y_test = np.array(y_test)
mask = predictions != -1
if mask.sum() != 0:
    accuracy = accuracy_score(y_test[mask], predictions[mask])
    print(f"Test set accuracy is {accuracy}")
```

Here, I extended the concept to a multi-class problem, where labels are 0, 1, or 2. I implement a similar rule to that in Example 2, assigning a new label -1 if the maximal predicted probability is below a threshold, representing an 'unknown' category. Again, the accuracy is computed with the samples with labels not -1.

For further reading, I recommend focusing on resources that cover topics such as model evaluation metrics, decision rules, and calibration methods for probability outputs. Books covering general machine learning or specifically classification algorithms often contain details about how different methods handle probabilities and how they select a final prediction. I would also suggest researching implementations and tutorials from the popular packages I have used.
