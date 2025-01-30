---
title: "How can I save the best model when custom metrics are unavailable?"
date: "2025-01-30"
id: "how-can-i-save-the-best-model-when"
---
The core challenge in saving the best model when custom metrics are unavailable stems from the reliance on default metrics, often insufficient for nuanced model evaluation.  My experience optimizing recommendation systems for e-commerce platforms highlighted this limitation repeatedly.  While accuracy might appear high using default metrics like precision or recall, the model's performance on specific user segments or product categories could be drastically underwhelming. This necessitates a robust strategy for model selection that transcends the limitations of readily available metrics.

The most effective approach involves crafting a surrogate metric, a quantifiable measure that correlates strongly with the desired, but unobtainable, performance characteristic. This entails a deep understanding of the problem domain and careful consideration of what constitutes "best" in the context of the application.  Simply maximizing default metrics without domain expertise is insufficient and can lead to deployment of models exhibiting unsatisfactory real-world behavior.


**1. Surrogate Metric Development and Implementation:**

The first step is to clearly define the target outcome that the unavailable custom metric would capture.  For instance, in my work with the aforementioned recommendation systems, the unavailable metric was user engagement (measured by time spent browsing items recommended by the model). This is not directly available through standard model evaluation tools. A viable surrogate metric could be the click-through rate (CTR) on recommended items. While not a perfect substitute for engagement time, CTR exhibits a strong positive correlation—higher CTR generally indicates higher engagement.

This surrogate metric needs to be explicitly calculated and incorporated into the model training and evaluation loop.  This necessitates modifications to the training pipeline beyond the standard libraries' default functionalities.  Specifically, a custom evaluation function needs to be defined and utilized to track and record the surrogate metric during the model's training iterations.  Only after this robust integration will the system accurately identify the iteration with the highest surrogate metric value.

**2. Code Examples:**

The following examples illustrate how to implement this strategy within different common machine learning frameworks.

**Example 1: Using TensorFlow/Keras**

```python
import tensorflow as tf
import numpy as np

# ... (Model definition and data loading) ...

def custom_evaluation(y_true, y_pred):
  # Assuming y_pred contains probabilities for each class
  # and y_true are the true labels. This example focuses on CTR
  clicks = tf.reduce_sum(tf.cast(y_true * tf.round(y_pred), tf.float32)) #Clicks on recommended items
  total_recommendations = tf.reduce_sum(tf.cast(tf.round(y_pred), tf.float32)) #Total recommendations
  ctr = clicks / (total_recommendations + tf.keras.backend.epsilon()) #Avoid division by zero
  return ctr

model = tf.keras.models.Sequential(...) #Your model definition

model.compile(optimizer='adam',
              loss='binary_crossentropy', #or appropriate loss
              metrics=['accuracy', custom_evaluation])

#...Training loop with ModelCheckpoint using custom_evaluation as monitor...

filepath = "best_model_by_ctr.h5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='custom_evaluation',
                                               verbose=1, save_best_only=True,
                                               mode='max')

model.fit(X_train, y_train, epochs=10, callbacks=[checkpoint])

```

This example leverages Keras's callback mechanism to save the model weights only when the `custom_evaluation` metric (CTR in this case) improves.  `save_best_only=True` ensures that only the model with the highest CTR is saved. The addition of `tf.keras.backend.epsilon()` prevents division by zero errors.

**Example 2: Using PyTorch**

```python
import torch
import torch.nn as nn
import torch.optim as optim

# ... (Model definition and data loading) ...

def custom_evaluation(y_true, y_pred):
  # Assuming y_pred contains probabilities, similar to TensorFlow example
  clicks = torch.sum((y_true * torch.round(y_pred)).float())
  total_recommendations = torch.sum(torch.round(y_pred).float())
  ctr = clicks / (total_recommendations + 1e-8) #Avoid division by zero
  return ctr

model = YourModelClass() #Your model definition
criterion = nn.BCELoss() #or appropriate loss
optimizer = optim.Adam(model.parameters())

best_ctr = 0
best_model_state = None

for epoch in range(10):
  # ... (Training loop) ...
  with torch.no_grad():
      y_pred = model(X_val) #Validation set predictions
      current_ctr = custom_evaluation(y_val, y_pred).item()
      if current_ctr > best_ctr:
          best_ctr = current_ctr
          best_model_state = model.state_dict()

torch.save(best_model_state, 'best_model_by_ctr.pth')
```

This PyTorch example iteratively evaluates the model on a validation set after each epoch and saves the model's state dictionary only if the surrogate metric (CTR) improves.  This avoids the overhead of saving the entire model after every epoch.

**Example 3: Using scikit-learn with a custom scorer**

```python
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
import numpy as np

#... (Model definition and data loading) ...

def ctr_scorer(y_true, y_pred):
  clicks = np.sum(y_true * np.round(y_pred))
  total_recommendations = np.sum(np.round(y_pred))
  return clicks / (total_recommendations + 1e-8) if total_recommendations > 0 else 0


ctr_scoring = make_scorer(ctr_scorer, greater_is_better=True)

model = YourModelClass() #Your model definition

param_grid = { ... } # hyperparameter grid

grid_search = GridSearchCV(model, param_grid, scoring=ctr_scoring, cv=5)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
#best_model is now the model that maximizes the custom CTR metric

import joblib
joblib.dump(best_model, 'best_model_by_ctr.pkl')
```

This illustrates how to incorporate a custom scorer into scikit-learn's `GridSearchCV`. This facilitates hyperparameter tuning based on the surrogate metric, directly selecting the best model according to the optimized CTR.


**3. Resource Recommendations:**

For a deeper understanding of model evaluation and selection, I recommend exploring texts on advanced statistical modeling, focusing on chapters dedicated to model selection criteria and cross-validation techniques.  Furthermore, in-depth exploration of the chosen machine learning framework's documentation will prove invaluable.  Finally, studying case studies of model deployment and performance optimization in relevant application domains will provide critical context and best practices.  Thorough review of literature on model bias and fairness is crucial when developing surrogate metrics, ensuring the chosen metric accurately reflects the desired outcome without introducing unintended biases.
