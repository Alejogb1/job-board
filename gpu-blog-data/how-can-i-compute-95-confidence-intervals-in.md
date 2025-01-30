---
title: "How can I compute 95% confidence intervals in PyTorch for classification and regression tasks?"
date: "2025-01-30"
id: "how-can-i-compute-95-confidence-intervals-in"
---
Confidence intervals quantify the uncertainty around a model’s predictions, and are particularly useful when evaluating model performance on unseen data. While PyTorch itself does not directly provide functions to compute these intervals, methods from statistical inference can be readily integrated to achieve this. I've routinely incorporated this approach, and found that the key lies in leveraging bootstrapping or Bayesian approaches based on the specific task, either classification or regression.

**Classification: Confidence Intervals Using Bootstrapping**

For classification tasks, the fundamental issue is that predictions are discrete (class labels). A single predicted label lacks an inherent notion of variability needed to construct a confidence interval. We can overcome this by employing bootstrapping, which involves repeatedly sampling, *with replacement*, from the test set and calculating predictions on these resampled datasets. We can then examine the distribution of the probabilities for each class, derived from these bootstrapped predictions.

In practice, for a classification task, I would typically proceed as follows: First, I generate predictions on the full test set. For each data point in the test set, I collect the class probabilities. These probabilities are based on the softmax output of the final layer. Bootstrapping starts here. I repeatedly (e.g., 1000 times) resample the test set, with replacement, to create bootstrap samples of equal size. I run inference on each sample, and for *each* test data point, I get a different set of class probabilities. This process gives us a distribution of probabilities for *each* test data point and *each* class. Now, for each test data point and each class, I would compute the lower and upper bound of the 95% confidence interval by finding the 2.5th and 97.5th percentiles of the probability distributions generated via bootstrapping. These bounds are then used to define a probabilistic interval estimate of a model's classification confidence.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

def bootstrap_classification_ci(model, test_loader, num_bootstraps=1000, confidence_level=0.95):
    """
    Calculates confidence intervals for classification using bootstrapping.

    Args:
        model: A trained PyTorch classification model.
        test_loader: A DataLoader for the test dataset.
        num_bootstraps: Number of bootstrap samples.
        confidence_level: Desired confidence level (e.g., 0.95 for 95% CI).

    Returns:
      A dictionary containing confidence intervals for each test sample,
      structured by test sample index and class label.
    """

    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(test_loader):
            outputs = F.softmax(model(inputs), dim=1)
            all_probs.append(outputs.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0) # (num_test, num_classes)
    num_test = all_probs.shape[0]
    num_classes = all_probs.shape[1]

    bootstrapped_probs = np.zeros((num_bootstraps, num_test, num_classes))

    for boot in range(num_bootstraps):
       sample_indices = np.random.choice(num_test, size=num_test, replace=True)
       bootstrapped_probs[boot] = all_probs[sample_indices]
    
    lower_percentile = (1 - confidence_level) / 2 * 100
    upper_percentile = (1 + confidence_level) / 2 * 100

    confidence_intervals = {}
    for test_idx in range(num_test):
        confidence_intervals[test_idx] = {}
        for class_idx in range(num_classes):
           distribution = bootstrapped_probs[:, test_idx, class_idx]
           lower_bound = np.percentile(distribution, lower_percentile)
           upper_bound = np.percentile(distribution, upper_percentile)
           confidence_intervals[test_idx][class_idx] = (lower_bound, upper_bound)

    return confidence_intervals


# Example usage:
if __name__ == '__main__':
    # Dummy Model
    class DummyClassifier(nn.Module):
        def __init__(self, num_classes=3):
          super(DummyClassifier, self).__init__()
          self.fc1 = nn.Linear(10, 20)
          self.fc2 = nn.Linear(20, num_classes)

        def forward(self, x):
          x = F.relu(self.fc1(x))
          x = self.fc2(x)
          return x

    dummy_model = DummyClassifier(num_classes=3)

    # Dummy Data
    num_test_samples = 100
    X_test = torch.randn(num_test_samples, 10)
    y_test = torch.randint(0, 3, (num_test_samples,)) # Dummy labels, not used for CIs
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    confidence_intervals = bootstrap_classification_ci(dummy_model, test_loader)
    
    # Example print for the first test sample
    print("Confidence Intervals for Sample 0:")
    for class_label, (lower, upper) in confidence_intervals[0].items():
      print(f"  Class {class_label}: [{lower:.4f}, {upper:.4f}]")
```

The `bootstrap_classification_ci` function takes the model and test data loader, and optionally the number of bootstraps and the confidence level as parameters. It iterates through the test set in mini-batches. For each data point, it computes the probability of each class. It then creates multiple bootstrap samples, computes their predicted probabilities using the model, and calculates the required percentiles for confidence intervals based on the distribution of these bootstrapped predictions, returning the lower and upper bound of the interval per sample, per class.

**Regression: Confidence Intervals Using Prediction Intervals**

For regression tasks, we are dealing with continuous predictions, and we can attempt to capture prediction variability using various techniques. Unlike classification, the output directly represents a continuous value. One approach is to generate prediction intervals by modelling the uncertainty associated with the model. A common approach is to use conformal prediction or quantile regression. Here, I will focus on an approach that uses the residual distribution from training data.

The core principle here involves analyzing the model's training-time residuals. A residual is the difference between the true label and the predicted value. During training, I would retain a record of these residuals on the training set. After training, and when evaluating on test data, I would calculate the standard deviation of the training residuals, and I would then use it to establish a confidence interval for new predictions. I would multiply this standard deviation by a constant derived from the properties of the normal distribution (1.96 for a 95% confidence interval, assuming normally distributed residuals). Then, by adding and subtracting this value from the predicted value, I can obtain a confidence interval for regression.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

def regression_ci(model, train_loader, test_loader, confidence_level=0.95):
    """
    Computes confidence intervals for regression using residual analysis.

    Args:
        model: A trained PyTorch regression model.
        train_loader: DataLoader for training data.
        test_loader: DataLoader for test data.
        confidence_level: Desired confidence level.

    Returns:
        A dictionary with prediction and associated CI for each sample in the test set.
    """

    model.eval()
    train_residuals = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            outputs = model(inputs).squeeze()
            residuals = (targets - outputs).cpu().numpy()
            train_residuals.extend(residuals)
    
    residual_std = np.std(train_residuals)
    z_score = 1.96 # Standard z-score for 95% CI if assuming normality

    confidence_intervals = {}
    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(test_loader):
           outputs = model(inputs).squeeze().cpu().numpy()
           for i, pred in enumerate(outputs):
              lower_bound = pred - z_score * residual_std
              upper_bound = pred + z_score * residual_std
              test_index = batch_idx * test_loader.batch_size + i
              confidence_intervals[test_index] = {"prediction": pred, "ci": (lower_bound, upper_bound)}
    return confidence_intervals

# Example usage
if __name__ == '__main__':
    class DummyRegressor(nn.Module):
        def __init__(self):
            super(DummyRegressor, self).__init__()
            self.fc1 = nn.Linear(5, 10)
            self.fc2 = nn.Linear(10, 1)

        def forward(self, x):
           x = F.relu(self.fc1(x))
           x = self.fc2(x)
           return x

    dummy_model = DummyRegressor()

    # Dummy training data
    X_train = torch.randn(100, 5)
    y_train = torch.randn(100)
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32)
    
    # Dummy test data
    X_test = torch.randn(50, 5)
    y_test = torch.randn(50) # Dummy labels, not used in CIs
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Training (a dummy training loop)
    optimizer = optim.Adam(dummy_model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    for epoch in range(5):
      for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = dummy_model(inputs).squeeze()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    

    confidence_intervals = regression_ci(dummy_model, train_loader, test_loader)
    print("Confidence Intervals for the first test sample:")
    first_sample_ci = confidence_intervals[0]
    print(f"  Prediction: {first_sample_ci['prediction']:.4f}, CI: [{first_sample_ci['ci'][0]:.4f}, {first_sample_ci['ci'][1]:.4f}]")
```

The `regression_ci` function calculates residuals on the training set after the model has been trained. It then calculates the standard deviation of these residuals. The function then computes the confidence interval for the test dataset using this residual standard deviation and the appropriate z-score. This approach provides a relatively straightforward method for obtaining a confidence interval, but its effectiveness is heavily dependent on the assumption that residuals are normally distributed and stationary. This assumption should be validated to obtain valid results.

**Additional Considerations**

When dealing with complex or high-stakes scenarios, it’s prudent to consider more advanced techniques, such as Bayesian inference, for calculating prediction uncertainty in both regression and classification settings. In Bayesian Neural Networks, for example, model parameters are treated as distributions, and predictions can be naturally associated with the level of uncertainty. Additionally, cross-validation should be employed to generate reliable results from the techniques mentioned.

**Resource Recommendations**

For individuals interested in deepening their understanding of confidence intervals, I suggest consulting texts on statistical inference and Bayesian statistics. These resources typically provide foundational knowledge on concepts such as hypothesis testing, parameter estimation, and interval estimation. Furthermore, machine learning textbooks often contain chapters dedicated to model evaluation, including techniques for handling uncertainty in predictions. Additionally, journals related to machine learning research frequently publish articles that demonstrate advanced methodologies and techniques for quantifying the uncertainty of neural networks.
