---
title: "Can logistic regression achieve 100% accuracy on random data?"
date: "2024-12-23"
id: "can-logistic-regression-achieve-100-accuracy-on-random-data"
---

Alright, let's unpack this question about logistic regression and perfect accuracy on random data. It's a scenario that, in my experience, seems counterintuitive initially but illuminates some core statistical concepts. I've seen this come up in several modeling challenges over the years, and the answer is nuanced, but ultimately, *yes*, under very specific and, frankly, contrived circumstances, logistic regression *can* achieve 100% accuracy on what appears to be random data. It's crucial, however, to understand *why* and what it actually signifies.

The idea that a statistical model could perfectly predict random outcomes raises red flags. We’re conditioned to think that randomness, by its very nature, should be unpredictable. When dealing with true, independent random variables, no model should consistently achieve perfect classification. Yet, we must consider what 'random' actually means in the context of the data and our model.

First, let’s clarify what logistic regression does. It’s a method used for binary classification tasks. It estimates the probability of an event occurring (e.g., a customer will click an ad, a patient has a specific condition) based on the input features. The model outputs a value between 0 and 1, which is then typically thresholded (usually at 0.5) to predict either class 0 or class 1. Now, the crux of the matter comes down to how the ‘random’ data is generated, and the dimensionality of that data compared to the number of samples. If the 'random' data, rather than stemming from an external generative process that cannot be modelled, is a result of a specific construction where, even without an apparent semantic or practical meaning to the features, an inherent pattern emerges *specifically in the data observed*, this is where the phenomenon occurs. This pattern is not in the underlying distribution but emerges in the actual dataset sample.

Let's consider a scenario where we generate ‘random’ data in a way that inherently creates separability by random assignment within a training set. It is this type of generation that is key to this phenomenon, not simply a random draw of an iid distribution. Imagine I’m working on some test validation during a previous consultancy, generating test datasets. I had to develop a method to generate test cases that have known answers, and this was where I ran into it. Assume we have two classes (0 and 1), and we generate *n* samples, each with *p* features. However, crucially, instead of generating *all* feature values independently and then assigning labels, we first assign labels randomly and then make a random generation constrained by the assigned label. In particular we can make the samples in each label distinct from another by ensuring that a model can use the feature space to separate them. For example, say *p* is 3, we could force the feature 1 value to be different, then a simple threshold will allow perfect separation in a model trained on this generated data.

Here's an example in python using `numpy` and `scikit-learn` to illustrate this:

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def generate_separable_random_data(n_samples, n_features):
    X = np.random.rand(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)  # Random labels initially
    for i in range(n_samples):
        if y[i] == 0:
           X[i, 0] = np.random.uniform(0, 0.4)
        else:
           X[i, 0] = np.random.uniform(0.6, 1) # enforce separation in first feature

    return X, y

n_samples = 100
n_features = 3
X, y = generate_separable_random_data(n_samples, n_features)

model = LogisticRegression(solver='liblinear')
model.fit(X, y)
y_pred = model.predict(X)

accuracy = accuracy_score(y, y_pred)
print(f"Accuracy: {accuracy}")
```

In the snippet above, the labels are still ‘random’, but the *features* are not completely independent of the label as the first feature has values that can be used to perfectly separate the data. This is not randomness in the distribution, but instead is separability of the *sample* under a label assignment. Because the first feature is constructed in a way where all label 0 samples are <0.4, and label 1 samples are >0.6, the model will quickly learn that if the first feature is <0.5, the label will be 0, otherwise the label is 1. The other features are completely random, but that doesn’t stop the model from achieving 100% accuracy. The separation is not in the underlying distribution, but present in this particular training data sample.

Now, you might ask, how is this different from genuine random data? In genuine random data, the underlying generating distribution lacks any patterns that would let us reliably distinguish classes. It might be something as simple as sampling from a Bernouli distribution to generate the labels and then sampling from a normal distribution, independent of the label, to generate the features. In that case, the model would not achieve 100% accuracy.

Let's demonstrate this using a different `generate_random_data` function:

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def generate_random_data(n_samples, n_features):
  X = np.random.rand(n_samples, n_features)
  y = np.random.randint(0, 2, n_samples)
  return X, y


n_samples = 100
n_features = 3
X, y = generate_random_data(n_samples, n_features)

model = LogisticRegression(solver='liblinear')
model.fit(X, y)
y_pred = model.predict(X)

accuracy = accuracy_score(y, y_pred)
print(f"Accuracy: {accuracy}")
```

In this case, the data is not constructed to enable any kind of separation. Here, it's genuinely random. The accuracy will likely hover around 50%, indicating the model isn’t learning meaningful patterns.

The crucial distinction lies in how we’re generating the data. Our first method *introduces* an artificial pattern during generation that can then be exploited by the logistic regression model. This is not the same as a meaningful underlying relationship, and this type of synthetic data is not representative of real world challenges.

One more example can be illustrative. If the number of features significantly exceeds the number of samples, and we allow the model to overfit, we can also achieve a close to 100% accuracy on this artificially structured data. We modify the first function, in this case, to have a large feature space, and then ensure, through the same construction as before, we force a separation.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def generate_high_dim_separable_random_data(n_samples, n_features):
    X = np.random.rand(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)  # Random labels initially
    for i in range(n_samples):
        if y[i] == 0:
           X[i, 0] = np.random.uniform(0, 0.4)
        else:
           X[i, 0] = np.random.uniform(0.6, 1) # enforce separation in first feature
    return X, y

n_samples = 100
n_features = 500
X, y = generate_high_dim_separable_random_data(n_samples, n_features)

model = LogisticRegression(solver='liblinear')
model.fit(X, y)
y_pred = model.predict(X)

accuracy = accuracy_score(y, y_pred)
print(f"Accuracy: {accuracy}")
```

In this scenario we still ensure the first feature is separable, but we dramatically increase the number of features. Now a logistic regression will overfit to the training data and give a perfect (or near-perfect) classification. This high dimensional feature space combined with structured data within the feature space is what leads to the phenomenon.

So, can logistic regression achieve 100% accuracy on random data? The answer, technically, is yes, but it comes with a large asterisk. This happens when the generation of what is *called* random data, is not true random data but artificial structured data that is separate within the feature space and when the feature space is large enough to allow overfitting. These situations don't violate the underlying logic of the algorithm; rather, they expose that what seems like ‘randomness’ might not be so random to the model, particularly with high-dimensional data. In general, if you see your model achieving perfect accuracy on data you believe to be random, it's time for a very critical review of both your data generation and validation procedures.

For those looking to dive deeper into the theoretical underpinnings of this behavior, I’d recommend starting with *The Elements of Statistical Learning* by Hastie, Tibshirani, and Friedman. It's a dense book, but it provides a thorough grounding in the statistical learning theory that is relevant here. For a more introductory text, *An Introduction to Statistical Learning* by James, Witten, Hastie, and Tibshirani is a very good resource. These provide a more rigorous framework for understanding the relationship between model complexity and the underlying properties of the data, including the dimensionality issues that lead to overfitting. Understanding the bias-variance trade-off, as thoroughly explained in these texts, is key to understanding why certain models perform so well (and often disastrously) on various types of datasets. The 'curse of dimensionality', a concept explored in-depth in these resources, also sheds light on why increasing features (while not necessarily adding value) can lead to misleading accuracy.

In real world problems, your dataset will not be constructed this way, so the issue is not usually something you would see directly. But understanding the underlying issues that enable this phenomenon, even on such artificially created datasets, helps greatly in interpreting the behaviour of models and the data they learn from. It's always useful to think about all of the ways you could introduce flaws into a dataset, even seemingly random data, as this ensures that your models remain reliable and robust.
