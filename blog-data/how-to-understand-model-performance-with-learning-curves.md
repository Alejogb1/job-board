---
title: "How to understand Model Performance with Learning Curves?"
date: "2024-12-14"
id: "how-to-understand-model-performance-with-learning-curves"
---

let's dive into learning curves, something i've spent way too many late nights staring at. it's one of those things that seems simple on the surface, but the devil’s always in the details. i'll try to break it down based on how i've used them over the years, and hopefully it will help you out.

the core idea is that learning curves show how well your model is performing, both on data it's been trained on (the training set) and data it hasn't seen before (the validation or test set) as you feed it more and more training examples. this is different from a single accuracy score or f1-score etc, which only gives you a snapshot of performance at one point. learning curves display the evolution of performance with increasing training data.

i remember back in the early days of my ml tinkering, i was working on a image classification task with a custom cnn i created using tensorflow. i had decent scores, around 85% but it turned out that my model was overfitting. the curves made it crystal clear. the training score was almost 100%, and the validation was just stuck. that was my crash course in using learning curves to understand bias variance tradeoff, if the gap between the two curves gets really large as you feed it more data is a big sign that your model was memorising data rather than learning it, which it is called overfitting, not generalising well at all.

generally, the x-axis of the curve is the number of training samples used, or some fraction of total data used so far, or even time or number of training steps. the y-axis will be your chosen performance metric which could be accuracy, error, f1-score, or any appropriate measure depending on the task at hand.

the interesting aspect is how these two scores behave, train and validation, relative to each other, as training progresses. here is what you may expect:

1. **low bias, high variance (overfitting):** the training performance is high, even near perfect, but the validation performance is much lower and plateaus. there is a big gap between the two lines, this situation indicates that your model is learning the training data too well and doesn't generalize to new data. the solution here might be to reduce model complexity, add regularization, or collect more training data. i've seen this time and time again. think of a model that can perfectly classify pictures of cats only because it memorised every single pixel of the training cat images, but struggles with a cat image with a different background, it is not learning to recognise the key features that define a cat.

2. **high bias, low variance (underfitting):** both training and validation performances are low and are very close to each other. the performance doesn't improve much as you add more training data. this usually means that your model isn't complex enough to capture the patterns in the data. you may need a model with more parameters or richer features, or even rethinking your algorithm. its like trying to fit a linear model to non-linear data, it is just not going to work well regardless of how much data you feed it.

3. **ideal learning:** both curves converge towards a good performance, ideally plateauing at a high value with a small gap between the two lines. in an ideal scenario, the training score is also increasing alongside validation score. this indicates your model is learning well and generalising well, but there could still be room for improvement, maybe by further fine tuning the model, hyperparameters, or adding more data.

now, lets look at the implementation part, using python, sklearn, and matplotlib. this code is not a copy and paste and it requires some tweaks depending on your specific machine learning task and chosen model.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

#generate dummy data, replace with your own data
x, y = make_classification(n_samples=1000, n_features=20, n_informative=2, n_redundant=0, random_state=42)
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

# create a model, replace with your model
model = LogisticRegression(solver='liblinear', random_state=42)

# create learning curve
train_sizes, train_scores, test_scores = learning_curve(
   model, xtrain, ytrain, train_sizes=np.linspace(0.1, 1.0, 10), cv=5, scoring='accuracy'
)

# calculate mean and standard deviation of scores
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

# plotting
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, label='training score')
plt.plot(train_sizes, test_scores_mean, label='validation score')
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.2)
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.2)
plt.xlabel('training examples')
plt.ylabel('accuracy')
plt.title('learning curves')
plt.legend()
plt.grid(True)
plt.show()
```

that’s one example, if you use tensorflow keras, you can plot learning curve using the model history. in that scenario you don't use the sklearn function `learning_curve`, but instead rely on the `fit` function to get the training history and then plot it by hand. here is a more hands on approach in tensorflow:

```python
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# create dummy data
x, y = make_classification(n_samples=1000, n_features=20, n_informative=2, n_redundant=0, random_state=42)
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

# build a simple keras model
model = tf.keras.models.Sequential([
   tf.keras.layers.Dense(64, activation='relu', input_shape=(xtrain.shape[1],)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# train the model and capture training history
history = model.fit(xtrain, ytrain, epochs=50, batch_size=32, validation_data=(xtest, ytest), verbose=0)

# plot the training history
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.title('learning curves (keras)')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('learning curves (keras)')
plt.legend()
plt.grid(True)
plt.show()
```

it displays two plots now, one for accuracy and another for the loss of the training procedure. it gives a better and more nuanced view of what is happening.

in both cases, you will want to experiment with different hyperparameters, model architectures etc, plotting learning curves at each turn will give an insight on which way is the correct one. the idea here is to compare how much a change moves the graphs.

one last example, this time for a regression task with sklearn, this is to showcase that is not exclusive to classification and can be used in any supervised task.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# create dummy data for regression
x, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

# create a model, replace with your own model
model = LinearRegression()

# generate learning curve, uses negative mean square error because it is a scoring function, it is converted to positive when plotted
train_sizes, train_scores, test_scores = learning_curve(
   model, xtrain, ytrain, train_sizes=np.linspace(0.1, 1.0, 10), cv=5, scoring='neg_mean_squared_error'
)

# calculate mean and standard deviation for plotting
train_scores_mean = -np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = -np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

# plotting
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, label='training mse')
plt.plot(train_sizes, test_scores_mean, label='validation mse')
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.2)
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.2)
plt.xlabel('training examples')
plt.ylabel('mean squared error')
plt.title('learning curves (regression)')
plt.legend()
plt.grid(True)
plt.show()
```

remember, the key takeaway from all of this is to see if your model has high bias or high variance or a bit of both. in the previous examples you could easily find areas for improvement just by plotting the curves and analysing them.

for further study i would recommend some books like "hands on machine learning with scikit learn keras & tensorflow" by aurélien géron, or even the classic "the elements of statistical learning" by hastie, tibshirani, and friedman. these have in-depth discussions of model evaluation and the bias-variance tradeoff, also good sources are the classic machine learning papers on model evaluation by different authors, if you are more keen on that.

and a random joke: why was the learning curve so steep? because the model was trying to learn a linear equation with a convolutional neural network!

i hope that sheds some light on using learning curves, it's a powerful tool if used properly. good luck, and may your models converge nicely.
