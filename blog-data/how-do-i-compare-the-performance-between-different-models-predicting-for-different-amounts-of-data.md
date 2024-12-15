---
title: "How do I compare the performance between different models predicting for different amounts of data?"
date: "2024-12-15"
id: "how-do-i-compare-the-performance-between-different-models-predicting-for-different-amounts-of-data"
---

alright, so you're looking into how to evaluate model performance when the amount of training data varies, i've been down that road a few times, and it can get tricky. it’s not as straightforward as just comparing final accuracy scores. let's break it down, i'll share some things i've picked up along the way, and show some code examples.

first off, the core issue is that the amount of data strongly influences a model's ability to learn. with very little data, the model might overfit and perform terribly on unseen data even though it looks pretty good on what little training data it has seen. with more data, it usually improves, up to a certain point. so, if one of your models has been trained on a lot more data than the other, a direct accuracy comparison isn't going to be very meaningful. it’s like comparing a student that only had one day to study with a student that had a whole year to prepare for the exam, unfair isn't it?

a common technique here is to use learning curves. basically, you train each model multiple times, each time with progressively more data. then you plot performance (could be accuracy, f1-score, loss, whatever metric you are using) against the amount of data. this will show you how each model's performance scales with increasing amounts of data, and you can use this to make more informed comparisons.

i remember a time, i think it was about 2018, when i was working on a text classification problem. i had a convolutional neural network and a recurrent neural network, and i was getting frustrated because sometimes the cnn was better, and sometimes the rnn. finally, i tried learning curves, i found that the rnn was much better with small amounts of training data, but after a certain point the cnn had better results. that helped me a lot to understand what was happening. in that case, i was able to save time and computational resources.

here is an example of some python code using `sklearn` and `matplotlib` to illustrate how to create learning curves:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def learning_curve(model, x, y, train_sizes=np.linspace(0.1, 1.0, 10), cv=5):
    train_scores = []
    test_scores = []
    for size in train_sizes:
        train_score_fold = []
        test_score_fold = []
        for i in range(cv):
            x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=size, shuffle=True)
            model.fit(x_train, y_train)
            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)
            train_score_fold.append(accuracy_score(y_train, y_train_pred))
            test_score_fold.append(accuracy_score(y_test, y_test_pred))

        train_scores.append(np.mean(train_score_fold))
        test_scores.append(np.mean(test_score_fold))
    return train_sizes, train_scores, test_scores

if __name__ == '__main__':
    #dummy data
    x = np.random.rand(1000, 10)
    y = np.random.randint(0, 2, 1000)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(solver='liblinear'))
    ])

    sizes, train_scores, test_scores = learning_curve(pipeline, x, y)

    plt.figure(figsize=(10, 6))
    plt.plot(sizes, train_scores, label='train accuracy')
    plt.plot(sizes, test_scores, label='test accuracy')
    plt.xlabel('training set size')
    plt.ylabel('accuracy')
    plt.title('learning curve')
    plt.legend()
    plt.grid(True)
    plt.show()
```
this script generates a basic learning curve, it uses `sklearn` for model and metrics, and `matplotlib` for plotting. the most important part is the `learning_curve` function that splits the data into different training sizes and then calculates the accuracy. you should adjust the `train_sizes`, `cv` (cross-validation splits), and the model according to your specific situation.

another thing to consider is the variance in performance, which means that one specific train could be better or worse than another, that’s why the code above is doing `cv` folds (it trains the model with n different initializations and returns an average). this becomes particularly apparent when dealing with small training sets. a single train can produce huge fluctuations in performance and this may cause confusion while comparing two different models.

that’s why it’s a good idea to run multiple trials with different train data splits, or different random initializations, for each data size and average out the scores. you also can calculate the standard deviation of those scores which will give you an idea about how much the results vary.

now, let's discuss the model selection process when the datasets size is not homogeneous. in my experience, if i need to pick one model with a specific dataset size, i use the learning curves to find the optimal size for training data, once i do this, i can train models with that specific data size and compare the performance directly using metrics like accuracy, recall or f1-score. if for some reason there isn't any data for making the learning curves it's better to use cross-validation for estimating model performance. the code bellow shows an example.

```python
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, f1_score

def evaluate_model(model, x, y, cv=5):
   scores = cross_val_score(model, x, y, cv=cv, scoring=make_scorer(f1_score))
   return scores.mean(), scores.std()


if __name__ == '__main__':
    # Dummy data
    x = np.random.rand(1000, 10)
    y = np.random.randint(0, 2, 1000)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(solver='liblinear'))
    ])
    mean, std = evaluate_model(pipeline, x, y)
    print(f"mean f1 score: {mean:.3f}, std: {std:.3f}")
```

this one uses cross-validation to give an estimation of the f1-score with it’s variance, i’m using the f1-score because it’s useful when classes are unbalanced, if the classes are balanced you can use the accuracy. here i am using an external `make_scorer` function to return f1-score as the metric used in cross-validation but you can replace `make_scorer(f1_score)` with `"accuracy"` if you want to use accuracy.

now, suppose that you have multiple different models with different architectures or different parameters, and you need to select the best. in that case you could use a grid search, in this case it is a good practice to use a specific cross-validation method called “nested cross-validation”. if you are using `sklearn` this can be done easily by using `GridSearchCV` as the outer cross validation strategy and inside it call again `GridSearchCV` with another model as the inner cross-validation. however, keep in mind this is computationally heavy. i once tried doing this with neural nets and it took me almost 3 weeks to complete the search (it did work though).

here is the code for doing this.

```python
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, f1_score

def nested_cross_validation(x, y, cv=5):

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression())
    ])

    param_grid = {'model__C': [0.1, 1.0, 10.0], 'model__solver':['liblinear', 'lbfgs']}
    outer_cv = KFold(n_splits=cv, shuffle=True)
    scores = []

    for train_index, test_index in outer_cv.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        inner_cv = KFold(n_splits=cv, shuffle=True)
        grid_search = GridSearchCV(pipeline, param_grid, cv=inner_cv, scoring=make_scorer(f1_score))
        grid_search.fit(x_train, y_train)
        best_model = grid_search.best_estimator_
        score = f1_score(y_test, best_model.predict(x_test))
        scores.append(score)
    
    return np.mean(scores), np.std(scores)


if __name__ == '__main__':
   # Dummy data
    x = np.random.rand(1000, 10)
    y = np.random.randint(0, 2, 1000)

    mean, std = nested_cross_validation(x, y)
    print(f"nested cross-validation score mean: {mean:.3f}, std: {std:.3f}")
```

in this code, we are performing cross-validation inside cross-validation. inside we select the best hyperparameters of a logistic regression and the final output is an estimate of its general performance. the `GridSearchCV` is doing the inner cross-validation.

as for resources, i recommend looking into books that discuss model evaluation techniques, for instance “the elements of statistical learning” by hastie, tibshirani, and friedman has a solid chapter on model assessment and selection. it might be a little dense if you're not into deep theory but it is very complete. also i recommend reading some papers on the topic, i remember that some years ago there was one paper that was very popular, it was titled “no model is ever exactly the same” (or something similar). there are a lot of papers on this topic.

remember that there isn't a single solution, different techniques could be used depending on the kind of data and models that you are using. this is the essence of experimenting, you can try many different approaches and see what works best in your case. oh, and remember to document what you did, or else you will forget what you did and wonder why your code is working, or why it isn’t working.
