---
title: "How to plot multiple ROC figures in a single plot?"
date: "2024-12-15"
id: "how-to-plot-multiple-roc-figures-in-a-single-plot"
---

ah, multiple roc curves on one plot, i've been there, done that, got the t-shirt (and the slightly singed eyebrows from that one time the plotting library decided to go rogue, but that's a story for another day). it's a common scenario when you're comparing different classifiers or different variations of the same classifier, and you need to see how they stack up visually. let's break it down.

first, you're dealing with the receiver operating characteristic (roc) curve which, as you probably are aware of, is a plot of the true positive rate against the false positive rate at various threshold settings. each classifier, or each variation thereof, will give you a separate set of these values and, thus, its own curve.

what you want is not to create separate images for each curve, but rather superimpose all of them in one go so you can visually compare and judge their overall performance. the key here is organizing your data correctly and then using a library that handles plotting. i've found that matplotlib, with a few tweaks, works beautifully for this purpose, and this is what i'm going to focus on here, although other visualization libraries can do it.

let’s get down to the nitty gritty, because that’s where most of these issues come up. i once spent half a night trying to get roc curves for multiple models to plot, only to find out i was passing a python list instead of numpy arrays in the wrong order to the plot function. let's avoid that from happening to you. first, assume you have already calculated all the false positive rates (fpr), true positive rates (tpr) and have them neatly stored along with the labels of the classifier. usually, these rates are calculated based on your classification model's output and its target values. libraries like scikit-learn provide handy ways to compute these curves and are widely used in the field.

now to the core of the task, getting those curves onto the same plot. below is a code example using matplotlib, the workhorse for most scientific plots in python, and using the assumption i previously stated about having your precomputed rates.

```python
import matplotlib.pyplot as plt

def plot_multiple_roc_curves(curves, title="roc curves comparison"):
    """
    plots multiple roc curves on a single figure.

    args:
        curves (dict): a dictionary where keys are model names (str) and values are
                      tuples containing (fpr, tpr).

    returns:
        none (displays the plot)
    """
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='random') # plot random classifier
    for name, (fpr, tpr) in curves.items():
       plt.plot(fpr, tpr, label=f'{name} curve')

    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    # fictional example data, replace with your actual data
    roc_curves = {
        'model a': ([0, 0.1, 0.3, 0.5, 0.9, 1], [0, 0.4, 0.7, 0.8, 0.95, 1]),
        'model b': ([0, 0.2, 0.4, 0.6, 0.8, 1], [0, 0.2, 0.5, 0.85, 0.97, 1]),
        'model c': ([0, 0.15, 0.25, 0.55, 0.75, 1], [0, 0.5, 0.65, 0.75, 0.99, 1])
    }

    plot_multiple_roc_curves(roc_curves, 'example roc curves')
```

in the example above, i am creating a dictionary where each key is the name of the model, and each value is a tuple containing the false positive rate and true positive rate of that particular model (both of them as an array-like structure). the `plot_multiple_roc_curves` function iterates through this dictionary and plots each curve. we also add a random baseline, represented as a dashed line, for comparison. we add labels, legend, titles and grid just to make it pretty to see.

you might notice i'm using f-strings for the labels. this way you can easily pass the name of the model dynamically, it comes in handy a lot when exploring different variations of models and classifiers. this is the kind of small detail that i wish someone told me when i was starting with scientific plotting, i used to create labels manually all the time, and that was a hassle.

the function is pretty flexible, and you can pass any dictionary of roc curves. however, it assumes your data is well structured and of the same shape. let me show you a way to compute the roc curve directly from the model's predictions. if you have scikit-learn, you can easily calculate the tpr and fpr.

```python
from sklearn.metrics import roc_curve
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

def get_roc_curves(models, x, y):
    """
    calculates roc curves for multiple models.

    args:
        models (dict): a dictionary of models where keys are model names (str) and values
                      are model instances that implement `predict_proba`.
        x (array like): the input data
        y (array like): the target data

    returns:
        dict: a dictionary where keys are model names (str) and values are
                      tuples containing (fpr, tpr).
    """
    curves = {}
    for name, model in models.items():
        # fit the model on the dataset
        model.fit(x, y)
        # get the predicted probabilities for the positive class (class 1)
        y_prob = model.predict_proba(x)[:, 1]

        # calculate the roc curve
        fpr, tpr, thresholds = roc_curve(y, y_prob)
        curves[name] = (fpr, tpr)
    return curves


if __name__ == '__main__':
    # fictional example data, replace with your actual data
    x, y = make_classification(n_samples=1000, n_features=10, random_state=42)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    models_to_test = {
        'model a': LogisticRegression(random_state=42, solver='liblinear'),
        'model b': LogisticRegression(random_state=1, solver='liblinear', penalty='l1', C=0.1),
        'model c': LogisticRegression(random_state=4, solver='liblinear', penalty='l2', C=10)
    }
    roc_curves_dict = get_roc_curves(models_to_test, x_train, y_train)
    plot_multiple_roc_curves(roc_curves_dict, 'roc curves from scikit learn')
```

here's a breakdown of what's happening above. i am simulating a dataset for the purpose of the example using scikit-learn utilities, then i'm creating three dummy models, all logistic regressions but with different variations for regularization purposes, and creating a dictionary where i’m storing each model's name with its object.  then i create a function `get_roc_curves` which receives the models, the inputs and the targets. inside this function, for each model, i fit the data and obtain the probability of each instance to be part of the positive class. with these probabilities and the true targets i calculate the `fpr` and `tpr` using scikit-learn `roc_curve` method. and lastly, i call the plot function we created before with the dictionary resulting from `get_roc_curves` function.

notice that i’m passing the results of `model.predict_proba(x)[:, 1]` to `roc_curve`, the `[:, 1]` part is crucial here, you should pass only the probabilities of the positive class. i've seen this mistake so many times when a colleague would plot completely wrong roc curves.

and of course, once you see your results you'll probably want to compare the area under the roc curve (auc). you can simply use another method from scikit-learn: `from sklearn.metrics import roc_auc_score`. with this function, you pass your true target and the probabilities of each instance to be part of the positive class to obtain the numerical value representing the area under the curve, this will help you quantify the models performance in case visual inspection is not enough.

```python
from sklearn.metrics import roc_auc_score

def print_auc_scores(models, x, y):
    """
    prints auc scores for multiple models.

    args:
        models (dict): a dictionary of models where keys are model names (str) and values
                      are model instances that implement `predict_proba`.
        x (array like): the input data
        y (array like): the target data

    returns:
        none (prints on standard output)
    """
    for name, model in models.items():
        # fit the model on the dataset
        model.fit(x, y)
        # get the predicted probabilities for the positive class (class 1)
        y_prob = model.predict_proba(x)[:, 1]
        auc = roc_auc_score(y, y_prob)
        print(f'auc for {name} is {auc:0.3f}')

if __name__ == '__main__':
    # fictional example data, replace with your actual data
    x, y = make_classification(n_samples=1000, n_features=10, random_state=42)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    models_to_test = {
        'model a': LogisticRegression(random_state=42, solver='liblinear'),
        'model b': LogisticRegression(random_state=1, solver='liblinear', penalty='l1', C=0.1),
        'model c': LogisticRegression(random_state=4, solver='liblinear', penalty='l2', C=10)
    }
    print_auc_scores(models_to_test, x_train, y_train)
```
here you can see i created the `print_auc_scores` function which works very similar to the `get_roc_curves` function, but instead of returning the curves, it prints the computed auc for each model. it should help you make more informed decisions. i also kept the same example setup as the previous example to be easily tested.

now, as for resources, i’ve personally found the following to be quite helpful:

*   "the elements of statistical learning" by hastie, tibshirani, and friedman: this book is a classic for a reason. it has a complete chapter on classification and model evaluation where the roc curve is explored. it's not always light reading, but it will give you a solid foundation.
*   "pattern recognition and machine learning" by bishop: similar to the one above, but with a more probabilistic view. the mathematical rigor in this book is worth the effort, also contains a complete section about roc curves and classification analysis.
*   scikit-learn’s documentation: the user guides are actually pretty good for understanding how to use the specific functions i mentioned above (`roc_curve`, and `roc_auc_score`). it’s worth checking. also look into the different example notebooks they have, you will learn a ton.

in conclusion, plotting multiple roc curves in one figure is pretty straightforward once you grasp the structure of the roc curve data and find the correct data structure for matplotlib. organize your data, use matplotlib, and always look closely at your axes to make sure they make sense (it's not fun trying to debug a plot that looks completely off but has a perfectly fine implementation under the hood). oh, and always double-check if you are selecting the right probability class before plotting, your results might look like something an ai drew if you don't. hope this helps!
