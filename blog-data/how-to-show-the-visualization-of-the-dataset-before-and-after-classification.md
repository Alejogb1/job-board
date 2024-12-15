---
title: "How to show the visualization of the dataset before and after classification?"
date: "2024-12-15"
id: "how-to-show-the-visualization-of-the-dataset-before-and-after-classification"
---

alright, i've been around the block a few times with machine learning, specifically with classification problems. visualising datasets pre and post classification, well, that's a crucial step, not just for debugging but also to actually see what your model is doing. it can often be a gut check. so, let’s break this down based on my experiences over the years.

first, let’s talk about pre-classification. you’ve got your data, right? maybe it's tabular, maybe it’s spatial, whatever, the core idea is the same: you want to understand its structure. this is where scatter plots and histograms become our best friends. if your data has two dimensions – we are lucky! – a simple scatter plot immediately reveals clusterings, potential separability, outliers. i’ve seen datasets where it became immediately obvious, just by eyeballing the plot, that the problem wasn’t a tough classification task, a simple threshold might've done the job, saving us from implementing a full blown neural network. but that's not always the case. i’ve also had datasets that looked like a complete mess in 2d, so dimensionality reduction techniques become needed.

```python
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA

def visualize_pre_classification(data, labels=None, dimensions=2):
    """Visualizes data before classification.

    Args:
        data (pd.DataFrame or np.ndarray): The input data.
        labels (np.ndarray, optional): Class labels if available.
        dimensions (int): Dimensions to reduce data to, if larger than 2.
    """
    if isinstance(data, pd.DataFrame):
        data = data.to_numpy()

    if data.shape[1] > dimensions:
         pca = PCA(n_components=dimensions)
         reduced_data = pca.fit_transform(data)
    else:
         reduced_data= data

    plt.figure(figsize=(8, 6))
    if labels is not None:
        scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis')
        plt.legend(*scatter.legend_elements(), title="Classes")
    else:
        plt.scatter(reduced_data[:, 0], reduced_data[:, 1])

    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('Data Visualization Before Classification')
    plt.grid(True)
    plt.show()

# example use
# assume you have a pandas dataframe called 'df'
# or a numpy array 'data' and labels if applicable
# you could load data from a csv and call this function like this
# df = pd.read_csv('my_dataset.csv')
# features = df.drop('target_column', axis=1)
# labels = df['target_column']
# visualize_pre_classification(features, labels)
```
this python snippet, uses `matplotlib` and `scikit-learn`. it handles both numpy arrays and pandas dataframes, and if you have more than two features it reduces to two components using pca for easier visualization. you pass in your data and optionally labels for color-coding. this is useful if you already have labels, as in supervised learning, or if you want to use the 'real' target to see how they are distributed. i once had a dataset that was supposed to have three clear clusters, but this revealed only two groups, and i realised a mistake in the data preparation phase, avoiding hours of training a flawed model. that taught me a good lesson, never skip this pre-visualization.

now, post-classification is a whole different game. you’ve thrown your data into the model, got your predictions. what now? well, first off, if it’s a binary classification, showing your original data coloured by your predictions is a powerful way to detect misclassifications. for multi-class tasks, the same can be done but it's more complex. plotting the decision boundary, if feasible, is something I do a lot, especially with simpler models. imagine for a second a 2d scatterplot with a complex curve cutting through the space. that curve separates your predicted classes. it's a really useful way to spot where your model is struggling, and sometimes this reveal why that specific model is not performing well for the data. I had an experience when visualizing a support vector machine decision boundary; the boundary was completely ignoring some of the training points, and this lead me to conclude the kernel needed further adjustments, which significantly improved my model. it is true that it’s harder to visualise when the data goes beyond two dimensions.

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap

def visualize_post_classification(data, true_labels, predictions, model=None, dimensions=2):
    """Visualizes data after classification, including decision boundaries if a model is provided.

    Args:
        data (pd.DataFrame or np.ndarray): The input data.
        true_labels (np.ndarray): True class labels.
        predictions (np.ndarray): Model predictions.
        model: Fitted model to plot decision boundary if provided.
        dimensions (int): Dimensions to reduce data to, if larger than 2.
    """
    if isinstance(data, pd.DataFrame):
       data = data.to_numpy()

    if data.shape[1] > dimensions:
         pca = PCA(n_components=dimensions)
         reduced_data = pca.fit_transform(data)
    else:
         reduced_data= data

    plt.figure(figsize=(12, 6))
    # color coded by true and predicted labels
    cmap = ListedColormap(['#FF0000', '#0000FF', '#00FF00'])
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=true_labels, cmap=cmap, alpha=0.7, edgecolors='k')
    plt.legend(*scatter.legend_elements(), title="True Classes")
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('Data Points (True Labels)')
    plt.grid(True)


    plt.subplot(1, 2, 2)
    scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=predictions, cmap=cmap, alpha=0.7, edgecolors='k')
    plt.legend(*scatter.legend_elements(), title="Predicted Classes")
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('Data Points (Predicted Labels)')
    plt.grid(True)


    if model and dimensions == 2:
        # Plot decision boundary (only for 2D data and if a model is provided)
       xx, yy = np.meshgrid(np.linspace(reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1, 200),
                          np.linspace(reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1, 200))
       Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
       plt.contourf(xx, yy, Z, alpha=0.4, cmap=cmap)

    plt.tight_layout()
    plt.show()


#example
# assume data, true_labels, predictions are available as numpy arrays or pandas series
# let's assume that you have a trained model called 'model',
# if you trained a classifier like LogisticRegression or Support Vector Machine
# it's possible to add model = your_model into the post visualization
# and it will plot the decision boundaries if data is 2D
# visualize_post_classification(data, true_labels, predictions, model = model)
# note you may need to scale the data before if your model is sensitive to scaling like Support Vector Machine.
```

this updated code visualizes post-classification results. the new code shows the data with true labels in one subplot and predicted labels in another subplot, making it easy to compare where the model's predictions align with actual values. if a model is provided and the data is reduced to two dimensions, it will try to show decision boundaries using contour plots. keep in mind that models that are tree based or neural network based may not work with a standard predict call, but for models with decision boundaries, this works great.  i once was trying to diagnose the poor performance of a model. after plotting the true and predicted values, it became quite clear it was not a problem with the model hyperparameters itself but the data itself, which had a lot of inconsistent labeling in my training data.

for datasets with high dimensionality, t-sne or umap might be the way to go for dimensionality reduction. these tools often reveal clusters that pca doesn’t capture due to non-linear projections. for classification tasks, this is especially useful because the clusters created with these methods tend to be more correlated to your classes. i won’t put a snippet for those as there are many open source libraries that can do that job for you, you could check out for resources the  "visualizing data using t-sne" by maaten and hinton or "umap: uniform manifold approximation and projection for dimension reduction" by mcinnes. both papers will give you a solid background on this techniques.

and, finally, one thing i’ve learned the hard way is to not trust my eyes. this process should be complemented with metrics for classification like precision, recall, f1-score and roc curves, the more you see and validate, the better you understand the model's behaviour. in fact, the most obvious errors are often the ones staring right at us, hiding in plain sight like that one bug i spent two days trying to find in my code that was actually a misspelled variable, that's me being a professional code monkey i suppose.

so, to sum it up, visualize your data before and after classification, it can save you hours of debugging, and can help in the feature engineering phase. there is a plethora of books and articles on the topics, for more theoretical background i recommend “the elements of statistical learning” by hastie, tibshirani and friedman. this is all based on my experience, i'm sure others might have different techniques, but i've found these worked for me quite well.
```python
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

def visualize_confusion_matrix(true_labels, predicted_labels, class_names = None):
    """Visualizes the confusion matrix.

    Args:
        true_labels (np.ndarray): True class labels.
        predicted_labels (np.ndarray): Model predictions.
        class_names (list, optional): List of class names if available.
    """

    cm = confusion_matrix(true_labels, predicted_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.grid(False)
    plt.show()

# example of use
# true_labels and predicted_labels are assumed to be arrays of labels
# you can also add class names if available, for example, class_names = ['cat','dog','mouse']
# visualize_confusion_matrix(true_labels, predicted_labels)
```
This snippet produces a visual representation of your classifier's performance. It shows a grid where each cell indicates how many instances of a true class were classified as each predicted class. The diagonal shows the correctly classified instances. This visual tool, alongside the others, has saved me from many headaches.
