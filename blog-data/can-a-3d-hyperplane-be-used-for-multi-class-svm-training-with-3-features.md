---
title: "Can a 3D hyperplane be used for multi-class SVM training with 3 features?"
date: "2024-12-23"
id: "can-a-3d-hyperplane-be-used-for-multi-class-svm-training-with-3-features"
---

Alright, let’s tackle this. A question about 3d hyperplanes and multi-class svm with three features – it's a topic I've encountered quite a few times, and I’ve definitely seen the confusion it can spark. Let's get into the weeds a bit.

The core idea here revolves around how Support Vector Machines (SVMs) function and how they extend to handle more than two classes. Now, when we talk about *features*, think of these as the dimensions of your data. In this case, you've got three. So each data point can be envisioned as existing in a 3D space – think of it as a point in a room where 'x', 'y', and 'z' are your three features. Now, an svm, in its simplest binary form, essentially tries to find the best *plane* (not hyperplane here, a simple 2d plane because the space is 3d) to divide these data points into two classes. This plane is defined by a normal vector and a bias (or offset) – that's how it's oriented and positioned in this 3d room.

The problem arises when you’ve got more than two classes. A single separating plane, however oriented, cannot logically separate three or more distinct groups within this 3d space, especially when data points are interleaved and not perfectly separable. To overcome this limitation, and get to what you were asking, you use a method of decomposition. This is a critical point – it's *not* a single 3d hyperplane separating everything at once. It involves decomposing the problem into a set of smaller, manageable binary classification problems.

The two major strategies I’ve seen, and personally implemented, are one-vs-all (ova), and one-vs-one (ovo). In ova, if you have, let's say, four classes, you would train four different SVM models. Each model would be trained to distinguish one particular class from all the others combined. In the 3d feature space case, each of these SVM models uses the same three features for classifying whether each point is in the current 'one' class or not (all other classes combined) using a 2d separating plane for each. In the one-vs-one strategy, it's different. With four classes, you’d construct *six* different models. Each one trained on only two class labels, again in each case using a separating plane.

Think of it this way. With three classes, using the ova technique, you would train three classifiers. The first classifier is trained to distinguish class 'a' from the combination of classes 'b' and 'c', again using a separating plane in this 3d space. The second distinguishes class 'b' from the combination of 'a' and 'c'. And the third separates class 'c' from the combined 'a' and 'b' classes. In all three of these situations, the decision boundary is a 2d plane within the 3d space defined by your features.

Let me illustrate this with code. I’m going to use python and scikit-learn as an example since it's commonly used in practice:

```python
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate some synthetic data for three classes with 3 features
np.random.seed(42) #for reproducibility
X = np.random.rand(150, 3)
y = np.random.choice([0, 1, 2], size=150)
X = X + y.reshape(-1,1) * np.random.rand(1,3)  # Add class-specific variation

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# One-vs-All (OVA) SVM
svm_ova = svm.SVC(kernel='linear', decision_function_shape='ovr')
svm_ova.fit(X_train, y_train)
y_pred_ova = svm_ova.predict(X_test)
accuracy_ova = accuracy_score(y_test, y_pred_ova)
print(f"OVA Accuracy: {accuracy_ova}")

# One-vs-One (OVO) SVM (implicitly handled by sklearn)
svm_ovo = svm.SVC(kernel='linear', decision_function_shape='ovo')
svm_ovo.fit(X_train, y_train)
y_pred_ovo = svm_ovo.predict(X_test)
accuracy_ovo = accuracy_score(y_test, y_pred_ovo)
print(f"OVO Accuracy: {accuracy_ovo}")

```

Notice in the above code that `decision_function_shape='ovr'` and `decision_function_shape='ovo'` are what trigger sklearn's handling of the multi-class problem using the two decomposition techniques. It's not that you're directly defining a 3d *hyperplane* for a multi-class decision boundary; you are creating combinations of 2d planes to determine which of the classes to predict.

Let's say you want a bit more control over how these different classifiers are combined. You could do something along the lines of this (although scikit learn handles it internally, this helps clarify what's actually happening under the hood):

```python
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import label_binarize

# same synthetic data from before
np.random.seed(42)
X = np.random.rand(150, 3)
y = np.random.choice([0, 1, 2], size=150)
X = X + y.reshape(-1,1) * np.random.rand(1,3)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Binarize the labels for one-vs-all
y_train_bin = label_binarize(y_train, classes=[0, 1, 2])
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])

classifiers = []
for i in range(3):
    clf = svm.SVC(kernel='linear', probability=True)
    clf.fit(X_train, y_train_bin[:, i])  # Train each classifier
    classifiers.append(clf)

# predict using probability and argmax to select the class with the highest
y_pred_proba = np.array([clf.predict_proba(X_test)[:, 1] for clf in classifiers]).T
y_pred = np.argmax(y_pred_proba, axis=1)

accuracy = accuracy_score(y_test, y_pred)
print(f"Manual OVA Accuracy: {accuracy}")
```

This shows what's happening with a bit more clarity. We train a separate binary classifier for each class using the `label_binarize` function, making the logic clear; each classifier predicts whether a data point belongs to a single given class *or not*. We then combine the results by taking the argmax (the class predicted to have the highest probability) to assign final classes. This again emphasizes that no single 3d hyperplane exists – instead it’s constructed from multiple 2d planes.

Finally, let’s consider this again conceptually. The 3 features define a 3d space. Each ova classifier would be finding a separating plane in *this* 3d space, and each one of these classifiers will contribute to the final multi-class classification by its vote. This doesn't mean you’re defining any sort of *3d* hyperplane that cleanly divides into multiple classes, which is a key point. The real "separation" is occurring within the projections defined by the individual planes, in different orientations within that 3d space.

For further technical detail, I'd highly recommend the book "The Elements of Statistical Learning" by Hastie, Tibshirani, and Friedman. It provides a rigorous theoretical background on SVMs and multi-class extensions, along with the mathematics behind it. Another strong resource would be "Pattern Recognition and Machine Learning" by Christopher Bishop which goes into depth on the theoretical underpinnings as well, focusing on probabilistic methods. Furthermore, research papers such as "A Comparison of Multiclass SVM Methods," by Hsu and Lin are valuable in understanding the different strategies employed for multi-class SVMs. I've personally found that referring back to these resources is incredibly helpful for building a robust understanding, especially when things get complex. I hope this has been useful, and let me know if you have more questions!
