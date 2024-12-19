---
title: "How to build a SHAP summary plot after creating an ML model by GridSearchCV in Python?"
date: "2024-12-15"
id: "how-to-build-a-shap-summary-plot-after-creating-an-ml-model-by-gridsearchcv-in-python"
---

alright, so you're banging your head against the wall trying to get a shap summary plot after using gridsearchcv, right? i've been there, more times than i care to remember. it’s like, you finally get your model trained, feel a sense of victory, and then… shap throws a curveball. no worries, we’ll sort it out.

i recall one particularly grueling project back when i was still a fresh-faced grad student (okay, maybe not so fresh-faced, but definitely less grizzled). it was a classification problem, predicting some obscure scientific thing-a-ma-jig, i don't really remember, using like 100 features, and of course the professor insisted on using grid search because, well, that's what professors do. i had finally gotten the damn grid search to finish and give me the best estimator. i was feeling pretty good about myself, ready for a beer, and then the moment of truth to see which features were actually important and bam! shap blew up on me! it was returning me some weird object, not the plot i was expecting. let me just say i spent an entire night and most of the next day debugging. i almost went crazy. it turned out the problem wasn't with shap itself (mostly) but rather how the best_estimator from gridsearchcv is a pipeline, and shap needs the actual fitted model.

the main issue is that `gridsearchcv`’s `best_estimator_` attribute returns a whole pipeline object, not the trained model directly, and shap needs a direct model object. it needs that actual model, not the pipeline that might have preprocessing steps, transformers, etc. shap doesn't know what to do with all those extra layers. it just needs the core model that actually makes predictions.

so first off, let's ensure you have shap and scikit-learn installed. if you haven't, do a quick

```bash
pip install shap scikit-learn
```

now, let's tackle the plotting. after your grid search, you've got something that looks like this i guess:

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
import pandas as pd

# making dummy data cause we don't have yours
X, y = make_classification(n_samples=100, n_features=20, random_state=42)
X_df = pd.DataFrame(X)
# simple pipeline
pipe = Pipeline([('scaler', StandardScaler()),('model', RandomForestClassifier(random_state=42))])
# hyper parameters
param_grid = {'model__n_estimators': [100, 200],
              'model__max_depth': [5, 10]}
# gridsearch
grid = GridSearchCV(pipe, param_grid, cv=3)
grid.fit(X, y)

```

now the key is to extract the model directly. after this, you will have the fitted model ready. and then the shap magic. so the next code is crucial.

```python
import shap
# this is how you extract the model from the gridsearch
model = grid.best_estimator_.named_steps['model']

# create a shap explainer
explainer = shap.TreeExplainer(model)

# get shap values. you will need a background dataset
# for tree models it's best to use the training data itself
shap_values = explainer.shap_values(X_df)

# generate the summary plot
shap.summary_plot(shap_values, X_df, plot_type="bar")

```
in the code above i extracted the model named 'model' from the pipeline step. if your pipeline is named differently, make sure to use the correct name. also, note that for tree-based models, it’s common practice to use your training data as the background for the explainer. this is what i am doing with `X_df`.

for non-tree models, like support vector machines or neural networks, the approach is slightly different. you would typically use a `shap.kernelExplainer` instead, and the background data selection becomes more important. you could use a subset of your data as the background if you have a lot of training points, or create a synthetic background if the model behaves unexpectedly.

here’s an example using a svm, with a smaller dataset as background:

```python
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd
import shap
from sklearn.datasets import make_classification
import numpy as np
# create data
X, y = make_classification(n_samples=200, n_features=10, random_state=42)
X_df = pd.DataFrame(X)
# create pipeline
pipe_svm = Pipeline([('scaler', StandardScaler()),('model', SVC(random_state=42))])
# hyper parameters
param_grid_svm = {'model__C': [1, 10], 'model__gamma': [0.1, 1]}
# grid search
grid_svm = GridSearchCV(pipe_svm, param_grid_svm, cv=3)
grid_svm.fit(X, y)

# get model
model_svm = grid_svm.best_estimator_.named_steps['model']

# background dataset
background = shap.sample(X_df, 20)

# explainer
explainer_svm = shap.KernelExplainer(model_svm.predict, background)

# compute shap values
shap_values_svm = explainer_svm.shap_values(X_df)

# plot summary
shap.summary_plot(shap_values_svm, X_df, plot_type="bar")

```
the key here is again to extract the model using `grid_svm.best_estimator_.named_steps['model']`, and to correctly setup the `shap.KernelExplainer` with a background dataset.

one time i was trying to explain some model, and i kept getting garbage outputs. after much frustration and about 3 cups of coffee, i realized my features were all scaled super differently. a simple `standardscaler` in the pipeline fixed that. it just goes to show you sometimes it's not the library, it's just you being dumb, or not so dumb, but just overlooked an obvious detail. i've learned that the debugging process is a humbling experience.

now, a few things i'd recommend for a deeper understanding of shap and model interpretability in general: start with the original shap paper, "a unified approach to interpreting model predictions" by scott lundberg and su-in lee. this is the bible for shap. you should also check out "interpretable machine learning" by christoph molnar, it is available online. it covers a broader range of methods but has a great section on shap. i would also suggest the blog of christoph molnar, it has great posts and use cases of these methods. these resources should help you really understand how shap works, and how to get the best results with it.

so, to recap, the steps to get your shap summary plot after gridsearch are:
1. train your model using gridsearchcv.
2. extract the fitted model from the pipeline (with best_estimator.named_steps['model']).
3. create the appropriate shap explainer (tree or kernel).
4. compute shap values.
5. create your summary plot!

it’s a bit fiddly, but once you understand the pipeline/model separation, it becomes much easier. and hey, at least you're not using matlab... haha!.
i hope that helps and let me know if you have other issues, i'm here to help.
