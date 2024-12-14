---
title: "How to Find how a variable affects the output of a time-series random-forest regression model?"
date: "2024-12-14"
id: "how-to-find-how-a-variable-affects-the-output-of-a-time-series-random-forest-regression-model"
---

hey there,

so, you're looking to pin down how individual variables are impacting the output of your random forest model when dealing with time series data. i've been there, trust me. it's not always a straightforward process, but it's definitely doable. random forests are these fantastic workhorses for regression, but their inner workings can feel like a black box sometimes. especially when you’ve got time dependencies thrown into the mix.

let's break this down. we are dealing with a time series random forest, which inherently makes things more interesting than regular rf regression. one important thing to remember is we can’t shuffle the data when working with timeseries data. the time-ordering of the data is of the essence to the model’s performance, because of autocorrelation. let's begin.

**feature importance: a starting point, but it's not the whole story**

most of us jump straight to feature importance scores that rf models conveniently provide, and that’s a good place to start. in python, if you are using scikit-learn, this would be: `model.feature_importances_`. these numbers are calculated based on how much a feature, on average, reduces the model’s impurity over all decision trees. a higher value suggests a more significant contribution to the final prediction accuracy, but it doesn't really tell you how that contribution *behaves*. it is an aggregate metric. you can use it, but consider it a first approximation. the feature could be important just because it is correlated with other features which are the actual cause of changes in output.

in my experience, feature importances can be a bit misleading in time series scenarios. a feature might score high simply because it's correlated with other features or with the output because of the time series aspect. or it might have a lot of variance or many values in training data that caused it to be chosen often in the tree splitting procedure, not necessarily because of its causal impact. a non-causal high correlation due to time series may lead to over-reliance on this variable. the real kicker is that it doesn't explain the *direction* of the impact either. does an increase in this variable lead to an increase or decrease in the output? the feature importance is agnostic to this information. it only tells you how “important” it is in making predictions. also, it’s usually not enough to know which variable *is* important, you want to know *how* it is important, meaning what is its effect on the final predicted value. for this, you need to go further than vanilla feature importances.

**permutation importance: a more robust take**

a better option, in my opinion, is permutation importance. instead of relying on model internals, permutation importance works by shuffling the values of one feature at a time and recomputing the model's prediction score. if the score drops considerably after the shuffling process, it implies that that feature is important in making accurate predictions. the rationale behind it is that you are removing information that the model was using by randomly re-ordering the values.

here's how it looks in python, also using scikit-learn:

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

# assuming your timeseries data is in a pandas dataframe called 'data'
# and the target variable is in a column called 'target'
# create some dummy data
np.random.seed(42)
data = pd.DataFrame({
    'feature1': np.random.rand(100),
    'feature2': np.random.rand(100) * 10,
    'feature3': np.sin(np.linspace(0, 10, 100)),
    'target': np.random.rand(100)
})
data['target'] = data['feature1']* 2 + data['feature2']*0.5 + data['feature3'] + np.random.rand(100)/10

X = data[['feature1', 'feature2', 'feature3']]
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
permutation_importances = pd.Series(result.importances_mean, index=X.columns)
print(permutation_importances)
```
in this snippet we first create some dummy data, in a real life example you would load your actual data. the main thing is to pay attention to `train_test_split(..., shuffle=False)` which is crucial for timeseries. this permutation importance will tell you how much the model's predictive performance suffers when the information of a specific feature is scrambled. it’s a better metric to use in timeseries, but it still doesn’t give the direction of the effect of the variable, so we need more investigation.

**partial dependence plots (pdp): showing the directional impact**

okay, now we're getting into the interesting part. feature importance tells us *which* features matter, but pdp tells us *how* they matter. partial dependence plots (pdp) show how the model's predicted output changes as we vary a specific feature, while keeping all other features constant (averaged out in a manner), in a timeseries setting, this becomes a very useful tool, if you do the proper procedures to generate it.

the principle of this plot is to show the average change in output when changing a given variable in the data domain and keeping all other variables constant (but in reality, averaged out to a specific value), this gives you a sense of the direction of impact and the magnitude. there are some limitations of these graphs, like not showing the interactions between variables, but it provides a more detailed insight than just feature importances.

here's how you'd whip up a pdp in python using scikit-learn:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import train_test_split

# assuming your timeseries data is in a pandas dataframe called 'data'
# and the target variable is in a column called 'target'
# create some dummy data
np.random.seed(42)
data = pd.DataFrame({
    'feature1': np.random.rand(100),
    'feature2': np.random.rand(100) * 10,
    'feature3': np.sin(np.linspace(0, 10, 100)),
    'target': np.random.rand(100)
})
data['target'] = data['feature1']* 2 + data['feature2']*0.5 + data['feature3'] + np.random.rand(100)/10

X = data[['feature1', 'feature2', 'feature3']]
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

fig, ax = plt.subplots(figsize=(8, 6))
display = PartialDependenceDisplay.from_estimator(model, X_test, features=['feature1'], ax=ax, kind='average')
plt.show()
```
the `PartialDependenceDisplay.from_estimator` function does all the heavy lifting for you, plotting the change in the output prediction based on the feature in question. by default the argument `kind='average'` will perform averaging across all samples to produce a single line on the graph. if you change it to `kind='both'` you would see also the individual partial dependence for each sample.

a nice thing about pdps is that you can visualize not only linear effects, but also non-linear effects, and also non-monotonic effects too (effects that go up and down within the range of the variable), something that is hard to see with simple statistical correlations.

**individual conditional expectation (ice) plots: even more granularity**

now, if you want to go a step further and inspect the effect of variables on individual data instances, you can use ice plots. they are like pdps, but instead of showing an average effect they show the effect on each observation individually. this could be very useful to see if a variable has different effects depending on some properties of the samples. maybe a feature is very important for some periods of time and not that much for others. these plots can display these kinds of effects.

here's the python code:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import train_test_split

# assuming your timeseries data is in a pandas dataframe called 'data'
# and the target variable is in a column called 'target'
# create some dummy data
np.random.seed(42)
data = pd.DataFrame({
    'feature1': np.random.rand(100),
    'feature2': np.random.rand(100) * 10,
    'feature3': np.sin(np.linspace(0, 10, 100)),
    'target': np.random.rand(100)
})
data['target'] = data['feature1']* 2 + data['feature2']*0.5 + data['feature3'] + np.random.rand(100)/10

X = data[['feature1', 'feature2', 'feature3']]
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

fig, ax = plt.subplots(figsize=(8, 6))
display = PartialDependenceDisplay.from_estimator(model, X_test, features=['feature1'], ax=ax, kind='individual')
plt.show()

```
the only change from the pdp code is the `kind='individual'` which plots the prediction changes of each sample individually. it may get hard to understand if you have too many samples, but on the other hand, you have the full picture on how the model is using a given variable. it is up to you to choose which representation is best for your needs.

**a note on time dependencies and interactions**

one more thing to keep in mind is that when we’re dealing with time series, it’s not just about individual variables but also about their lagged values, and interactions with other features. you might need to create features that explicitly capture the time dynamics, for example, by using shifted versions of your original variables.

for example, you could include a feature that lags feature1 by one time step:

```python
data['feature1_lagged_1'] = data['feature1'].shift(1)
```

you can create higher lags or more complex aggregations as you need it. remember that the correlation in time-series data can be high and a lagged version of a variable might be more important to the prediction than the variable itself, even when the variable is the driver of the effect, just due to the time-series dependencies.

after doing all this it might be useful to read papers on causal inference, like "elements of causal inference: foundations and learning algorithms" by jonas peters, dominik janizing and bernhard schölkopf to get a deeper understanding of the causal relationships between variables and to prevent misinterpretations from pure statistical dependencies.

and, as a final joke, what did the random forest say to the decision tree? "stop branching out so much, it's making me dizzy!"

anyways, hope this helps with your quest to understand your random forest model. good luck!
