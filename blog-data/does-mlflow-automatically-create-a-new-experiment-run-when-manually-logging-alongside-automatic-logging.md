---
title: "Does MLflow automatically create a new experiment run when manually logging alongside automatic logging?"
date: "2024-12-23"
id: "does-mlflow-automatically-create-a-new-experiment-run-when-manually-logging-alongside-automatic-logging"
---

, let’s tackle this one. The intricacies of mlflow experiment runs, especially when mixing manual and automatic logging, is something I’ve definitely spent a fair amount of time debugging over the years. It’s a very common point of confusion, and it’s absolutely necessary to understand how mlflow handles this to maintain organized experiments. So, to the point: does mlflow automatically create a new experiment run when manually logging alongside automatic logging? The answer isn’t a straightforward ‘yes’ or ‘no’ – it’s conditional, and it depends on the exact context of your code. Let's dissect it.

The core of the matter revolves around mlflow’s concept of an 'active run'. Mlflow keeps track of an active run, typically initialized through `mlflow.start_run()`, or implicitly through certain automatic logging mechanisms. If you are using an auto-logging library, it may begin a run in the background, which makes the situation trickier. When you manually log parameters, metrics, or artifacts, these are associated with the *currently active run*. If no active run is present, mlflow typically attempts to start a new one implicitly, usually based on any set configurations such as experiment name, or if none is provided a default one is initialized, or in some cases, throws an error.

However, and this is the crucial part, if you have an active run (started manually or automatically) and then initiate an automatic logging function, *it will typically attach that automatic logging to the existing run*. This means that no new run is automatically created; everything gets logged under the same run id. This is particularly relevant when using auto-logging functions provided by libraries like scikit-learn or tensorflow, which can get tangled with manually logged parameters.

Let's make this concrete with a few code snippets based on scenarios I've encountered.

**Scenario 1: Manual Run, then Automatic Logging - No New Run**

Let’s imagine you’re explicitly starting a run, then using scikit-learn's `mlflow.sklearn.autolog` function:

```python
import mlflow
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Create sample data
np.random.seed(42)
X = np.random.rand(100, 5)
y = 2 * X[:, 0] + 3 * X[:, 2] + np.random.randn(100)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run() as run:
    mlflow.log_param("seed", 42)

    # Auto-log Scikit-learn models
    mlflow.sklearn.autolog()

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)

    mlflow.log_metric("test_mse", mse)
    print(f"Run ID: {run.info.run_id}")
```

In this scenario, the `mlflow.start_run()` explicitly creates a run. Then, when `mlflow.sklearn.autolog()` is called and the model is trained, everything gets logged under *that same run ID*, and no new run is automatically started. This is a key thing to understand – auto logging generally doesn’t introduce new runs once you have one active. We've manually logged a parameter, used autolog and also logged a metric, but they all belong to one and only one run.

**Scenario 2: Automatic Logging First, Manual Logging Afterward - Still One Run**

Let’s consider the reverse case: auto-logging starts the run implicitly, and then you log more manually. This also results in everything belonging to one run. This scenario happens frequently when dealing with frameworks like TensorFlow or Pytorch where auto-logging can be activated without explicit `start_run()`. Here, I will use a dummy model with TensorFlow and auto-log it using `mlflow.tensorflow.autolog()`.

```python
import mlflow
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

# Generate some data
np.random.seed(42)
X = np.random.rand(100, 5)
y = 2 * X[:, 0] + 3 * X[:, 2] + np.random.randn(100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


mlflow.tensorflow.autolog()

# Define a simple model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')


# Train the model (this triggers auto logging)
model.fit(X_train, y_train, epochs=10, verbose=0)


with mlflow.active_run() as run:
    mlflow.log_param("model_type", "sequential")
    print(f"Run ID: {run.info.run_id}")
    predictions = model.predict(X_test)
    loss = model.evaluate(X_test, y_test, verbose=0)
    mlflow.log_metric("test_loss", loss)
```
In this instance, the `mlflow.tensorflow.autolog()` function starts a background run, the model training logs metrics, then we manually log a parameter under the same active run and a metric. So, you still have a single run, and all information is neatly packaged under that run. Again, if an active run exists, auto logging attaches to that instead of starting a new one.

**Scenario 3: No Active Run, Then Manual Logging, Then Auto Logging - Still one Run**

Finally, let's illustrate a scenario where we start with no active run explicitly, and then manually log before triggering autolog. This usually starts a new run, and then automatically logged information is added to that same run.

```python
import mlflow
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# Generate some data
np.random.seed(42)
X = np.random.rand(100, 5)
y = np.random.randint(0, 2, 100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


mlflow.log_param("preprocessing_type", "random")

mlflow.sklearn.autolog()

model = LogisticRegression()
model.fit(X_train, y_train)

with mlflow.active_run() as run:
    print(f"Run ID: {run.info.run_id}")
```

Here, we manually log a parameter *before* any run has been explicitly started. Then, `mlflow.sklearn.autolog()` sees no current active run and will implicitly start one.  The manual parameter logging goes into this run. All subsequent automatic logging by scikit-learn also falls under this single run. If, we try to log parameter before the autolog call *and* without starting a run, mlflow will often auto start a run for us.

In summary, the core principle is that if an active run exists, any logging, whether manual or automatic, will generally attach to it. The question of a new run primarily boils down to whether an active run was present when automatic logging is triggered. It also becomes a question of whether you attempt to log without a started run at all (either explicitly or implicitly). If no active run is present when you try to log something, it will often start a new run for you.

Now, for resources to dive deeper, I'd highly recommend the official mlflow documentation, especially the sections on experiment tracking and auto-logging. It’s always a reliable starting point. For a more formal treatment of experiment management and MLOps in general, I also found that the book "Building Machine Learning Pipelines" by Hannes Hapke and Catherine Nelson, provides solid theoretical insights which helped me put the whole picture together. You could also take a look at 'Introducing MLOps: How to Scale Machine Learning in the Real World' by Mark Treveil, et al. These resources offer a blend of conceptual understanding and practical advice.

I hope that clarifies it. Feel free to ask if any follow-up questions pop up.
