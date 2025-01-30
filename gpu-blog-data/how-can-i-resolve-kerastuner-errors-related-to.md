---
title: "How can I resolve KerasTuner errors related to missing positional arguments when tuning ANN models?"
date: "2025-01-30"
id: "how-can-i-resolve-kerastuner-errors-related-to"
---
KerasTuner, while simplifying hyperparameter optimization, often throws frustrating `TypeError` exceptions during model definition when a positional argument is missing in the builder function, typically during a search process. I’ve encountered this precise issue numerous times, frequently with custom layers or when attempting to inject configurable elements into the model graph based on sampled hyperparameters. The core problem stems from KerasTuner’s interaction with the `build_model` function (or its equivalent) you provide – specifically, it expects a function that exclusively accepts a single argument, a `keras_tuner.HyperParameters` object. Any deviation from this structure leads to errors during the tuning process.

The `build_model` function I provide to `keras_tuner.Tuner` is a function signature that must consistently process the `hp` object. The `hp` object holds the values for sampled hyperparameters during the tuner search. A common error, and the one that triggers the positional argument error, is when my original model-building function was initially designed to take other parameters that are not hyperparameterized and thus not within the scope of the `hp` object. This leads to unexpected conflicts with the tuner's attempts to provide hyperparameter values via the `hp` object. The positional errors occur because the `Tuner` does not automatically supply non-hyperparameter arguments to the builder function. I need to architect my build function solely around the provided `HyperParameters` object.

Let's illustrate this with a couple of code examples I've encountered in past projects, including a working solution.

**Example 1: Incorrectly Structured Builder Function**

Initially, my model building function was similar to this for experimentation:

```python
import tensorflow as tf
from tensorflow import keras
from keras_tuner import HyperModel, RandomSearch
from keras_tuner import HyperParameters

def build_model_incorrect(input_shape, num_classes, dropout_rate):
    inputs = keras.layers.Input(shape=input_shape)
    x = keras.layers.Dense(128, activation='relu')(inputs)
    x = keras.layers.Dropout(dropout_rate)(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    return keras.Model(inputs=inputs, outputs=outputs)


class MyHyperModel(HyperModel):
  def __init__(self, input_shape, num_classes):
    self.input_shape = input_shape
    self.num_classes = num_classes

  def build(self, hp):
    dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)
    return build_model_incorrect(self.input_shape, self.num_classes, dropout_rate)



input_shape = (10,)
num_classes = 5

hypermodel = MyHyperModel(input_shape=input_shape, num_classes=num_classes)

tuner = RandomSearch(
    hypermodel,
    objective='val_accuracy',
    max_trials=5,
    directory='my_dir',
    project_name='my_project'
)
tuner.search(tf.random.normal(shape=(100, 10)), tf.random.uniform(shape=(100,), minval=0, maxval=5, dtype=tf.int64), epochs=2)
```

Here, the `build_model_incorrect` function accepts `input_shape`, `num_classes`, and `dropout_rate` as positional arguments. My `HyperModel` class attempts to pass these directly to the build model. However, KerasTuner only invokes the `build` method with a single `hp` argument. The error will surface during `tuner.search()`, reporting a `TypeError` stating that `build_model_incorrect` expects three positional arguments but receives only one, the `HyperParameters` object. This setup erroneously attempts to provide those arguments from the class scope when instead they need to be made available during the build process and configured through hyperparameters. My mistake was in mixing data configuration and hyperparameterization in the signature of the underlying model-building function.

**Example 2: Still Incorrect, but with a Step Toward the Solution**

I then tried to refactor the build process by including `input_shape` and `num_classes` as hyperparameters, thinking this would fix it.

```python
import tensorflow as tf
from tensorflow import keras
from keras_tuner import HyperModel, RandomSearch
from keras_tuner import HyperParameters


def build_model_incorrect2(hp):
    input_shape = hp.Int('input_shape', min_value=1, max_value=20)
    num_classes = hp.Int('num_classes', min_value=2, max_value=10)
    dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)
    inputs = keras.layers.Input(shape=(input_shape,))
    x = keras.layers.Dense(128, activation='relu')(inputs)
    x = keras.layers.Dropout(dropout_rate)(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    return keras.Model(inputs=inputs, outputs=outputs)



class MyHyperModel2(HyperModel):
  def __init__(self):
      pass
  def build(self, hp):
      return build_model_incorrect2(hp)


hypermodel = MyHyperModel2()

tuner = RandomSearch(
    hypermodel,
    objective='val_accuracy',
    max_trials=5,
    directory='my_dir',
    project_name='my_project'
)
tuner.search(tf.random.normal(shape=(100, 10)), tf.random.uniform(shape=(100,), minval=0, maxval=5, dtype=tf.int64), epochs=2)
```
Although the `build` method no longer produces an argument mismatch, the issue is not fully resolved. The shape parameter is set to an integer within the hyperparameter range. When the model attempts to create an input layer, it fails, because an integer passed to the Input layer as shape requires a tuple, so `(input_shape,)` fixes that. However, `input_shape` and `num_classes` are typically configuration constants of the dataset, and not hyperparameters that we should be tuning. Additionally, since this is now a randomly sampled input shape each trial, the random tensors provided to search will fail during training since the training tensors are not matched to the shape. This is a fundamental problem of how data and parameters are considered when designing a build function.

**Example 3: Corrected Builder Function and HyperModel**

The working approach is to structure the `build_model` function to *only* use the `hp` argument for hyperparameter values, and to externalize the fixed data parameters using class-level storage.

```python
import tensorflow as tf
from tensorflow import keras
from keras_tuner import HyperModel, RandomSearch
from keras_tuner import HyperParameters


def build_model_correct(hp, input_shape, num_classes):
    dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)
    inputs = keras.layers.Input(shape=input_shape)
    x = keras.layers.Dense(128, activation='relu')(inputs)
    x = keras.layers.Dropout(dropout_rate)(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    return keras.Model(inputs=inputs, outputs=outputs)

class MyHyperModel3(HyperModel):
  def __init__(self, input_shape, num_classes):
    self.input_shape = input_shape
    self.num_classes = num_classes
  def build(self, hp):
    return build_model_correct(hp, self.input_shape, self.num_classes)


input_shape = (10,)
num_classes = 5

hypermodel = MyHyperModel3(input_shape=input_shape, num_classes=num_classes)

tuner = RandomSearch(
    hypermodel,
    objective='val_accuracy',
    max_trials=5,
    directory='my_dir',
    project_name='my_project'
)
tuner.search(tf.random.normal(shape=(100, 10)), tf.random.uniform(shape=(100,), minval=0, maxval=5, dtype=tf.int64), epochs=2)

```

Here, `build_model_correct` now also accepts the constant `input_shape` and `num_classes`. The `MyHyperModel3` class initializes these constants and passes them to `build_model_correct`. Crucially, the `build` method passes `hp` as the only argument it receives to the `build_model_correct`. This follows the proper approach to parameterization that separates the `hp` object for hyperparameter space, and class storage for data-derived configuration. The `build` method uses *both* the `hp` object for tuning purposes and the class storage for fixed constants. This resolves the positional argument error, allowing KerasTuner to correctly initialize and tune the model, using the `hp` object as intended.

In summary, the primary resolution strategy involves refactoring your model-building function to accept only the `HyperParameters` object (usually named `hp`) and using class level storage for fixed parameters. These stored parameters are then available during the call to your model build function when a new model is being created by the KerasTuner. Additionally, I must be very clear about which parameters I am considering hyperparameters, so I don't misconfigure my model creation process. I've found this to be the most robust and consistent approach to resolving these types of errors with KerasTuner in practice, regardless of my model's complexity.

For further learning, I recommend reviewing documentation specifically for `keras_tuner.HyperModel`, `keras_tuner.HyperParameters`, and the different tuner classes such as `RandomSearch` and `BayesianOptimization`. Understanding how KerasTuner interacts with your provided builder function is essential. Additionally, reviewing examples and tutorials that highlight proper usage of `HyperModel` can solidify these concepts. Focusing on the proper architecture of your `build` method, and understanding how KerasTuner invokes it, will eliminate positional argument errors during hyperparameter optimization.
