---
title: "How can I get a list of losses per epoch using a FastAI Learner?"
date: "2025-01-30"
id: "how-can-i-get-a-list-of-losses"
---
The FastAI library, while elegantly streamlining deep learning workflows, doesn't directly expose a per-epoch loss list as a readily accessible attribute of the `Learner` object.  This requires a slightly more involved approach leveraging callbacks and custom functions.  My experience debugging model training issues across numerous projects has consistently highlighted the importance of meticulous loss monitoring;  a simple, easily overlooked detail can significantly impact model performance and reproducibility. Therefore, achieving granular loss tracking necessitates a proactive, rather than reactive, approach.

The key is to utilize the `Callback` mechanism within FastAI.  Callbacks are user-defined functions that allow intervention at specific points in the training process, granting precise control over data collection and manipulation.  We'll leverage a custom callback to append losses to a list within each epoch.

**1. Clear Explanation:**

The fundamental strategy involves creating a custom callback inheriting from `Callback` which overrides the `after_epoch` method.  This method is automatically executed after each epoch completes. Inside this method, we'll access the current epoch's loss from the `Learner` object and append it to a list. This list is then accessible after training is finished, providing the desired per-epoch loss data.  The critical aspect is understanding how to correctly retrieve the loss value; it is not directly a member variable of the `Learner`, but rather accessed through the `Recorder` object which maintains training history.

It's important to note that the loss value accessed through the `Recorder` represents the average loss across all batches within the given epoch. For a more fine-grained analysis, one would need to access loss from each batch individually, requiring a different callback approach and potentially increasing computational overhead.


**2. Code Examples with Commentary:**

**Example 1: Basic Loss Tracking**

This example demonstrates the simplest implementation, appending the average epoch loss to a list.


```python
from fastai.vision.all import *

class LossRecorderCallback(Callback):
    def __init__(self):
        self.losses = []
    def after_epoch(self):
        self.losses.append(self.recorder.values['loss'][-1])

# ... your data loading and model definition ...

learn = Learner(data, model, loss_func=CrossEntropyLossFlat(), metrics=[accuracy]) # Example loss function
learn.add_cb(LossRecorderCallback())
learn.fit_one_cycle(10) # train for 10 epochs

print(learn.cbs[0].losses) # Access the list of losses.  Note indexing [0] to access our specific callback
```


This code defines a custom callback `LossRecorderCallback` which initializes an empty list `losses`. The `after_epoch` method appends the last value of the loss metric tracked by the `Recorder`  (`self.recorder.values['loss'][-1]`) to the `losses` list.  Finally, after training, the loss list is printed.  Note the use of indexing `learn.cbs[0]` to access our specific callback. This is crucial because `Learner` can have multiple callbacks.


**Example 2:  Loss and Metric Tracking**

This example extends the previous one to track both the loss and a chosen metric (here, accuracy).

```python
from fastai.vision.all import *

class LossMetricRecorderCallback(Callback):
    def __init__(self):
        self.losses = []
        self.metrics = []
    def after_epoch(self):
        self.losses.append(self.recorder.values['loss'][-1])
        self.metrics.append(self.recorder.values['accuracy'][-1])

# ... your data loading and model definition ...

learn = Learner(data, model, loss_func=CrossEntropyLossFlat(), metrics=[accuracy])
learn.add_cb(LossMetricRecorderCallback())
learn.fit_one_cycle(10)

print(learn.cbs[0].losses)
print(learn.cbs[0].metrics)
```

This demonstrates the flexibility of callbacks; multiple metrics can be tracked simultaneously by adding corresponding entries to the `after_epoch` method and the callback's internal storage. This provides a richer understanding of training progress.  It assumes your learner has 'accuracy' metric added, adapt as needed based on metrics in use.


**Example 3:  Handling Multiple Metrics and Epoch-Specific Data**

In complex scenarios involving numerous metrics, a more structured approach becomes beneficial.

```python
from fastai.vision.all import *
from collections import defaultdict

class MultiMetricLossRecorderCallback(Callback):
    def __init__(self):
        self.epoch_data = defaultdict(list)
    def after_epoch(self):
        for metric_name, metric_values in self.recorder.values.items():
            self.epoch_data[metric_name].append(metric_values[-1])

# ... your data loading and model definition ...

learn = Learner(data, model, loss_func=CrossEntropyLossFlat(), metrics=[accuracy, precision, recall])
learn.add_cb(MultiMetricLossRecorderCallback())
learn.fit_one_cycle(10)

print(learn.cbs[0].epoch_data) # Access data using metric names as keys
```

This example uses a `defaultdict` to store all metrics and loss, making it scalable to an arbitrary number of metrics.  Accessing data is done via the metric names as keys, enhancing readability and maintainability.


**3. Resource Recommendations:**

The FastAI documentation, specifically the sections on callbacks and the `Learner` object, are invaluable.  A solid grasp of Python's object-oriented programming principles, especially class inheritance and method overriding, is crucial for effectively implementing custom callbacks.  Exploring examples of existing FastAI callbacks can provide insightful patterns and best practices.  Finally, a working understanding of the underlying concepts of backpropagation, loss functions, and gradient descent is essential for interpreting the obtained loss values.
