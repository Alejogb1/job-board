---
title: "How can I plot multiple lines with incremental logging using Weights & Biases?"
date: "2024-12-23"
id: "how-can-i-plot-multiple-lines-with-incremental-logging-using-weights--biases"
---

Alright,  I’ve certainly been down this road before – plotting multiple lines with incremental logging in Weights & Biases (wandb) can seem a bit tricky initially, but once you grasp the core concepts, it becomes quite straightforward and powerful. I recall back when we were training that large-scale transformer for a natural language task; managing those multiple loss and metric curves in a clear manner was crucial for quickly identifying bottlenecks and debugging, and frankly, wandb made it a significantly less painful process.

The key lies in understanding how wandb handles data logging and structuring your logging process accordingly. Wandb primarily logs numerical data, strings, images, and other multimedia types against a timestamp associated with each logged entry. When plotting lines, particularly multiple lines representing different metrics or values changing over time (e.g., loss and accuracy for training and validation sets, or losses associated with distinct network components), you need to provide the data in a format that wandb can effectively interpret and visualize. This means ensuring that your metric data is logged in a manner that allows wandb to differentiate between the distinct lines you intend to plot.

Basically, what we're looking at is a structured approach to how you feed data into the `wandb.log()` function. You don’t just throw in a scalar; you construct a dictionary where the keys become the series labels and the values the corresponding scalar or a list (in cases of multiple values for the same logged metric in one step). Let me illustrate this with a few examples.

First, consider a scenario where you're tracking training and validation loss. Here's how I've tackled this in the past:

```python
import wandb
import time
import random

wandb.init(project="line_plot_demo")

for step in range(100):
    train_loss = random.uniform(0.1, 2.0)
    val_loss = random.uniform(0.2, 1.5)

    wandb.log({"train_loss": train_loss, "val_loss": val_loss, "step": step})

    time.sleep(0.01)

wandb.finish()
```
In this example, within each loop iteration, I log a dictionary with two keys: `train_loss` and `val_loss`. Wandb recognizes each as a separate series, and upon plotting, you get two distinct lines. Crucially, I also include a key called "step" to be used as the x-axis to display the metrics changing over time. Without this, wandb would just plot based on the order of logging rather than any actual metric progression. I’ve seen some folks forget this 'step' parameter and wonder why their graphs were…well, messy. It’s a common oversight.

Now, let's complicate things a bit. Suppose you have multiple metrics for both training and validation—maybe, in addition to loss, you also track accuracy. In this case, you’d expand the logged dictionary accordingly:

```python
import wandb
import time
import random


wandb.init(project="complex_line_plot_demo")

for step in range(100):
    train_loss = random.uniform(0.1, 2.0)
    train_acc = random.uniform(0.6, 0.9)
    val_loss = random.uniform(0.2, 1.5)
    val_acc = random.uniform(0.5, 0.85)

    wandb.log({
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "step": step,
    })
    time.sleep(0.01)


wandb.finish()
```
Here, you end up with four distinct lines on your wandb dashboard, labeled by those dictionary keys and plotted against the step number. This approach ensures that each metric is clearly distinguishable and tracked individually throughout the training process. The x-axis is important since wandb defaults to the ‘step’ key for its plots. If you omit the step key in your log dictionary, then wandb will just use the index of logging, which might not be what you want.

It’s not always necessarily different metrics per se that you want to chart, sometimes you want to track the same metric calculated on different inputs. Imagine you have an ensemble of models and are logging a training loss calculated for each individual model in the ensemble, you can approach this by defining the unique key name for each model, like `loss_model_1`, `loss_model_2`, and so on, and then charting those in wandb.

Lastly, let's consider a scenario where you need to log a multi-dimensional metric per step. For instance, let's assume you want to log the probability distribution of the classes predicted by the model, then you can log the metric as a list of numbers within each step, where each list represents the distribution of probability over each class. For example:
```python
import wandb
import time
import random
import numpy as np


wandb.init(project="multidimensional_plot_demo")

num_classes = 5

for step in range(100):
    distribution = np.random.dirichlet(np.ones(num_classes), size=1)[0].tolist()

    wandb.log({"class_distribution": distribution, "step":step})
    time.sleep(0.01)

wandb.finish()
```
In this example, I logged the `class_distribution` as a list. Wandb understands that it needs to log each item in the list as separate plot, creating multiple lines to show how the probability distributions changed over training.

A word of caution on logging very high-dimensional distributions. If the number of dimensions is large, it will significantly increase the volume of log data, which could slow things down. It may be more appropriate to log a lower dimensional representation, like the variance, or a summary statistic instead, if detailed visualization of the multi-dimensional metric is not needed.

For more detailed information on data logging with Weights & Biases, I would suggest going through the official wandb documentation directly, specifically focusing on the "Logging Data" section; this will give you the most accurate information and show examples of other functionalities I haven’t touched here. Also, "Deep Learning with Python" by Francois Chollet has a great chapter on understanding metrics during training, which is fundamental for knowing what exactly needs to be logged and how to analyze the resulting charts. And, of course, while it's not directly focused on wandb, the seminal paper on tensorboard by Google ("TensorBoard: Visualizing Learning") provides an excellent theoretical background on how to approach logging and visualization, as many of the same principles apply.

In short, while getting started with plotting multiple lines using wandb can initially appear confusing, it's a matter of structuring the data properly and leveraging its robust logging system to visualize the data clearly. My past experiences have shown me that it’s not a magic bullet, it just requires understanding the specific data structures that wandb interprets for each line that you want to visualize. The examples provided here should give you a solid foundation to handle most scenarios and to further refine your technique for effectively visualizing your training metrics.
