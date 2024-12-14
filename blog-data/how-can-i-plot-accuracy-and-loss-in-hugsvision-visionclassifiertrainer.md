---
title: "How can i plot accuracy and loss in hugsvision (VisionClassifierTrainer)?"
date: "2024-12-14"
id: "how-can-i-plot-accuracy-and-loss-in-hugsvision-visionclassifiertrainer"
---

alright, so you’re looking to visualize your training process in hugsvision, specifically the accuracy and loss curves when using `VisionClassifierTrainer`. it's a common need and a crucial step to understanding how your model's learning. i’ve definitely been there, staring at a wall of numbers and wishing for a nice plot. been through this particular struggle myself more times than i care to remember. let me break down how i usually tackle it.

the `VisionClassifierTrainer` doesn't, out of the box, give you plots directly like some libraries. it's more of an engine than a dashboard. the good news is, it does a great job at logging the metrics to things like tensorboard or wandb. if you are not using these, then it stores the data internally which we can access. so you gotta grab these values after training, and then use something like matplotlib to get your visuals.

i'll show you some code snippets and talk through the process. let's assume you've already run your training, and your `VisionClassifierTrainer` object is named `trainer`. the first thing to know is that the history of training is stored in the trainer's state attribute under `log_history`.

here's a simple example of how to extract the training and validation data:

```python
import matplotlib.pyplot as plt

def plot_training_metrics(trainer):
    history = trainer.state.log_history
    train_losses = [entry["loss"] for entry in history if "loss" in entry]
    val_losses = [entry["eval_loss"] for entry in history if "eval_loss" in entry]
    train_accuracies = [entry["accuracy"] for entry in history if "accuracy" in entry]
    val_accuracies = [entry["eval_accuracy"] for entry in history if "eval_accuracy" in entry]
    epochs = range(1, len(train_losses) + 1) #assuming each entry corresponds to an epoch


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.plot(epochs, train_losses, 'b', label='Training loss')
    ax1.plot(epochs, val_losses, 'r', label='Validation loss')
    ax1.set_title('Training and validation loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(epochs, train_accuracies, 'b', label='Training accuracy')
    ax2.plot(epochs, val_accuracies, 'r', label='Validation accuracy')
    ax2.set_title('Training and validation accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    plt.show()


# Assuming your trainer object is named 'trainer'
plot_training_metrics(trainer)
```

what this does is grab the loss and accuracy from each log entry, and separate training from validation, then plots them with matplotlib. this assumes your `log_history` dictionary contains "loss","eval_loss", "accuracy", and "eval_accuracy" keys. which they should if you use `compute_metrics` in the `trainer`. if you are using other metrics, you need to adjust the keys according to their names in the log history. also important, remember that these logged metrics vary depending on the trainer used.

let's say your trainer doesn't have a built-in "accuracy" you are using `f1-score` in `compute_metrics` but it's not directly logged by that name. you might be calculating it, but not using the standard 'accuracy' label, this happened to me once i was calculating the area under the curve or auc as well as f1. in this case, you’d need to adjust the code. here’s how you can tailor the metrics you're extracting:

```python
import matplotlib.pyplot as plt

def plot_custom_metrics(trainer, train_metric_key, val_metric_key, title, ylabel):
  history = trainer.state.log_history
  train_metrics = [entry[train_metric_key] for entry in history if train_metric_key in entry]
  val_metrics = [entry[val_metric_key] for entry in history if val_metric_key in entry]
  epochs = range(1, len(train_metrics) + 1)

  plt.plot(epochs, train_metrics, 'b', label='Training ' + title)
  plt.plot(epochs, val_metrics, 'r', label='Validation ' + title)
  plt.title('Training and validation ' + title)
  plt.xlabel('Epochs')
  plt.ylabel(ylabel)
  plt.legend()
  plt.show()

# assuming that in the trainer metric computation you use a 'f1' key.
# for train f1, and eval_f1 for validation f1
plot_custom_metrics(trainer, 'f1', 'eval_f1', 'F1 Score', 'F1 Score')
```

this version takes arguments for the specific metric keys you want to use and lets you plot any two metrics. it makes things a bit more flexible. remember, always print your logs to see what are the correct metrics keys.

now, let’s get a bit more detailed. sometimes you might be doing more than one training pass, like using k-fold cross-validation. in that case, you need to keep track of the training data across all the folds and then plot the averaged result to better understand the model's stability. i have done some experiments with that in my early days when i was building an image classifier for medical scans, and i had to get the right parameters to make it work reliably. this gets more complicated, since we need to keep the track of the metrics for every fold. the next code snippets goes into this direction:

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_kfold_metrics(kfold_histories, metric_key, eval_metric_key, title, ylabel):
  num_folds = len(kfold_histories)
  max_epochs = max(len(history) for history in kfold_histories)

  train_metrics = np.zeros((num_folds, max_epochs))
  val_metrics = np.zeros((num_folds, max_epochs))

  for fold_idx, history in enumerate(kfold_histories):
    for epoch_idx, entry in enumerate(history):
        if metric_key in entry:
          train_metrics[fold_idx, epoch_idx] = entry[metric_key]
        if eval_metric_key in entry:
          val_metrics[fold_idx, epoch_idx] = entry[eval_metric_key]

  mean_train_metrics = np.nanmean(train_metrics, axis=0)
  mean_val_metrics = np.nanmean(val_metrics, axis=0)
  epochs = range(1, len(mean_train_metrics) + 1)
  plt.plot(epochs, mean_train_metrics, 'b', label=f'Average Training {title}')
  plt.plot(epochs, mean_val_metrics, 'r', label=f'Average Validation {title}')
  plt.title(f'Average Training and Validation {title} (Across Folds)')
  plt.xlabel('Epochs')
  plt.ylabel(ylabel)
  plt.legend()
  plt.show()


# Assume you have a list of trainer histories after k-fold cross-validation
# where each item of the list are trainer.state.log_history
# we assume that each log history has the same keys.
# if not, the code would need to be improved.
#
# example of using the function would be:
# plot_kfold_metrics(kfold_histories, 'accuracy', 'eval_accuracy', 'Accuracy', 'Accuracy')


```
this function will go through the logs of each k-fold history, extract the metrics of every epoch, and then average them out, to have a plot of the average behaviour across all the folds. a little note of caution here, if different folds have different number of epochs. this function will pad the shorter ones with nan values. if you are using a different number of training epochs between folds, this can be a problem, but you can adjust the logic to match your data.

a final thought. it’s always worth diving a bit deeper into the process. while `matplotlib` is fantastic for quick plots, for something more complex, consider using `seaborn`, especially for visualizing distributions of metrics. when you start to fine-tune hyperparameters, having better ways to visualize the results will save you time and headaches. always spend the time upfront to get your plots working right and you will save a lot of trouble when the model is not working as expected, you will avoid wasting time going back to the code and instead focus on the model itself. remember that a good plot can say more than many lines of code. and don't forget, when in doubt, read the docs.

for deeper dives into data visualization, "python data science handbook" by jake vanderplas is a fantastic resource. it's one of those books i always come back to. for practical deep learning i’d also recommend “deep learning with python” by francois chollet. not directly linked to the visualization stuff but they are important books for your journey in ml.

and if this still seems like a lot, well, remember, debugging neural networks is like trying to find a specific grain of sand on a beach, sometimes a plot can help you find it!
