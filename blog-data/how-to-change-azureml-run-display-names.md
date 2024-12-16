---
title: "How to change AzureML run display names?"
date: "2024-12-16"
id: "how-to-change-azureml-run-display-names"
---

Okay, let’s tackle this. I’ve personally stumbled upon this particular nuance of Azure Machine Learning more times than I care to remember, and it’s a common frustration point for many practitioners. You're facing the issue of AzureML run names not being as informative as you'd like them to be in the portal, often defaulting to generic ids. It's not a deal-breaker, but it certainly impacts workflow efficiency, especially when trying to sift through multiple experiments.

Essentially, AzureML assigns default names to runs which are typically auto-generated identifiers, not human-friendly. These names are created when you initialize an `azureml.core.Run` object via `experiment.start_logging()` or similar functions. Fortunately, there's a straightforward way to override these defaults and establish more descriptive names. The crucial element here is the `display_name` property of the `Run` object, which is set *before* you submit your run. This can be achieved through the `start_logging()` method’s `display_name` parameter or by setting it directly on the created run object. Let’s explore the practical aspects with some code examples.

Let's start with a scenario I encountered while training a convolutional neural network for image classification. I was running multiple experiments with varying hyperparameters. Initially, my run dashboard was a jumble of cryptic run ids; not particularly conducive to effective analysis. I needed a better way to distinguish between experiments based on, say, the learning rate and batch size.

**Example 1: Setting Display Name During Run Initiation**

The first, and arguably cleanest approach, is to set the display name while initiating the run using the `start_logging()` method. Here's how I'd typically structure this approach:

```python
from azureml.core import Workspace, Experiment, Run
from azureml.core.runconfig import RunConfiguration
from azureml.core.conda_dependencies import CondaDependencies

# Load workspace config (ensure you have the config.json file in your working directory)
ws = Workspace.from_config()
experiment_name = 'cnn-training'
experiment = Experiment(workspace=ws, name=experiment_name)

# Create run configuration
run_config = RunConfiguration()
run_config.environment.python.conda_dependencies = CondaDependencies.create(conda_packages=['scikit-learn','tensorflow','pandas']) #Example packages

learning_rate = 0.001
batch_size = 32
display_name_str = f"lr_{learning_rate}_bs_{batch_size}"

# Start the run, providing the display_name
run = experiment.start_logging(run_config=run_config, display_name=display_name_str)

# Simulate the training code (replace with your actual training logic)
import time
time.sleep(30)
# Log some metrics to show the process is working
run.log('learning_rate',learning_rate)
run.log('batch_size', batch_size)

run.complete()

print(f"Run '{run.id}' completed with display name: {display_name_str}")
```

In this code, I craft a `display_name_str` incorporating relevant experimental parameters, resulting in a more informative label in the AzureML portal. This approach is especially useful when the parameter values are known *before* you initiate the run. The run id remains a unique identifier but, the display name is significantly more helpful for navigation.

Now, let’s consider a slightly different circumstance. Imagine you have a complex hyperparameter tuning loop where you don’t know the optimal parameters in advance but derive them during the experiment. You might use a hyperparameter search method like random or Bayesian search within your training script. In such cases, we'd adjust the display name *after* the optimal parameters are found, during the run itself.

**Example 2: Setting Display Name Mid-Run**

Here’s a scenario where you might set the display name during runtime:

```python
from azureml.core import Workspace, Experiment, Run
from azureml.core.runconfig import RunConfiguration
from azureml.core.conda_dependencies import CondaDependencies

# Load workspace config (ensure you have the config.json file in your working directory)
ws = Workspace.from_config()
experiment_name = 'hyperparam-tuning'
experiment = Experiment(workspace=ws, name=experiment_name)

# Create run configuration
run_config = RunConfiguration()
run_config.environment.python.conda_dependencies = CondaDependencies.create(conda_packages=['scikit-learn','tensorflow','pandas']) #Example packages

# Start the run
run = experiment.start_logging(run_config=run_config)

# Placeholder for an hyperparameter tuning loop
best_learning_rate = 0.002
best_batch_size = 64
# Simulate a finding of optimal values (replace with your hyperparameter search logic)
import time
time.sleep(30)

# Set the display name after determining optimal values
display_name_str = f"best_lr_{best_learning_rate}_bs_{best_batch_size}"
run.display_name = display_name_str
run.update() # update() is important for runtime changes to take effect.

# Log parameters after they are selected.
run.log('learning_rate',best_learning_rate)
run.log('batch_size',best_batch_size)

run.complete()

print(f"Run '{run.id}' completed with display name: {display_name_str}")

```

In this second snippet, the display name is dynamically generated once the “best” hyperparameters are obtained, which is common in more complicated training routines. I used `run.display_name = ...` followed by `run.update()` to ensure that this name change is reflected in AzureML. This is vital because, unlike the first example where we specified the name at creation, we are now modifying it mid-run. It’s a critical step often missed, causing confusion.

Finally, there are cases where I found myself retrospectively needing to change a bunch of run display names. Suppose I neglected to set them during the initial experiments. There is an alternative: we can iterate through already completed runs to update display names based on some logical criteria. Let's use metrics to rename a few runs from a particular experiment.

**Example 3: Updating Names for Existing Runs**

Here’s the last scenario, updating previously completed runs:

```python
from azureml.core import Workspace, Experiment, Run

# Load workspace config (ensure you have the config.json file in your working directory)
ws = Workspace.from_config()
experiment_name = 'hyperparam-tuning'
experiment = Experiment(workspace=ws, name=experiment_name)

# Fetch all runs within the experiment, regardless of completion status.
runs = experiment.get_runs()

for run in runs:
    if run.status == "Completed":
        try:
            # get metric of 'learning_rate' and 'batch_size'
            learning_rate = run.get_metrics()['learning_rate']
            batch_size = run.get_metrics()['batch_size']

            display_name_str = f"post_update_lr_{learning_rate}_bs_{batch_size}"
            run.display_name = display_name_str
            run.update()
            print(f"Run {run.id} updated to {display_name_str}")

        except KeyError:
            print(f"Run {run.id} did not contain 'learning_rate' or 'batch_size' metrics. Skipped.")

        except Exception as e:
            print(f"Error updating run {run.id}: {e}")
```
In this example, I loop through existing runs, check if they’re completed, and, if so, try to read metrics, construct a new display name and then update it. It adds an extra layer of complexity but provides a practical method to retrospectively fix the naming if needed. It’s not ideal to be doing this as standard practice, but it's a practical solution when things haven’t gone as planned. It also emphasizes the need to use try/except blocks when working with existing runs as you might face cases of missing metrics.

For a deeper understanding, I’d recommend referring to the official Azure Machine Learning documentation, particularly the sections on `azureml.core.Run` and `azureml.core.Experiment`. Specifically, reading the class definitions for `azureml.core.Run` and `azureml.core.Experiment` will clarify the available properties and functions. Additionally, the "Machine Learning with Python" by Tarek A. El-Gaaly is a comprehensive guide that would provide more context about how to work with this AzureML.

These methods represent the core of how I handle run naming. Remember, consistency is key, so adopting a systematic approach to naming your runs will greatly improve the readability and maintainability of your machine learning experiments. You might also consider documenting your naming conventions for your team's use, further standardizing and improving overall workflow.
