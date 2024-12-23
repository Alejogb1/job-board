---
title: "How do I change the run display name in azureml?"
date: "2024-12-23"
id: "how-do-i-change-the-run-display-name-in-azureml"
---

Alright,  Changing the display name of an azureml run – it's a common requirement, and I remember dealing with this myself quite a few times, particularly when trying to keep experiments well organized. Over the years, I’ve found the standard documentation can sometimes be a tad too generic, especially when you’re after something specific like this. So, here’s my approach, reflecting lessons learned through real-world scenarios.

The challenge fundamentally arises from the way azureml initially names your runs; it often generates something rather generic and less than descriptive. You're usually staring at a run id that’s difficult to distinguish from another when browsing through your experiment history. The good news is that you *can* modify this, albeit with a few nuances to be mindful of.

The key lies in the `Run` object and its `display_name` property. This property is mutable, allowing you to update it dynamically during or even after the run, subject to certain conditions. Now, this ‘after the run’ bit is something I’ve had to leverage more times than I care to recall, particularly when the initial run parameters don’t allow for an informative name, or when post-processing reveals important insights that I want to highlight in the display name.

Let's dive into some practical examples using the azureml sdk for python.

**Scenario 1: Setting the Display Name at Run Start**

This is the cleanest method. The best practice I've found over time is to set the display name as soon as you initialize your run. This makes debugging and tracking more intuitive from the get-go. Here's how you can do it:

```python
from azureml.core import Workspace, Experiment, Run
from azureml.core.runconfig import RunConfiguration

# Load workspace (assuming configuration is already set)
ws = Workspace.from_config()
experiment_name = 'my-experiment-name'
experiment = Experiment(workspace=ws, name=experiment_name)

# Prepare run configuration (if needed) - use a simple configuration for this example
run_config = RunConfiguration()
run_config.environment.python.conda_dependencies.add_pip_package("scikit-learn") # adding a simple dependency example

# Setting the display_name *before* the run starts
display_name = "Training Run: Model v2, Adam Optimizer, 100 Epochs"

run = experiment.start_logging(run_config=run_config, display_name=display_name)
print(f"Run ID: {run.id}, Display Name: {run.display_name}")

# Your training code would go here. For illustration purposes, lets simulate training.
import time
for i in range(1,6):
    time.sleep(1)
    run.log('iteration',i)

run.complete()

```
In this snippet, I'm using the `experiment.start_logging()` method and pass in the `display_name` parameter directly. Note that I did not specify a script to execute. If you were using the `experiment.submit()` method instead, you would pass the run_config and also the `display_name` parameter directly at the time of submission. This is the recommended approach since your experiment run name will be meaningful throughout its lifecycle.

**Scenario 2: Updating the Display Name *During* a Run**

Sometimes, you might want to change the name based on information that becomes available during the run itself, like the best-performing parameter set or the overall training accuracy. This is also very achievable.

```python
from azureml.core import Workspace, Experiment, Run
from azureml.core.runconfig import RunConfiguration
import random

# Load workspace (assuming configuration is already set)
ws = Workspace.from_config()
experiment_name = 'my-experiment-name'
experiment = Experiment(workspace=ws, name=experiment_name)

# Prepare run configuration (if needed) - use a simple configuration for this example
run_config = RunConfiguration()
run_config.environment.python.conda_dependencies.add_pip_package("numpy") # adding a simple dependency example

# Start the run with an initial display name.
run = experiment.start_logging(run_config=run_config, display_name="Initial Run")
print(f"Initial Run ID: {run.id}, Display Name: {run.display_name}")

# Simulate a training loop and some calculations
best_accuracy = 0
for i in range(1,10):
    accuracy = random.random()
    run.log('iteration',i)
    run.log('accuracy',accuracy)
    if accuracy > best_accuracy:
      best_accuracy = accuracy

# Update display_name *during* the run, now knowing best accuracy
new_display_name = f"Run: Best Accuracy={best_accuracy:.2f}"
run.display_name = new_display_name
run.log('final_display_name',run.display_name) # Log the updated name if required
print(f"Updated Run ID: {run.id}, Updated Display Name: {run.display_name}")

run.complete()
```
Here, the `display_name` is updated by simply assigning a new string to `run.display_name` once we know the best accuracy. It’s important to note that this update takes effect relatively quickly in the azureml user interface and the name change will be visible in the portal.

**Scenario 3: Updating the Display Name *After* a Run is Complete**

There have been situations where I've needed to modify run names *after* the fact. This is especially true when you are reviewing old experiments and realize their names are not descriptive enough. To update a run display name after it has completed, you retrieve the run via the `experiment.get_run` method and then update the `display_name`.

```python
from azureml.core import Workspace, Experiment, Run
from azureml.core.runconfig import RunConfiguration
import time
# Load workspace (assuming configuration is already set)
ws = Workspace.from_config()
experiment_name = 'my-experiment-name'
experiment = Experiment(workspace=ws, name=experiment_name)

# Start the run with a default display name
run = experiment.start_logging(display_name="Default Display Name")
print(f"Initial Run ID: {run.id}, Initial Display Name: {run.display_name}")

# Simulate training
time.sleep(2)
run.complete()

# retrieve the finished run
run_id = run.id
retrieved_run = Run(experiment=experiment, run_id=run_id)


# Update the name
new_display_name = "Updated Display Name Post-Run"
retrieved_run.display_name = new_display_name
print(f"Updated Run ID: {retrieved_run.id}, Updated Display Name: {retrieved_run.display_name}")
```
In this final example, I first complete a run with an initial name. Then, I demonstrate how you can retrieve a run by its `id` using `experiment.get_run()` (or `Run(experiment=experiment,run_id=run_id)` as shown), and then change the `display_name`. Again, note this will update in the azureml portal within a few seconds. The ability to change run names after completion is especially useful for retrospectively correcting any missteps in your initial naming conventions.

**Important Notes and Considerations**

*   **Timing:** As highlighted in the examples, the most efficient way to make the naming process manageable is to try to define a run's display name as close to the start of the run as possible, or even better - *before* the start of the run. This promotes a cleaner, more organized workflow. However, the flexibility to modify the `display_name` at any stage is valuable.
*   **Consistency:** If working within a team, agree on a naming convention beforehand to avoid a hotchpotch of run names. The goal here is improved clarity and organization and not a free for all.
*   **Script Submission:** When using `experiment.submit`, remember the `display_name` attribute is a parameter of `ScriptRunConfig`, and should be supplied at submission time to ensure a descriptive name from the start.
*   **Azure Portal:** The changes you make to display names will be reflected in the Azure portal but might not appear instantaneously. Be patient; the updates will propagate.

**Recommended Resources**

*   **Microsoft Azure Machine Learning Documentation:** The official documentation remains your most comprehensive source of truth. Specifically, explore the sections on the `Run` class and experiment management.
*   **"Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow" by Aurélien Géron:** While not directly azureml-focused, the concepts around model training, experiment tracking, and the importance of well-organized projects are foundational to using azureml effectively.

In closing, managing run display names effectively in azureml might seem like a small detail, but it’s crucial for maintaining project clarity and efficient collaboration. The ability to dynamically modify these names based on your specific needs, whether before, during, or after a run, is one of the many aspects that make azureml a flexible and powerful tool. Keep experimenting and keep learning – that's the best approach.
