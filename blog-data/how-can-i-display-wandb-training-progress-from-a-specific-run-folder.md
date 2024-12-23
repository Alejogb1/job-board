---
title: "How can I display wandb training progress from a specific run folder?"
date: "2024-12-23"
id: "how-can-i-display-wandb-training-progress-from-a-specific-run-folder"
---

Alright, let's tackle this one. I've been down this road a few times, particularly when working on large-scale distributed training jobs that require very granular monitoring. The issue of effectively displaying wandb training progress from a specific run folder isn't just about seeing metrics; it's about building a workflow that's resilient, repeatable, and, frankly, sane.

The core problem is this: wandb's primary interface, the web UI, is designed around the concept of *active* runs—those that are actively sending data. When a training run is complete, or perhaps a container crashed, and you're left with the remnants of a run folder, directly displaying that information in a live manner, as you would a live, running experiment, isn't readily available. We need to leverage wandb's api and, depending on our needs, other libraries to extract, parse, and visualize that historical data.

My initial thought always goes to the `wandb.Api()`. It’s your primary tool for programmatically accessing everything within your wandb project. This isn't about some roundabout hack; this is the intended route for such tasks. The key is to understand that even though the data is no longer "live," it's still stored as historical run data associated with your project.

First, I would reach for `wandb.Api()`. Then, I usually follow with fetching the specific run object, followed by extracting its history. This "history" object within the run contains the data you need: the loss, accuracy, or whatever metrics your model was tracking. Here is an illustrative python example:

```python
import wandb
import pandas as pd

def fetch_and_display_history(project_name, run_id):
    """
    Fetches and displays the history of a specific wandb run.

    Args:
        project_name (str): The name of the wandb project.
        run_id (str): The ID of the wandb run.
    """

    api = wandb.Api()

    try:
        run = api.run(f"{project_name}/{run_id}")
    except wandb.errors.CommError as e:
        print(f"Error fetching run {run_id}: {e}")
        return

    if not run:
        print(f"Run with id {run_id} not found in project {project_name}.")
        return

    history_df = run.history()

    if history_df.empty:
        print(f"No history available for run {run_id}.")
        return

    print(f"Displaying history data for run {run_id}:\n")
    print(history_df)


if __name__ == '__main__':
    # Replace with your actual project and run information
    project = "your_wandb_username/your_project_name"  # Example: "my_user/my_project"
    run_identifier = "some_run_id"  # Example: "abcdef12"
    fetch_and_display_history(project, run_identifier)
```

In the above code, replace `your_wandb_username/your_project_name` with the actual path to your wandb project, and replace `some_run_id` with the identifier of the specific run whose data you are seeking. The result is a pandas dataframe which makes the data easily consumable.

However, the output of this code is just a printout of the dataframe which, while informative, may not be the most ideal format for visually assessing trends during training. To visualize the data, you may consider using libraries such as matplotlib or seaborn. Here’s how we could modify the code to plot the data:

```python
import wandb
import pandas as pd
import matplotlib.pyplot as plt

def fetch_and_plot_history(project_name, run_id, metric_to_plot="loss"):
    """
    Fetches, processes, and plots the history of a specific wandb run.

    Args:
        project_name (str): The name of the wandb project.
        run_id (str): The ID of the wandb run.
        metric_to_plot (str): The metric to plot, defaulting to 'loss'.
    """

    api = wandb.Api()

    try:
       run = api.run(f"{project_name}/{run_id}")
    except wandb.errors.CommError as e:
       print(f"Error fetching run {run_id}: {e}")
       return

    if not run:
        print(f"Run with id {run_id} not found in project {project_name}.")
        return

    history_df = run.history()
    if history_df.empty:
        print(f"No history available for run {run_id}.")
        return
    if metric_to_plot not in history_df.columns:
        print(f"Metric '{metric_to_plot}' not found in run history.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(history_df[metric_to_plot], label=metric_to_plot)
    plt.title(f"Training {metric_to_plot} over time (Run ID: {run_id})")
    plt.xlabel("Step")
    plt.ylabel(metric_to_plot)
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    # Replace with your actual project and run information
    project = "your_wandb_username/your_project_name"  # Example: "my_user/my_project"
    run_identifier = "some_run_id" # Example: "abcdef12"
    metric = "loss"
    fetch_and_plot_history(project, run_identifier, metric) # plots loss data
    metric = "accuracy"
    fetch_and_plot_history(project, run_identifier, metric) # plots accuracy data
```
This improved code now visualizes the training metrics and allows you to quickly visualize specific metrics. Be mindful to adjust the `metric_to_plot` argument to represent metrics that your model actually logs. You can plot multiple metrics by calling the method multiple times with different metrics as shown.

Now, you may encounter situations where your data is not directly in a column but has nested data, as wandb sometimes structures it. To handle this, we’ll delve a little deeper into how we can extract such information. Let’s take an example where you logged the value of 'some_metric' inside a dictionary called 'batch_metrics', along with its epoch. Here’s how we could retrieve and display that:

```python
import wandb
import pandas as pd

def fetch_and_extract_nested_history(project_name, run_id):
    """
    Fetches and extracts history with nested metric data from a specific wandb run.

    Args:
        project_name (str): The name of the wandb project.
        run_id (str): The ID of the wandb run.
    """
    api = wandb.Api()

    try:
        run = api.run(f"{project_name}/{run_id}")
    except wandb.errors.CommError as e:
        print(f"Error fetching run {run_id}: {e}")
        return

    if not run:
        print(f"Run with id {run_id} not found in project {project_name}.")
        return

    history_df = run.history()
    if history_df.empty:
         print(f"No history available for run {run_id}.")
         return

    extracted_data = []
    for index, row in history_df.iterrows():
        try:
            batch_metrics = row['batch_metrics']
            if isinstance(batch_metrics, dict) and 'some_metric' in batch_metrics:
               extracted_data.append({
                   'step': index,
                   'epoch': row.get('epoch', -1), #use get to gracefully handle cases when not in run history
                   'some_metric': batch_metrics['some_metric']
                  })
        except KeyError:
            continue #handle case where key not available.

    extracted_df = pd.DataFrame(extracted_data)

    if extracted_df.empty:
        print(f"No nested 'some_metric' data found within 'batch_metrics'.")
        return
    print(f"Displaying nested metric data for run {run_id}:\n")
    print(extracted_df)


if __name__ == '__main__':
    # Replace with your actual project and run information
    project = "your_wandb_username/your_project_name"  # Example: "my_user/my_project"
    run_identifier = "some_run_id" # Example: "abcdef12"
    fetch_and_extract_nested_history(project, run_identifier)
```

This revised snippet now iterates through the `history_df`, looks for the nested 'batch_metrics' key, safely checks its type, and then attempts to extract the `some_metric` field along with its associated epoch. If it fails, the error is handled gracefully, logging an informative message, and continuing with the next row. You can then adapt this method to extract other nested keys as necessary.

For further understanding and deeper dives into these concepts, I’d strongly recommend consulting *“Effective Python: 90 Specific Ways to Write Better Python”* by Brett Slatkin for practical techniques around python development, and the official wandb documentation, specifically the section on the api. Furthermore, diving into the pandas documentation is crucial to take full advantage of its capabilities in data manipulation. You may also find academic research papers on distributed training systems in the field of machine learning useful in understanding the broader context of training at scale.

In my experience, mastering these programmatic methods for retrieving and displaying wandb training data is essential. It moves you beyond relying purely on the interactive UI and gives you far greater control and flexibility when dealing with historical or offline analysis. It's not just about displaying data—it’s about building robust and reusable workflows that streamline your entire research process.
