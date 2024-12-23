---
title: "How do I change the display name for a run in AzureML?"
date: "2024-12-23"
id: "how-do-i-change-the-display-name-for-a-run-in-azureml"
---

Let's talk about display names in Azure Machine Learning runs. I've had my share of late nights debugging pipelines where the generic run ids staring back at me felt more like enemies than allies. Changing a run’s display name isn’t a direct property you can modify post-execution. Rather, it's primarily a design consideration implemented during the submission process of a new run. The process is not immediately obvious if you’re accustomed to modifying things on the fly, so let me detail the mechanics behind it and illustrate with some concrete examples based on my experiences.

The core principle is that you specify a display name within the configuration settings *before* the run is initiated. This name becomes the more human-readable identifier that appears within the Azure Machine Learning studio and in other monitoring tools. It's not something you can just reach into and alter once the run is complete. Think of it as a metadata tag that's etched into the run's genesis.

Now, let's break down how to achieve this, referencing my past experiences:

**The Key Mechanism: The `RunConfiguration` Object**

When initiating an Azure ML run through the SDK, whether it's a script run or an experiment run, the `RunConfiguration` object is your central control point for all configuration settings. Within this object is the crucial `display_name` property, which, if specified, is what the run will be labeled with in the Azure ML Studio. Neglecting this means you're left with the auto-generated id, often a confusing string of alphanumeric characters.

In the field, I often saw issues arising because teams initially relied on default run naming which is quite unhelpful when you're tracing back specific model experiments. We often ended up implementing a systematic naming convention using a run configuration strategy rather than using the default approach. We built tooling to generate display names using placeholders that included parameters or the training dataset being used. This allowed us to swiftly identify and compare runs, saving us significant debugging time.

**Practical Implementation (Code Examples):**

Let me show you how you would do this in practice. Here are three examples demonstrating various approaches.

**Example 1: Basic Display Name:**

This is the most straightforward example. Here we are just setting a basic name on the run configuration. This is useful for quick runs or for simple experiments.

```python
from azureml.core import Workspace, Experiment, RunConfiguration, Environment
from azureml.core.script_run_config import ScriptRunConfig

# Retrieve your workspace
ws = Workspace.from_config()

# Define your experiment
experiment = Experiment(workspace=ws, name="my-experiment")

# Create a run configuration object
run_config = RunConfiguration()

# Define an environment (e.g., create one or use an existing curated environment)
env = Environment.get(workspace=ws, name="AzureML-Minimal")
run_config.environment = env

# Set the display name
run_config.display_name = "My Simple Training Run"

# Configure script run
src = ScriptRunConfig(source_directory=".", script="train.py", run_config=run_config)


# Submit the run
run = experiment.submit(config=src)
run.wait_for_completion(show_output=True)

```

In this example, after executing the script, you’ll see the run listed in the Azure ML studio with the name "My Simple Training Run" instead of a generated identifier. I've found starting with something this simple is helpful when you're working on quick proofs of concept.

**Example 2: Dynamic Display Names with Experiment Parameters:**

This example illustrates how to make the display name more informative by including experiment parameters. Here we're pulling specific values to help identify the purpose of the run.

```python
from azureml.core import Workspace, Experiment, RunConfiguration, Environment
from azureml.core.script_run_config import ScriptRunConfig

# Retrieve your workspace
ws = Workspace.from_config()

# Define your experiment
experiment = Experiment(workspace=ws, name="parameterized-experiment")


# Create a run configuration object
run_config = RunConfiguration()

# Define an environment
env = Environment.get(workspace=ws, name="AzureML-Minimal")
run_config.environment = env

# Define Parameters
param1 = "lr0.01"
param2 = "100epochs"

# Set the display name dynamically using parameters
run_config.display_name = f"ParamTest-{param1}-{param2}"

#Configure script run
src = ScriptRunConfig(source_directory=".", script="train.py", run_config=run_config)


# Submit the run
run = experiment.submit(config=src)
run.wait_for_completion(show_output=True)
```

Here, the run name would be something like 'ParamTest-lr0.01-100epochs', making it much easier to identify at a glance which settings were used for the run. This significantly cut down the time my team spent trying to remember what hyperparameters went into different runs during past projects.

**Example 3: Using a Function for Dynamic Naming:**

For more complex scenarios, you might need more sophisticated logic. Here, we use a function to generate the display name. This can be useful when incorporating model types or data configurations into the run display name.

```python
from azureml.core import Workspace, Experiment, RunConfiguration, Environment
from azureml.core.script_run_config import ScriptRunConfig
import datetime

def generate_run_name():
    """Generates a dynamic run name using current time"""
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    model_type = "CNN"
    return f"run-{model_type}-{timestamp}"


# Retrieve your workspace
ws = Workspace.from_config()

# Define your experiment
experiment = Experiment(workspace=ws, name="complex-experiment")

# Create a run configuration object
run_config = RunConfiguration()

# Define an environment
env = Environment.get(workspace=ws, name="AzureML-Minimal")
run_config.environment = env

# Dynamically set the display name using the function
run_config.display_name = generate_run_name()


#Configure script run
src = ScriptRunConfig(source_directory=".", script="train.py", run_config=run_config)

# Submit the run
run = experiment.submit(config=src)
run.wait_for_completion(show_output=True)

```

This example generates a name that includes the model type and a timestamp, allowing for better tracking of runs over time. This kind of dynamic naming is useful when there are a number of different experiments being conducted concurrently by a team.

**Key Takeaways and Recommendations:**

*   **Plan Ahead:** Display names should be part of your experiment management strategy from the beginning. Retroactively applying a name change isn't possible; you must plan upfront.
*   **Consistency is Key:** Adopt a standardized naming convention that works for your team. This makes it easier to interpret and compare runs.
*   **Be Descriptive:** The display name should give you sufficient information to quickly understand the nature of the run, without having to inspect the underlying details. This often involves adding key hyperparameters or identifying the data set.
*   **Utilize Functions:** If you need more complex display naming, define functions to programmatically generate names.
*   **Avoid Static Names**: Consider adding timestamps or model parameters to avoid overwriting runs in the console and to keep the list organized.

**Further Reading:**

For deeper understanding of these concepts, I recommend the following resources:

*   **"Programming Machine Learning: From Data to Applications" by Paolo Ferragina and Francesco Bonchi:** This provides a solid foundation for understanding machine learning pipeline orchestration, including concepts related to experimental tracking that influence Azure ML best practices.
*   **Azure Machine Learning documentation:** Look directly at Microsoft's official documentation on `RunConfiguration`, and `ScriptRunConfig`, since these objects have direct influence on what you're dealing with here. Specifically, examine the sections related to experiment configuration, run submission, and logging.
*   **"Machine Learning Engineering" by Andriy Burkov:** This is excellent for establishing best practices for managing machine learning workflows.

While modifying a run's display name after the fact might seem desirable, the architecture of Azure Machine Learning encourages meticulous planning and clear documentation *before* the run begins. It's not a limitation, but a methodology that encourages good machine learning experiment management practices. Based on my experience, implementing these best practices will save you many headaches down the line.
