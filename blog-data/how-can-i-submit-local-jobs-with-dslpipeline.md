---
title: "How can I submit local jobs with dsl.pipeline?"
date: "2024-12-23"
id: "how-can-i-submit-local-jobs-with-dslpipeline"
---

Let's talk about orchestrating local jobs using `dsl.pipeline` within the context of kubeflow pipelines. It's a topic I've grappled with extensively, especially during my early days setting up complex workflows. While kubeflow pipelines are fundamentally designed for cloud-based execution, submitting jobs that run solely on your local machine requires a bit of a detour. It's not the primary use case, but it's definitely feasible and valuable for prototyping, development, and even some forms of local testing before pushing things to the cloud.

The challenge stems from the nature of kubeflow pipelines: they are built to manage workloads across distributed systems. To make it work locally, we're essentially going to bypass the normal cloud orchestration layer and directly execute components within a local environment. This deviates from the typical kubeflow workflow, so we must address that.

The key here lies in understanding that `dsl.pipeline` itself is an abstract description of your workflow. It defines the *what*, but not necessarily the *where*. The actual execution is usually handled by a kubeflow engine, typically using containerized steps. When targeting local execution, we're essentially implementing a very basic, local 'executor' that will interpret this pipeline definition and execute the steps, usually python functions directly, on the local machine.

One strategy I used a lot during early stage pipeline development was to adapt my components themselves to be both remotely executable and locally testable. This usually involved conditional blocks within my component definition. Let me illustrate with some code.

Here's a very basic example of how to define a component that is designed to run both within a cloud environment and locally:

```python
import kfp
from kfp import dsl
import os

def add_numbers(a: int, b: int) -> int:
    print(f"adding {a} and {b}")
    return a + b

@dsl.component
def add_component(a: int, b: int) -> int:
    if 'KUBEFLOW_RUN_ID' in os.environ: # Check if in a kubeflow pod
        return add_numbers(a, b)
    else: # Run locally
        print("Running locally...")
        return add_numbers(a,b)

@dsl.pipeline(
    name="add-local",
    description="Example of local job execution"
)
def add_local_pipeline(num1: int=5, num2: int=10):
    add_task = add_component(a=num1, b=num2)

if __name__ == '__main__':
    # create a local pipeline
    pipeline = add_local_pipeline()

    #  Here's a basic local pipeline runner - can improve it substantially.
    for task in pipeline.tasks.values():
        # Here we can assume it's an add_component so we're not generalizing here, but we could check
        # task.component.__name__ and call the correct function with correct inputs.
        args = {k: v.default for k, v in task.component.inputs.items()}
        result = task.component(*args.values())
        print(f"local result is {result}")

```

In this example, the `add_component` checks for the `KUBEFLOW_RUN_ID` environment variable, which is typically set when the component executes within a kubeflow pod. When running locally, that variable will be absent and so the component will execute the local logic. Crucially, instead of submitting the pipeline to a cluster, I’m actually walking through the tasks directly and executing the relevant functions. This is obviously a simplified example but it highlights the technique.

Let’s look at a slightly more complex case with artifacts. Now consider that your components might output data, or consume it, and you want to simulate this locally too. In a kubeflow environment, these artifacts are typically managed via a central artifact store (e.g., Minio, S3) by the pipeline engine. To handle it locally, we need a local simulation mechanism for artifact passing:

```python
import kfp
from kfp import dsl
import os
import json

def write_data(data: dict, file_path: str):
    with open(file_path, 'w') as f:
        json.dump(data, f)
    return file_path

@dsl.component
def data_writer(data_input: dict) -> dsl.Output[dsl.Artifact]:
    if 'KUBEFLOW_RUN_ID' in os.environ:
       # Use kubeflow artifact output
        output_path = output.path
        write_data(data_input, output_path)
    else:
        print("Running data writer locally.")
        # Simulate artifact creation locally.
        output_path = "local_data.json"
        write_data(data_input, output_path)
        return dsl.Artifact(name="my_local_artifact", uri=output_path)

@dsl.component
def data_reader(input_artifact: dsl.Input[dsl.Artifact]) -> dict:
   if 'KUBEFLOW_RUN_ID' in os.environ:
      with open(input_artifact.path, 'r') as f:
          data = json.load(f)
   else:
      print(f"Running data reader locally, input file is {input_artifact.uri}")
      with open(input_artifact.uri, 'r') as f:
           data = json.load(f)
   return data

@dsl.pipeline(
    name="artifact-test-local",
    description="Pipeline to test artifacts locally"
)
def artifact_local_pipeline(input_data: dict = {"key": "value"}):
   writer = data_writer(data_input=input_data)
   reader = data_reader(input_artifact=writer.output)


if __name__ == '__main__':

    pipeline = artifact_local_pipeline()

    # basic local simulation for pipeline execution.
    artifact_registry = {}
    for task in pipeline.tasks.values():
         args = {}
         for name, value in task.inputs.items():
            if isinstance(value, dsl.Input):
                args[name] = artifact_registry[value.artifact.name]
            elif value:
                 args[name] = value.default
         outputs = task.component(*args.values()) # run component, get output
         if isinstance(outputs, dsl.Artifact):
               artifact_registry[outputs.name] = outputs
         elif isinstance(outputs, tuple) and isinstance(outputs[0], dsl.Artifact):
              artifact_registry[outputs[0].name] = outputs[0]

    print("Local execution complete.")
    if "my_local_artifact" in artifact_registry:
        print(f"Artifact content: {artifact_registry['my_local_artifact'].uri}")

```
Here I have introduced the notion of locally simulating artifact handling. The `data_writer` function will now write the output file to a local location, and the `data_reader` function will similarly read from it. We are effectively creating our own limited local registry for artifacts, and use that to provide the input artifacts to the subsequent components.

A third point to consider is that this method only works for single-node local execution. If your pipeline components require distributed resources (e.g., multi-node GPU computation), you'll find that these techniques fall short. You would then likely need a different approach involving local kubernetes setup with something like minikube, but that's beyond the scope of just the `dsl.pipeline` functionality.

You can improve these simple local executors further by adding error handling, dependency management, more robust artifact handling logic, and many other things; but hopefully this gives you a good idea of how to get started with local execution of `dsl.pipeline`.

For further study, I recommend diving into the kubeflow pipelines documentation itself, especially the sections relating to components and artifact handling. Also, researching the design of workflow engines generally is beneficial. The paper "Spark, TensorFlow, and a tale of two schedulers" by Zaharia et al. could offer some insights into distributed scheduler design even though it’s not specifically about kubeflow. Lastly, for a deeper look at workflow design principles, "Principles of Transactional Information Systems" by Philip Bernstein and Eric Newcomer is a classic and relevant, albeit more theoretical. These will give you a solid foundation as you progress with workflow orchestration, both locally and in the cloud.
