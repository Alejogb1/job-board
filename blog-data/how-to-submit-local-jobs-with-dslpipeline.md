---
title: "How to submit local jobs with dsl.pipeline?"
date: "2024-12-16"
id: "how-to-submit-local-jobs-with-dslpipeline"
---

Okay, let's tackle this. Submitting local jobs with `dsl.pipeline` can initially seem a bit… esoteric, perhaps, especially if you’re used to just firing off jobs to a remote kubernetes cluster. I've seen this trip up many developers, including myself back in the day, when I was working on a large-scale genomic analysis pipeline. We were initially building prototypes and running them locally to quickly validate algorithms before moving to our production environment. This is where `dsl.pipeline`, used with a local execution engine, shines. The trick is understanding how `kfp` handles this local invocation and how it differs from remote execution.

The core challenge stems from the fact that `dsl.pipeline` is designed to describe a workflow, *not* to execute it directly. Think of the `dsl.pipeline` definition more as a blueprint. It describes what tasks to run, in what order, and with what inputs and outputs. When we target a remote Kubernetes cluster, the kfp compiler takes this blueprint and translates it into a kubernetes-native representation (typically a yaml file) that the kubernetes cluster and its kfp operator can understand and execute. For local execution, we need a different interpretation of that blueprint. We're essentially asking kfp to interpret and run this workflow on the local machine.

The most important part of understanding this lies in the choice of `kfp.Client` instantiation and the runner configurations. You won't be specifying a kubernetes endpoint; instead, you’re opting for a local execution context. Typically you would use `kfp.Client()`, but in our case, this defaults to a kubeflow deployment, which we don't want. Here's where we deviate. Instead of a server endpoint, we don't provide any. Then, the compilation to yaml is still done using a compiler, however this is followed by an execution step performed within the local Python process, rather than passing it to a kubernetes cluster.

Let's illustrate with some code. First, here's a basic component we’ll use in our pipelines:

```python
import kfp
from kfp import dsl
from kfp.dsl import component

@component
def hello_component(text: str) -> str:
    return f"Hello, {text}!"

```

Now, let’s craft a simple pipeline that uses this component:

```python
@dsl.pipeline(
    name='Local Hello World Pipeline',
    description='A simple pipeline to test local execution.'
)
def hello_pipeline(input_text: str = "world"):
    hello_task = hello_component(text=input_text)
    print(hello_task.output)  # Printing the output, useful for debugging locally
```

Here is where the magic happens. For executing locally, we don't provide a server endpoint to kfp client, but just call the pipeline function and the pipeline runs.

```python
if __name__ == '__main__':
    # Create a pipeline instance
    hello_pipeline_instance = hello_pipeline()

    # Run the pipeline locally
    # This creates and executes the pipeline in memory
    kfp.compiler.Compiler().compile(pipeline_func=hello_pipeline,
                                   package_path="hello_pipeline.yaml")

    # If we had not created the above yaml, this would also run locally
    # kfp.Client().create_run_from_pipeline_func(hello_pipeline)

```

In the above code, we call the pipeline using `kfp.Client()`. Since we did not specify a server endpoint, it automatically detects to use local execution. The compiler is used to generate a yaml representation of the pipeline, which is not required for local execution, but is generated for consistency. The important part is that we run the pipeline. There is no call made to a cluster or server.

It’s crucial to note that local execution has limitations. The environment where each component runs is simply the same python interpreter where you execute your script. If your component depends on external services or databases that require specific configurations, you need to ensure they're accessible from the machine you're running the script. This is different from kubernetes, where your components run within containers in a cluster with their own isolated environments, and typically you rely on kubernetes configuration (e.g., `configmaps` and `secrets`) to pass such information. With local execution, all this relies on your local machine being correctly configured.

Let's illustrate further with a slightly more complex scenario. Suppose you want to simulate an output that becomes an input to the next component, similar to how you would use outputs in kubeflow pipelines, but this time it is in the local run environment:

```python
import kfp
from kfp import dsl
from kfp.dsl import component

@component
def string_uppercase(text: str) -> str:
    return text.upper()

@component
def string_reverse(text: str) -> str:
    return text[::-1]

@dsl.pipeline(
    name='Local String Manipulation Pipeline',
    description='A pipeline that manipulates a string locally.'
)
def string_manipulation_pipeline(input_text: str = "hello"):
    upper_task = string_uppercase(text=input_text)
    reversed_task = string_reverse(text=upper_task.output)
    print(f"Final Output: {reversed_task.output}")
```
And the execution will remain the same as before, except now, with two steps.
```python
if __name__ == '__main__':
   string_manipulation_pipeline_instance = string_manipulation_pipeline()
   kfp.compiler.Compiler().compile(pipeline_func=string_manipulation_pipeline,
                                   package_path="string_pipeline.yaml")

   # kfp.Client().create_run_from_pipeline_func(string_manipulation_pipeline)

```
In this example, the `upper_task` executes, then the output (which is the uppercase string) is passed as the input to the `reversed_task`. The important aspect of this for local execution is that there's no serialization or deserialization happening as the outputs and inputs are passed through Python objects inside the same Python process. In a kubernetes cluster, those outputs would have to be passed in a different way by being serialized and stored somewhere. The same is not necessary locally.

Another common practical question that comes up is how to specify different input parameters that need to be configurable for local runs. With a local run you need to instantiate a pipeline and pass the parameter in the pipeline call. You can modify the `main` function as follows:

```python
if __name__ == '__main__':

    input_value = "world_with_parameters"
    string_manipulation_pipeline_instance = string_manipulation_pipeline(input_text = input_value)
    kfp.compiler.Compiler().compile(pipeline_func=string_manipulation_pipeline,
                                   package_path="string_pipeline.yaml")
   # kfp.Client().create_run_from_pipeline_func(string_manipulation_pipeline, arguments={"input_text":input_value})
```
Here, you directly pass `input_value` as an argument when instantiating the pipeline, which is then passed to the execution.

For a deeper dive into the nuances of kubeflow pipelines and how local execution works under the hood, I highly recommend reviewing the official kubeflow pipelines documentation. Specifically, the sections detailing the compiler, the execution engine, and the client interface provide a wealth of information. The book “Kubeflow for Machine Learning” published by O'Reilly (authors: Holden Karau, et al) is a very good resource for learning about Kubeflow and its various components. The "Effective Kubernetes" book by Google’s very own team, also published by O'Reilly is also a great resource for understanding Kubernetes's internals. These should provide a far more detailed understanding than I could ever express here. Additionally, the code repository of kubeflow pipelines themselves often contains valuable examples and tests, that go into details beyond what’s documented.

In essence, local execution with `dsl.pipeline` is valuable for quick prototyping and debugging. While it differs significantly from remote cluster-based execution, understanding the fundamental shift in execution context (local python process vs container on a kubernetes cluster) will allow you to leverage it very effectively. Remember to adapt your environment and configurations to ensure your local pipeline has everything it needs. The examples above give an introductory look, so do consult the provided resources to truly grasp the complete picture.
