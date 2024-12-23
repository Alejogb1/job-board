---
title: "How do I submit local jobs with dsl.pipeline?"
date: "2024-12-23"
id: "how-do-i-submit-local-jobs-with-dslpipeline"
---

Okay, let’s unpack this. Submitting local jobs using `dsl.pipeline` in the context of, say, kubeflow pipelines, is a topic I've had to navigate more than a few times. It sounds straightforward – and conceptually it is – but the devil, as they say, is in the details, especially when you start dealing with persistent volumes, resource limitations, and network configurations in a simulated local environment. I recall one project, codenamed 'Project Chimera', where we had to completely overhaul our local pipeline testing strategy. Before we get into the practical code, let's establish what we're trying to do. We are effectively aiming to emulate a full-fledged kubernetes cluster’s job execution environment on our local machine, which the kubeflow pipelines sdk is set up to ultimately target. This often involves running mini-kubernetes environments like minikube, kind, or docker desktop's kubernetes support, depending on your specific needs.

Now, the key element here is that `dsl.pipeline` defines the structure of your pipeline – the component dependencies and data flow – but it doesn't inherently handle *local* execution in the way you might imagine if you're coming from traditional scripting. It primarily operates in the context of the kubeflow pipelines runtime environment, which generally resides in a kubernetes cluster. So, how do we bridge this gap? We will have to rely on helper functions and libraries that create a kubeflow client and then use it to submit the pipeline run. Specifically, instead of deploying your pipeline directly to a kubernetes cluster, we'll configure the client to submit the pipeline run using a local kubeflow instance you've likely provisioned using one of those local k8s environments I previously mentioned.

The first, and likely the most common scenario, is using the `kfp` client library directly to submit a pipeline that has been compiled to a json or yaml file. We use the function `create_run_from_pipeline_func` or `create_run_from_pipeline_package` for this purpose, which will handle creating a new run given the specified pipeline.

Here’s a simple code snippet illustrating this:

```python
import kfp
from kfp import dsl
from kfp.compiler import compiler
from kfp.client import Client

@dsl.component
def hello_world(text: str) -> str:
    print(f"Hello, {text}")
    return f"Processed: {text}"

@dsl.pipeline(name='Local Pipeline Example')
def my_pipeline(input_text: str = 'World'):
    hello_task = hello_world(text=input_text)

if __name__ == '__main__':
    pipeline_func = my_pipeline
    pipeline_name = pipeline_func.__name__ + '.yaml'
    compiler.Compiler().compile(pipeline_func, package_path=pipeline_name)
    client = Client(host="http://localhost:8080")
    run = client.create_run_from_pipeline_package(
       pipeline_name,
       arguments={"input_text": "local testing" }
       )
    print(f"Pipeline submitted with run id: {run.run_id}")
```

In this example, I’m defining a basic pipeline using `@dsl.pipeline`, which consists of one component, `hello_world`. The `if __name__ == '__main__':` block is where the real action happens. Firstly, the pipeline is compiled into a yaml file via the compiler and secondly we create an instance of `kfp.client.Client` pointing to our local kubeflow dashboard. Finally, `create_run_from_pipeline_package` takes this compiled pipeline and submits it as a run to our local kubeflow instance. You'll notice I’ve also passed an argument `input_text`, demonstrating how we can parameterize pipeline inputs even in a local setting. Keep in mind that you need a functioning kubeflow installation running at `http://localhost:8080` (or whatever your endpoint is), for this code to work.

Another scenario arises when you're iterating and testing frequently. Compiling and submitting a file each time can become cumbersome. Instead, you can directly submit the pipeline *function* by using `create_run_from_pipeline_func` instead of compiling to a file first. This is often more convenient for development. Here’s how that looks:

```python
import kfp
from kfp import dsl
from kfp.client import Client

@dsl.component
def simple_component(message: str) -> str:
    print(f"Component says: {message}")
    return f"Received: {message}"

@dsl.pipeline(name='Direct Function Submission')
def direct_submit_pipeline(input_message: str = "Hello!"):
    task_1 = simple_component(message=input_message)

if __name__ == '__main__':
    client = Client(host="http://localhost:8080")
    run = client.create_run_from_pipeline_func(
        direct_submit_pipeline,
        arguments={'input_message': "Direct submission"}
    )
    print(f"Pipeline submitted with run id: {run.run_id}")
```

Here, instead of compiling first, we pass the pipeline function `direct_submit_pipeline` directly to `client.create_run_from_pipeline_func`. The outcome is identical in terms of running the pipeline on your local kubeflow setup. This is especially useful during active development where you're making frequent changes to the pipeline’s definition, as it shortens the feedback loop.

A third, and sometimes more complex, scenario occurs when you are trying to debug specific components or require finer control over resource allocation. In these cases, it is often necessary to create components with an image that runs directly in the local environment, rather than trying to use the system docker installation and having potential conflicts with resource consumption, libraries, or packages. You can achieve this through the use of a base image with all requirements, and build an image that runs locally.

```python
import kfp
from kfp import dsl
from kfp.client import Client

@dsl.component(base_image="python:3.9")
def local_component(message: str) -> str:
    import time
    print(f"Starting local component with message: {message}")
    time.sleep(5)
    print(f"Local component finished with message: {message}")
    return f"Done: {message}"

@dsl.pipeline(name='Local Component Test')
def local_pipeline(input_message: str = "Test Message"):
    local_task = local_component(message=input_message)

if __name__ == '__main__':
    client = Client(host="http://localhost:8080")
    run = client.create_run_from_pipeline_func(
        local_pipeline,
        arguments={'input_message': "Local test run"}
    )
    print(f"Pipeline submitted with run id: {run.run_id}")
```
In this instance, setting a base image using the `base_image` parameter allows the underlying system to run the component with the specified image instead of the local image. This allows us to run code that needs specific python libraries, tools or configuration without interfering with the local environment.

For deeper understanding, I highly recommend exploring the official kubeflow pipelines documentation, which covers these topics in detail. Also, consider reading "Kubernetes in Action" by Marko Lukša for a comprehensive understanding of how Kubernetes operates; a solid grounding in the fundamentals will clarify the underlying mechanisms involved in orchestrating pipelines. The official kubeflow website also contains numerous tutorials that are worthwhile. Furthermore, the source code for the `kfp` library (usually found on GitHub) is an excellent reference for understanding the nuances of the submission process and how it interacts with the kubeflow API.

In conclusion, submitting local jobs using `dsl.pipeline` is more about leveraging the `kfp` client in a way that submits pipelines to a locally running kubeflow instance, instead of directly executing anything locally on your development machine. Understanding this distinction is key to effectively utilizing kubeflow pipelines in a development environment. Remember the examples I've provided; they should provide a solid foundation for building more complex workflows in your own projects. And, as always, don't hesitate to thoroughly explore the resources I've mentioned. They will equip you with the knowledge and understanding necessary to navigate the finer details of kubeflow pipelines.
