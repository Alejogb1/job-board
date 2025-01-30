---
title: "Why can't I run the matmul example app locally using COMPSs?"
date: "2025-01-30"
id: "why-cant-i-run-the-matmul-example-app"
---
The primary reason local execution of a COMPSs application, specifically the *matmul* example, often fails relates to COMPSs's reliance on a distributed execution environment. COMPSs is fundamentally designed to orchestrate tasks across multiple nodes, thereby leveraging parallelism. It is not, by default, configured for straightforward, single-machine local runs that one might expect from standard Python scripts.

My experience debugging similar COMPSs configurations across various research environments has consistently highlighted this core issue. While a local installation of the COMPSs runtime is possible, it requires meticulous configuration mimicking, in part, a cluster setup. Without this configuration, COMPSs struggles to identify resources (nodes and workers) necessary for task execution, leading to the failure of application launch, and often generates errors related to lack of active workers or the inability to connect to a specified resource manager. This contrasts sharply with Python's standard direct execution model.

COMPSs employs a programming model where tasks are defined using decorators, and these decorated functions aren't directly executed during the main program's flow. Instead, they are translated into a task graph that the COMPSs runtime manages. This runtime distributes task execution across available workers, which are separate processes or even nodes. Local execution necessitates configuring a 'local' node and worker to simulate this distributed environment, which is not intuitive and demands specific configuration.

The matmul example, like many COMPSs sample applications, leverages this distributed task execution model. The example's design typically presumes the availability of a configured COMPSs environment with running worker daemons. Thus, directly attempting to execute the *matmul.py* script without appropriately configuring the local environment causes the runtime to either hang, output a series of obscure errors, or in some cases, execute nothing at all.

Below, I present three code examples illustrating common pitfalls and configuration requirements:

**Example 1: The Naive Attempt and its Failure**

Assume the following simplified `matmul.py` file (following typical COMPSs style):

```python
from pycompss.api.task import task
from pycompss.api.api import compss_wait_on, compss_barrier

import numpy as np

@task(returns=1)
def multiply(a, b):
  return np.dot(a, b)

def generate_matrix(size):
  return np.random.rand(size, size)

if __name__ == "__main__":
  size = 1024

  a = generate_matrix(size)
  b = generate_matrix(size)

  c = multiply(a, b)
  c = compss_wait_on(c)
  print("Result shape:", c.shape)
  compss_barrier()
```

Directly executing this script, with `python matmul.py`, often results in either a stalled execution, or an error message like: “*No available worker to execute the task...*”. This is because COMPSs does not know where to launch the `multiply` task. No workers are explicitly registered by default for the local machine. This attempt fails because the user assumes implicit local execution.

**Example 2: Explicit Local Configuration (Partial)**

To partially address the above, configuration is necessary. This involves creating a *project.xml* file that dictates the execution environment. A minimal *project.xml* targeting local execution could look like this:

```xml
<Project>
    <Resources>
        <Resource name="localhost" adaptor="local" workingDir="/tmp/compss">
            <ComputingUnits>
                <ComputingUnit type="CPU" cores="4"/>
            </ComputingUnits>
        </Resource>
    </Resources>
    <Application>
        <Execution>
          <Parameter key="log_level" value="debug"/>
        </Execution>
        <Tasks>
            <Task>
                <Implementation/>
            </Task>
        </Tasks>
    </Application>
</Project>
```

This *project.xml* file specifies a single local resource named "localhost" and that COMPSs should use local execution to run the tasks, with `/tmp/compss` as the worker's working directory. The key is the adaptor="local" parameter.

Now, attempting to run the *matmul.py* script while pointing COMPSs at this configuration:

`runcompss --project=project.xml matmul.py`

While this approach eliminates the “*No available worker*” error, it might still result in execution problems or potentially hang if the worker is not fully configured. This is because, even with the XML configuration, there is no guarantee that the local environment is set up to run the worker in the way expected by COMPSs. Often it can be permissions issues or the absence of the `/tmp/compss` directory, or more fundamental issues with runtime initialization. This does not fully configure the needed internal runtime process.

**Example 3: Full Local Setup Including Worker Start**

To enable successful local execution, in addition to the *project.xml*, one must explicitly start a local worker daemon before application execution.  While there isn't a python code example for starting the worker, the standard COMPSs workflow includes a command line call for this, typically:

`compss_worker -d &`

This command starts a COMPSs worker in daemon mode in the background. This worker will read the project.xml and configure itself based on it.

After starting the worker and *then* running:

`runcompss --project=project.xml matmul.py`

The application will now execute correctly as COMPSs is able to find a worker to run the task `multiply`, and results will be printed on standard output. This illustrates the need for more than just the XML configuration, but an active worker. This is also critical because the *runcompss* script doesn’t implicitly start a worker like a local single-process framework might.

In summary, local execution of COMPSs applications is not a simple 'run it locally' operation. The framework requires a level of explicit resource management and worker configuration, even when running on a single machine, mimicking its distributed execution design. The `project.xml` outlines the resource and the `compss_worker` command launches a working process that can execute task methods. The crucial missing piece in the initial attempt was not that the COMPSs application was broken, but that COMPSs had nowhere to run its tasks.

**Recommendations:**

For users encountering such issues, I suggest consulting the official COMPSs documentation for detailed guidance on local configuration. Focus on sections describing local deployment, resource management, and the setup of execution environments, specifically the *project.xml* format, *runcompss* command and *compss_worker* script. In addition to the documentation, reviewing examples provided by the COMPSs installation and exploring online tutorials from reputable sources can give practical hands-on knowledge, that directly targets common use cases. I would specifically recommend spending extra time in the official user manuals that outline the various flags for the main shell scripts, as these are vital for understanding specific runtime execution needs. These resources provide the specific details needed to overcome these common initialization issues.
