---
title: "How can I debug code in VS Code when using an Azure ML workspace?"
date: "2024-12-23"
id: "how-can-i-debug-code-in-vs-code-when-using-an-azure-ml-workspace"
---

Alright, let's talk about debugging within an Azure ML workspace using vscode. It’s a process I've become quite familiar with over the years, especially during a large-scale distributed training project I was involved in. We were pushing the boundaries with a deep learning model, and effective debugging became absolutely critical. It’s not always straightforward, but with the correct setup, you can drastically cut down on iteration time.

The core challenge, of course, is that your code isn't running locally; it's executing on a remote compute target within Azure. This means traditional breakpoints and local debuggers won’t just work out of the box. We need to bridge that gap, and VS Code provides the necessary tools. What's also crucial to acknowledge from the outset is that, depending on *where* you're running the code (i.e., a training job vs. an inference script vs. interactive notebook), there are slightly different approaches. I'll break down the most common scenarios.

The most frequent case you'll encounter is debugging a training script. In this situation, your code typically resides in your VS Code environment (either locally or connected remotely to a compute instance) but executes on a remote target such as a cluster or a VM. To achieve this, we lean on the VS Code remote debugging capabilities in tandem with Azure ML’s SDK and compute configuration.

Here's the workflow I’ve found most reliable:

1.  **Remote Compute Preparation:** Firstly, your target compute instance (whether it's a compute cluster or a compute instance) must have the `debugpy` library installed. This is the critical component that enables remote debugging. We also need to ensure the debugger port is open and accessible. AzureML takes care of most of the port allocation in case of compute instances but you might need to explicitly open the port in case of a compute cluster.
2.  **VS Code Debugger Configuration:** We'll need to create a launch configuration in VS Code (`launch.json`). This configuration will instruct VS Code on how to connect to the remote debugger running on the Azure compute. We set the `request` to `attach` as we're not starting a new process, but rather connecting to an existing one.

Here’s an example of a `launch.json` configuration that you can tailor:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Azure ML Remote Debug",
            "type": "python",
            "request": "attach",
            "connect": {
                "host": "your_compute_instance_ip_or_hostname",
                "port": 5678 //ensure this is consistent
            },
            "pathMappings": [
                 {
                    "localRoot": "${workspaceFolder}",
                    "remoteRoot": "." // Assuming your project root is the same in both environments.
                 }
            ],
            "justMyCode": false //important, otherwise it might step into library internals
        }
    ]
}
```

*   **`your_compute_instance_ip_or_hostname`:** You need to replace this with the actual IP address or hostname of your compute instance. This isn't needed for compute instances connected directly through VSCode, where `localhost` can work instead. For compute clusters, the approach is a bit more involved, and I’ll discuss it shortly.
*   **`port: 5678`:**  This is a default port that `debugpy` uses. You can change it, but keep it consistent with what's configured in the python script that you want to debug, described next.
*   **`pathMappings`:** Crucial to avoid debuggers getting lost. You are telling the debugger where to find the corresponding local files when it's debugging remotely. It assumes that your folder structure is consistent between your local and remote environments.

3.  **Python Code Integration:** Before running your Azure ML job, you need to integrate the `debugpy` code into your python script. I usually do this conditionally, triggering the debugger only if an environment variable is set. This allows for production runs without the debugger enabled. I usually put this snippet at the very beginning of my python code:

```python
import os
import debugpy

if os.getenv("DEBUG_TRAINING") == "true":
    debugpy.listen(("0.0.0.0", 5678)) #important, 0.0.0.0 listens to connections on all available interfaces.
    print("Debugger listening on port 5678, waiting for connection...")
    debugpy.wait_for_client()
    print("Debugger attached, starting training...")

#rest of your training code here
```

4.  **Job Submission with Debugging Flag:** Finally, you initiate your Azure ML training job, making sure to set the `DEBUG_TRAINING` environment variable to `true` within the environment configuration of your job definition. This ensures that your script will start the debugger.
5.  **Connecting the Debugger:** Run the training job. Then, in VS Code, select the "Azure ML Remote Debug" configuration we created earlier and start the debugger using the “Run” button. If everything is set correctly, VS Code will attach to the remote debugging process, and you can step through the code by setting breakpoints, viewing variables, etc.

Now, concerning debugging on a compute cluster, things get slightly trickier. Since a cluster involves multiple nodes and dynamic allocation, getting the specific ip addresses beforehand becomes impractical. My solution, after struggling with this several times, is to use the concept of *port forwarding*. The idea is that after the training starts on the compute cluster, the entry point will start debugging. Instead of manually trying to find the right ip addresses, we can configure the job to expose a single port of one of the nodes, and forward it to a local port in our vscode environment:

```python
import os
import debugpy
import socket


def get_free_port():
    sock = socket.socket()
    sock.bind(('', 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


if os.getenv("DEBUG_TRAINING") == "true":
    debug_port = get_free_port()
    debugpy.listen(("0.0.0.0", debug_port))
    print(f"Debugger listening on port {debug_port}, waiting for connection...")
    os.environ['DEBUG_PORT']=str(debug_port) #export it for the job later
    debugpy.wait_for_client()
    print("Debugger attached, starting training...")
```

In this case, we are programmatically getting a random port and exporting it as a variable that we can use when submitting the job. When submitting the job using the azure ml sdk, you would add something along the lines of the following to your `job.run` configuration:

```python
from azure.ai.ml import command
from azure.ai.ml.entities import Environment, CommandJob

env = Environment(
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04-py38-cpu", #or any custom image
    conda_file="conda.yaml",  #define your dependencies here
)


job = command(
    code="./src", #your code directory
    command="python train.py",  #your entrypoint script
    environment=env,
    compute="my-compute-cluster",
    environment_variables={"DEBUG_TRAINING":"true"},
    distribution={
        "type": "pytorch",
        "process_count_per_node": 1
    },
    display_name="debug_job"
)

#port forwarding configuration
from azure.ai.ml import Input
from azure.ai.ml.entities import Port
job.inputs["debug_port"] = Input(type="string",  mode='download',  data=os.environ['DEBUG_PORT'])

job.ports = {
            "debug": Port(port=job.inputs["debug_port"],
                target_port=int(os.environ['DEBUG_PORT']),
                protocol="tcp",
                publish=True
            ),
       }



ml_client.jobs.create_or_update(job)


```

The above snippets of python code would effectively dynamically allocate a port on the cluster and forward it to one of the cluster nodes. Using this we can attach the debugger by specifying `localhost` and the corresponding exposed port in the vscode launch.json:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Azure ML Remote Debug",
            "type": "python",
            "request": "attach",
            "connect": {
                "host": "localhost",
                "port": "${command:azureml.getPort}"  // dynamic port lookup
            },
            "pathMappings": [
                 {
                    "localRoot": "${workspaceFolder}",
                    "remoteRoot": "." // Assuming your project root is the same in both environments.
                 }
            ],
            "justMyCode": false //important, otherwise it might step into library internals
        }
    ]
}
```

Notice the `"${command:azureml.getPort}"` in the `port` field. The AzureML VSCode extension will dynamically map the debug port on the cluster to a port in your local environment, so that the debugger can correctly connect.

For deeper dives into distributed debugging strategies and practices I highly recommend:

*   "Programming Machine Learning: From Coding to Deployable Products" by O'Reilly. It covers strategies for handling more complex debugging scenarios with distributed systems.
*   The official debugpy documentation, which is available on the project’s GitHub repository, gives you a more thorough understanding of its functionalities.

Keep in mind that successful debugging with remote compute targets demands a keen attention to detail. Make sure your network settings allow communication between your VS Code instance and the remote machine, and that your environment variable configurations are properly set. It might feel cumbersome initially but, as it becomes routine, it will drastically reduce the time it takes to iron out issues.
