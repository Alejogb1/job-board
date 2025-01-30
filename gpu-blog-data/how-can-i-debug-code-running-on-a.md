---
title: "How can I debug code running on a SLURM compute node within VS Code?"
date: "2025-01-30"
id: "how-can-i-debug-code-running-on-a"
---
Debugging code directly on a SLURM compute node from VS Code requires a bridge, given that these nodes are typically headless and not directly accessible through traditional IDE debugging features. I've encountered this scenario frequently managing HPC resources and have refined a workflow that uses SSH port forwarding and a Python debugger. This approach avoids installing GUI elements on the compute nodes, which is often restricted and inefficient, while still leveraging VS Code's debugging interface.

The fundamental problem lies in the separation between your local development environment where VS Code runs, and the remote compute node where the code executes.  Direct debugging protocols like those used for local applications are not readily available across this network boundary. SLURM, being a batch scheduler, adds another layer of abstraction. Therefore, we need a mechanism to establish a communication channel, and a means for VS Code's debugger to connect through this channel.

My solution hinges on the following principles: Firstly, establishing a secure tunnel from my local machine to a specific port on the compute node using SSH. Secondly, launching the application with a debugger in server mode on the compute node, bound to that port. Lastly, configuring VS Code to act as a client, connecting to the exposed port on the remote server. This creates a bidirectional communication channel that facilitates control of the remote debugging session from the local VS Code instance.

Let's consider a simple Python application that might be submitted to SLURM:

```python
# example.py

def calculate_sum(a, b):
    result = a + b
    return result

if __name__ == "__main__":
    x = 10
    y = 20
    total = calculate_sum(x, y)
    print(f"The sum is: {total}")
```

Now, let's detail the steps and code needed to enable debugging within VS Code.

First, the Python script needs to be modified to work with a remote debugger, using `debugpy`:

```python
# debug_example.py
import debugpy
import time

def calculate_sum(a, b):
    result = a + b
    return result

if __name__ == "__main__":
    debugpy.listen(("0.0.0.0", 5678)) # Listens on all available interfaces on port 5678
    print("Waiting for debugger to attach...")
    debugpy.wait_for_client()
    time.sleep(1) #Allow debugger time to connect
    
    x = 10
    y = 20
    total = calculate_sum(x, y)
    print(f"The sum is: {total}")
```

In this version, I have imported the `debugpy` library. The line `debugpy.listen(("0.0.0.0", 5678))` starts the debugger in server mode, listening for incoming connections on port 5678. `debugpy.wait_for_client()` ensures the script will pause execution until a debugger is connected, allowing for setting breakpoints before the core logic executes. The `time.sleep(1)` provides a brief pause for the VS Code debugger to establish connection without racing. Importantly, listening on `0.0.0.0` makes it accessible from other machines, which is critical for this setup.

Next, let's examine how I would write the SLURM script and configure local SSH tunneling:

```bash
#!/bin/bash
#SBATCH --job-name=debug_job
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --output=debug_job_%j.out
#SBATCH --partition=your_partition # Replace with your partition

module load python/3.10 # Load the python module where debugpy is installed

#Ensure the SSH tunnel is setup before launching the job
echo "Establishing ssh tunnel on port 5678"

# Execute the python script with the debugger
srun python debug_example.py
```
This is a minimal SLURM script specifying a single node, task, a time limit, and output location. Crucially, I load the required Python environment using `module load`. It is critical that the Python environment that the script executes in is the same that has `debugpy` installed, typically via `pip install debugpy`. The `srun` command executes the Python script, this ensures that the debugger can be launched on the node the job is allocated to.

Before submitting the job, I would establish an SSH tunnel from my local machine to the remote compute node. The command I use in my terminal is:

```bash
ssh -L 5678:localhost:5678 <username>@<login_node_address> -N
```

This command forwards port 5678 on my local machine to port 5678 on the compute node. I would then submit the SLURM job using:

```bash
sbatch your_script_name.sh
```
Finally, on the VS Code side, I need to create a debug configuration in the `launch.json` file:

```json
{
  "version": "0.2.0",
  "configurations": [
      {
          "name": "Python Remote Attach",
          "type": "python",
          "request": "attach",
          "connect": {
              "host": "localhost",
              "port": 5678
          },
          "pathMappings": [
              {
                  "localRoot": "${workspaceFolder}",
                  "remoteRoot": "."
              }
          ],
            "justMyCode": false
      }
  ]
}
```

The key parts here are the `request` set to `"attach"`, which indicates that VS Code will connect to an already running debugger, and the `connect` configuration. The `host` is set to `localhost` because of the SSH tunnel I've set up, and `port` is set to 5678. `pathMappings` are critical as the file location on your local machine may differ from the compute node and this directs the debugger where to find the corresponding source code. The `justMyCode` configuration set to `false` means that I can debug into library files, which is useful for more advanced debugging.

This process creates a bridge. The application executes on the SLURM node, but its debugger is accessible locally thanks to the SSH tunnel. VS Code then acts as a client, allowing me to step through code, examine variables, set breakpoints on the remote execution. I have used variations of this approach for complex MPI jobs and it is crucial to understand the interplay of these components, especially when debugging multi-node applications which would involve additional steps with the debugger.

In terms of recommended resources, I would advise exploring the official `debugpy` documentation for a more detailed understanding of its features. The VS Code documentation on debugging Python is essential for familiarizing yourself with its debugging interface. Furthermore, I have found that thoroughly reviewing the SSH documentation on port forwarding is beneficial in understanding the tunneling concept. A deep understanding of SLURM job scripting will help tailor the setup to specific computational needs. Finally, testing the connection and the setup without any breakpoint before debugging into code is good practice in complex distributed systems.
