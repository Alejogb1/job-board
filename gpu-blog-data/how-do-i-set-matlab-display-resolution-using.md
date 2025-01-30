---
title: "How do I set MATLAB display resolution using SLURM?"
date: "2025-01-30"
id: "how-do-i-set-matlab-display-resolution-using"
---
The core challenge in setting MATLAB display resolution within a SLURM environment stems from the inherent separation between the interactive login node and the compute nodes where jobs execute.  MATLAB's display settings are typically tied to the X11 display server running on the login node, which is generally inaccessible from within a compute node job.  Therefore, headless execution, coupled with post-processing of generated results, is often the preferred approach. However, in scenarios requiring real-time visualization from within the compute node (e.g., debugging or interactive simulations), a more sophisticated solution is needed.  My experience working on large-scale simulations within a high-performance computing cluster has highlighted the intricacies of this problem.

**1. Understanding the Limitations and Approaches**

SLURM allocates compute resources—CPU, memory, network—but doesn't directly manage graphical interfaces.  Attempting to directly set MATLAB's display resolution within a SLURM job script often fails because the compute node lacks the necessary X11 forwarding or a local display manager.  Consequently,  MATLAB functions like `set(0,'ScreenSize')` will either fail or report an incorrect resolution reflecting the compute node's default, typically low-resolution, virtual console.

The viable solutions hinge on either bypassing the need for a graphical display on the compute node entirely or establishing a secure connection to a remote display server.  The former is generally simpler for batch jobs, whereas the latter necessitates careful configuration of X11 forwarding and potential security considerations.

**2. Code Examples and Explanations**

**Example 1: Headless Execution and Post-Processing**

This is the most reliable method for most SLURM-MATLAB workflows. We avoid display manipulation entirely by focusing on data generation and subsequent visualization on the login node or a separate machine with a graphical interface.

```matlab
% MATLAB script (my_script.m)
data = generateData(); % Your data generation function
save('results.mat', 'data');
exit;
```

```bash
#!/bin/bash
#SBATCH --job-name=matlab_job
#SBATCH --partition=your_partition
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=matlab_job.out

module load matlab/your_matlab_version
matlab -batch "my_script.m"

# Post-processing on login node
matlab -nodisplay -r "load('results.mat'); visualizeData(data); exit;"
```

This script first runs the MATLAB code in headless mode (`-batch`). The generated data is saved. Afterwards, the post-processing command loads the saved data and executes visualization code on the login node, ensuring the display is available.


**Example 2: X11 Forwarding (with precautions)**

This approach attempts to forward the X11 display from the login node to the compute node.  It requires careful configuration and poses security risks if not implemented correctly.  It's generally not recommended for production unless absolutely necessary.

```bash
#!/bin/bash
#SBATCH --job-name=matlab_x11
#SBATCH --partition=your_partition
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=matlab_x11.out
#SBATCH -x <list of nodes to exclude>

export DISPLAY=:0.0 # crucial: export DISPLAY before matlab call

module load matlab/your_matlab_version

matlab -r "set(0,'ScreenSize',[1 1 1920 1080]); my_visualization_script; exit;"
```

Here, we explicitly set the `DISPLAY` environment variable before launching MATLAB. The `-x` option might be necessary to prevent a job from being scheduled on nodes where X11 forwarding might be unreliable.  Note:  Success relies on your cluster's X11 configuration; you might need to use `ssh -X` within your script for secure forwarding.   This approach is inherently fragile due to network conditions and cluster policies.  Thorough testing is essential.


**Example 3:  Using a Remote Display Manager (e.g., Xming)**

For greater control, you could use a remote X server such as Xming on your local machine and point MATLAB to it. This bypasses direct SLURM interaction with X11. However, network latency might impact interactive performance.

```bash
#!/bin/bash
#SBATCH --job-name=matlab_xming
#SBATCH --partition=your_partition
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=matlab_xming.out

export DISPLAY=your_local_machine_ip:0.0  #Replace with local IP and display number

module load matlab/your_matlab_version
matlab -r "set(0,'ScreenSize',[1 1 1920 1080]); my_visualization_script; exit;"
```

This requires the X server to be running on your local machine and the appropriate firewall rules configured. You need to replace `your_local_machine_ip` with your actual machine's IP address.  The performance here hinges on network bandwidth and latency between your local machine and the compute node.


**3. Resource Recommendations**

Consult your cluster's documentation regarding X11 forwarding and secure shell options.   Review MATLAB's documentation on headless operation and the `-batch` and `-nodisplay` command-line options. Understand the intricacies of your specific SLURM environment, including partition settings and available modules.  Explore the options for remote desktop access provided by your cluster administrators. Consider learning about alternative remote visualization tools if interactive graphical output on compute nodes is crucial to your workflow.  Finally, thorough testing on a small scale before attempting to deploy your solution for large-scale simulations is crucial to avoid unexpected issues.
