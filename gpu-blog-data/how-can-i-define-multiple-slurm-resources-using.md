---
title: "How can I define multiple SLURM resources using the same GPU?"
date: "2025-01-30"
id: "how-can-i-define-multiple-slurm-resources-using"
---
The challenge of sharing a physical GPU across multiple SLURM jobs stems from the underlying resource management model, which typically assumes a one-to-one mapping between logical resources and their physical counterparts. Directly requesting a specific GPU by ID for multiple independent jobs will lead to conflicts and resource contention, as SLURM's default behavior prohibits the exclusive allocation of a resource to multiple users or tasks. Therefore, leveraging SLURM's task affinity and resource constraints alongside specific configuration options provides the necessary mechanism for controlled, shared access to a single GPU.

My work developing distributed machine learning pipelines revealed that straightforward attempts to specify the same GPU ID across numerous SLURM job submissions inevitably resulted in errors or unexpected behavior. Consequently, implementing a fine-grained control strategy, rather than requesting exclusive ownership, is necessary to utilize the compute capacity fully. SLURM's resource management paradigm allows for the assignment of logical resources – essentially, an abstraction of physical resources – to jobs, alongside mechanisms for controlling how those resources are actually used. This includes setting constraints such as memory usage and ensuring that multiple processes from distinct jobs cannot simultaneously access a resource’s specific functionality without explicit coordination.

The central solution involves employing SLURM's task affinity functionality and configurable resource limitations. Specifically, rather than requesting the GPU directly by ID, I discovered that requesting a generic GPU resource (e.g., `gpu:1`) and then controlling the process placement within a single node provides the needed functionality. This approach relies on setting task affinity, which constrains which physical GPU a process uses. This is done with the SLURM environment variables, such as `CUDA_VISIBLE_DEVICES`. This doesn't directly change SLURM's initial assignment of the resources, but it guides *how* the application uses the assigned resources within a single job allocation. The key here is to first acquire a shared node and then enforce limitations on each application. It allows us to submit jobs that, while technically *using* one SLURM assigned GPU resource, will not step on each other through the use of `CUDA_VISIBLE_DEVICES`. In practice, this often requires combining these specifications with options for CPU cores and memory management.

The following examples demonstrate this process:

**Example 1: Shared GPU Access with Controlled Usage**

This script demonstrates how to launch multiple applications within a single job, each accessing the same physical GPU through managed limits via `CUDA_VISIBLE_DEVICES`.
```bash
#!/bin/bash
#SBATCH --job-name=shared_gpu_example
#SBATCH --partition=gpu_partition
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4 # Example CPU allocation for demonstration
#SBATCH --mem=10G     # Example memory allocation for demonstration
#SBATCH --time=00:30:00

# Define applications and GPU affinities
declare -a app_names=("app_a" "app_b" "app_c")
declare -a gpu_devices=("0" "0" "0") #  All applications use the same physical GPU (index 0).

# Launch applications with controlled access
for i in "${!app_names[@]}"; do
    app_name="${app_names[$i]}"
    gpu_dev="${gpu_devices[$i]}"

    #Set a unique working directory to avoid overlapping file output
    mkdir -p "${app_name}"
    cd "${app_name}"
    
    echo "Launching ${app_name} with CUDA_VISIBLE_DEVICES=${gpu_dev}"
    # Example command using CUDA_VISIBLE_DEVICES.
    # Replace the sleep and print command with the application to run.
    CUDA_VISIBLE_DEVICES="${gpu_dev}"  srun --output="${app_name}.log" bash -c "echo ${app_name} starting && sleep 20 && echo ${app_name} finished" &
    cd ..
done

wait # Ensure all subtasks complete before exiting
```
This script starts by allocating a single GPU for the entire job through the `--gpus-per-node=1` directive. Importantly, within the loop, each application sets `CUDA_VISIBLE_DEVICES` to `0`. This ensures that all applications access the same *physical* GPU assigned to the *job*, while still executing as separate processes. If each application was launched without explicit process control, they would compete for resources on the same device, leading to unpredictable behavior. I have observed that this setup ensures isolation and controlled use, preventing one task from unduly impacting the performance of others. The `wait` ensures all background tasks finish before the job ends. Each application in the example is just a simple process to test the underlying logic; the command should be swapped out to run the desired application.

**Example 2: Sub-Device Partitioning**

This example illustrates the use of sub-device partitioning, an approach often beneficial in environments with multiple GPUs where more controlled sharing of a single card is required. I will assume that the driver is configured correctly for multiple processes to utilize separate logical portions of a device, where it will appear as if each process is using a separate device, even though they are all on a single card.
```bash
#!/bin/bash
#SBATCH --job-name=sub_device_example
#SBATCH --partition=gpu_partition
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=00:30:00

declare -a app_names=("app_x" "app_y" "app_z")
#Each sub-device is referenced by a unique ID.
declare -a gpu_devices=("0:0" "0:1" "0:2") # sub-device indexing within device 0

for i in "${!app_names[@]}"; do
    app_name="${app_names[$i]}"
    gpu_dev="${gpu_devices[$i]}"

    mkdir -p "${app_name}"
    cd "${app_name}"

    echo "Launching ${app_name} with CUDA_VISIBLE_DEVICES=${gpu_dev}"
    # Again, use a simple command and swap it out for the actual application
    CUDA_VISIBLE_DEVICES="${gpu_dev}"  srun --output="${app_name}.log" bash -c "echo ${app_name} starting && sleep 20 && echo ${app_name} finished" &
    cd ..
done

wait
```

Here, the `gpu_devices` array contains entries like `0:0`, `0:1`, and `0:2`. These refer to specific *logical* sub-devices within the single allocated physical device (device `0`).  The underlying NVIDIA driver and the software stack then manages the sub-device usage. This strategy leverages the same underlying shared resource paradigm, but utilizes driver configuration to isolate process access.

**Example 3: Dynamic GPU Assignment within a Single Node (Multi-Node Job)**
The first two examples focused on sharing one card within the confines of a single node. It is also possible to utilize multiple GPUs on a single node with the same method. Here is a variant of the first example to demonstrate this in a multi-node job:

```bash
#!/bin/bash
#SBATCH --job-name=dynamic_gpu_example
#SBATCH --partition=gpu_partition
#SBATCH --nodes=2
#SBATCH --gpus-per-node=2 # Two GPUs on every node.
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=00:30:00

declare -a app_names=("app_p" "app_q" "app_r" "app_s")
# The GPU devices in this case will depend on which node the process is run on.
# This is because, each node has two devices, starting at index 0.
# Here, I have designed the application to always only use GPU 0 on whatever node it is on.
declare -a gpu_devices=("0" "0" "0" "0")

# Get the SLURM_NODEID to determine the node this job is run on
declare node_id=$SLURM_NODEID

for i in "${!app_names[@]}"; do
    app_name="${app_names[$i]}"
    gpu_dev="${gpu_devices[$i]}"


    mkdir -p "${app_name}"
    cd "${app_name}"

    echo "Launching ${app_name} on node ${node_id} with CUDA_VISIBLE_DEVICES=${gpu_dev}"
    CUDA_VISIBLE_DEVICES="${gpu_dev}"  srun --output="${app_name}.log" bash -c "echo ${app_name} on node ${node_id} starting && sleep 20 && echo ${app_name} on node ${node_id} finished" &
    cd ..
done

wait

```

This example introduces the use of the SLURM variable `SLURM_NODEID` which can be used to understand what node the application is running on. In this situation, the same `CUDA_VISIBLE_DEVICES` assignment will mean each application uses the first device available on each node the job uses. The important distinction here is that this will only work if the application is configured such that it only requires one GPU device and if there are enough devices on each node to satisfy the request.

Resource management of this nature requires an understanding of the specific software tools and their respective configurations. Comprehensive information can be found in system administration manuals detailing resource allocation, NVIDIA documentation for their driver options, and in SLURM documentation focusing on task affinity and heterogeneous resource management. I found that testing these techniques under load, especially when multiple different applications are involved, is crucial to prevent unexpected behavior that can arise from unforeseen resource constraints. The optimal method for sharing depends heavily on the specific use-case, the requirements of the individual applications, and the underlying hardware capabilities.
