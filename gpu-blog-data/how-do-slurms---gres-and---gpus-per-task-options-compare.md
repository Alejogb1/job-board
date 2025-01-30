---
title: "How do Slurm's `--gres` and `--gpus-per-task` options compare when allocating GPUs, and how does this differ between `mpirun` and `srun`?"
date: "2025-01-30"
id: "how-do-slurms---gres-and---gpus-per-task-options-compare"
---
The crucial distinction between Slurm’s `--gres` and `--gpus-per-task` lies in their intended purpose and effect on resource allocation granularity, primarily observed when integrating with `mpirun` versus `srun`. I’ve navigated this repeatedly in high-performance computing environments, and subtle nuances exist that can drastically affect performance.

`--gres`, or generic resources, is Slurm's mechanism for managing resources beyond CPUs and memory, such as GPUs, accelerators, and specific hardware. It provides a coarse-grained allocation. With `--gres=gpu:N`, I’m requesting *N* total GPUs for the entire job across all nodes, or if used with a node specification, for the specified nodes. This allocation is at the job level; all tasks within that job, irrespective of how they're launched with `mpirun` or `srun`, have access to these collectively allocated GPUs. The specific way these GPUs are assigned to individual processes is outside the control of the `--gres` flag.

`--gpus-per-task`, conversely, dictates the number of GPUs assigned to each task launched *directly* by `srun`. This is a fine-grained approach focusing on individual task resource requirements. `srun` directly interprets and enforces this allocation. The flag has no inherent effect when used with `mpirun`.  Instead, the environment variables (e.g., `CUDA_VISIBLE_DEVICES`) must be used within the application's context alongside `mpirun` to manage GPU access, often using information about the process rank provided by MPI itself.

When using `mpirun`, the situation becomes more complicated. Because `mpirun` manages the distribution of processes across nodes and their communication, it needs the environment to inform it which GPUs a particular process should use. Slurm’s direct allocations through `--gpus-per-task` are ignored. Instead, we must use `--gres` to allocate the total GPU resource and configure the application to use environment variables to selectively assign GPUs to individual MPI processes based on process rank, or use a framework-specific method to map tasks to GPUs. I've seen numerous instances where neglecting this detail leads to all processes competing for the same GPU, creating an unexpected bottleneck.

Let’s examine some code examples to illustrate these distinctions:

**Example 1: Using `--gres` with `srun`:**

```bash
#!/bin/bash
#SBATCH --job-name=gres_test
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --time=00:10:00
#SBATCH --output=gres_test.out
#SBATCH --error=gres_test.err

srun -n 4 nvidia-smi
```

In this scenario, I request 4 GPUs in total using `--gres=gpu:4`. I am using `srun` to launch four processes. Even though I specify four processes, each `srun` instance launched will potentially try to access all 4 GPUs, as `--gres` assigns them to the job, not the specific srun. This means I need to manage the GPU allocation inside the application itself with environment variables such as `CUDA_VISIBLE_DEVICES`. If I ran four processes that directly tried to claim a specific GPU without explicit environment variable settings, all would try to access the same GPU by default which would cause resource contention. I've observed this often causing performance bottlenecks or program errors during debugging. If I ran on a single node with four GPUs, all four processes could use different GPUs, but this is not ensured by Slurm. Running this across two nodes could result in two processes on one node using the same GPU unless this is configured within the code.

**Example 2: Using `--gpus-per-task` with `srun`:**

```bash
#!/bin/bash
#SBATCH --job-name=gpus_per_task_test
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=gpus_per_task_test.out
#SBATCH --error=gpus_per_task_test.err

srun nvidia-smi
```
Here, I utilize `--gpus-per-task=1`. I am running with two nodes, and two tasks per node. I have set `--gpus-per-task=1`.  This makes `srun` allocate one GPU per task across the allocated compute resources. Each `srun` will have access to only a single specific GPU (or in case there are not enough GPUs, only one process per GPU). In this scenario, if each node has two GPUs, the system will correctly assign one GPU to each process via the `CUDA_VISIBLE_DEVICES` environment variable which can be seen with `nvidia-smi`.  This mode of operation is much easier to manage for single-node parallelism using `srun` directly, as the association of tasks and GPUs is handled by Slurm without additional programming intervention.

**Example 3: Using `--gres` with `mpirun`:**

```bash
#!/bin/bash
#SBATCH --job-name=mpirun_test
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:4
#SBATCH --time=00:10:00
#SBATCH --output=mpirun_test.out
#SBATCH --error=mpirun_test.err

mpirun -np 4 ./gpu_aware_app
```

In this third scenario, I leverage `--gres=gpu:4` to request 4 GPUs as before, but I’m employing `mpirun` instead of `srun` to launch 4 processes of `./gpu_aware_app`.  Crucially, note that there is no `--gpus-per-task` flag being used; that would be completely ignored by `mpirun`. The program, `gpu_aware_app`, must be designed to correctly interpret MPI rank information and use it to select a GPU from available GPUs. Typically, this involves reading the `CUDA_VISIBLE_DEVICES` environment variable set by Slurm and using the MPI rank to select which GPU each process should use. This is frequently achieved by dividing the number of GPUs available by the number of processes to calculate which GPU is assigned to each MPI process. If `gpu_aware_app` fails to do this, all processes could attempt to access the same GPU. I’ve debugged countless performance issues stemming from programs not handling this correctly.  `mpirun` uses the Slurm-provided GPUs; however, it is the responsibility of the application itself to select and use them.

In summary, the key difference lies in the allocation granularity. `--gres` provides job-level allocation, where the responsibility for assigning specific GPUs to tasks falls on the application when used with `mpirun` and can lead to uncontrolled usage with `srun` without appropriate environment variables. `--gpus-per-task`, used with `srun`, allows task-specific GPU assignments, simplifying development for simpler parallelization. When integrating with `mpirun`, one must explicitly control the allocation through a combination of the `--gres` flag, environment variables and the MPI rank. The choice depends heavily on how your parallel application is designed and launched. Choosing the correct flag, and more importantly ensuring that the application is properly utilizing the allocated resources is critical to achieving the best performance. This is a skill I've honed after years of experience troubleshooting large-scale parallel simulations.

For users wishing to further explore these topics, I recommend consulting the following resources. Firstly, thoroughly review the Slurm documentation; especially the sections about resource allocation and generic resources. Secondly, examine the documentation for the specific MPI implementation used as it often contains advice about resource management and environment variable handling within the context of distributed applications. Thirdly, study best practices documents related to GPU utilization in HPC environments provided by GPU vendors and HPC centres. These documents usually provide practical guidelines that will save time and effort.
