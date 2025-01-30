---
title: "Can Slurm run multiple arrays on the same node?"
date: "2025-01-30"
id: "can-slurm-run-multiple-arrays-on-the-same"
---
As someone who has managed high-performance computing clusters for over a decade, I've encountered situations where optimal resource utilization requires running multiple independent job arrays within a single compute node. The short answer is: yes, Slurm can manage multiple arrays on the same node, but it requires careful consideration of resource constraints and job dependencies. This is not an automatic behavior and depends heavily on how you define your array jobs and the resources they request.

The fundamental principle is that Slurm schedules jobs based on resource requests (CPU cores, memory, GPUs, etc.). If a node has enough available resources to satisfy the requirements of multiple array jobs, Slurm will distribute those array tasks across the available cores. However, to understand *how* this happens and avoid oversubscription or contention issues, we need to delve into the configuration of your job submission scripts and the Slurm configuration itself.

A standard array job submission using the `--array` flag will, by default, attempt to distribute array tasks across the entire cluster as evenly as possible. If you want to specifically constrain multiple arrays to a single node or a subset of nodes, you need to combine `--array` with other flags such as `--nodes`, `--nodelist`, or `--cpus-per-task`. Additionally, if one array job is dependent on the completion of another, you must also manage their dependencies using the `--dependency` option or similar Slurm tools. These mechanisms control the order and placement of jobs.

Let’s consider a scenario: Suppose I have two array jobs. One, `data_processing`, needs to pre-process files; and another, `model_training`, uses this processed data. I need each array job to utilize half of a 16-core node for its various tasks. I will also manage the dependence of `model_training` on `data_processing`. Here’s how I’d approach this with Slurm job submission scripts.

First, here's the `data_processing.sh` script:

```bash
#!/bin/bash
#SBATCH --job-name=data_processing
#SBATCH --array=1-10
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=data_processing_%A_%a.out

echo "Starting data processing task $SLURM_ARRAY_TASK_ID on node $SLURM_NODELIST"

# Simulate processing. In real code, use file names based on SLURM_ARRAY_TASK_ID
sleep 60
echo "Data processing task $SLURM_ARRAY_TASK_ID completed."
```

In this script:

*   `#SBATCH --array=1-10` defines an array with 10 tasks.
*   `#SBATCH --nodes=1` explicitly constrains the job to run on a single node.
*   `#SBATCH --cpus-per-task=8` requests 8 cores per array task. Because we intend to fit two jobs on the node, we use half of its 16 cores.
*   `#SBATCH --output=data_processing_%A_%a.out` creates unique output files, using the Slurm job ID (`%A`) and array task ID (`%a`).

Now let's examine the `model_training.sh` script:

```bash
#!/bin/bash
#SBATCH --job-name=model_training
#SBATCH --array=1-5
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=model_training_%A_%a.out
#SBATCH --dependency=afterany:data_processing

echo "Starting model training task $SLURM_ARRAY_TASK_ID on node $SLURM_NODELIST"

# Simulate training. In real code, use data processed by the prior array tasks
sleep 90
echo "Model training task $SLURM_ARRAY_TASK_ID completed."
```

Here, we have an array job `model_training` with 5 tasks. Like `data_processing`, we explicitly restrict this to a single node, with the same resource requirement, and also specify the output files. Crucially, `#SBATCH --dependency=afterany:data_processing` ensures that this array job only starts after the successful completion of *all* tasks in the `data_processing` job array.

To run these, I would first submit `data_processing.sh` using `sbatch data_processing.sh`. Once the `data_processing` job completes, I would then submit `model_training.sh` using `sbatch model_training.sh`. Because `model_training` depends on the prior job’s completion and is constrained to the same node, you will see Slurm schedule both on the same node without additional intervention.

However, let’s now consider a situation where the arrays do not have a direct dependency but simply must run concurrently on the same nodes.

Here's another scenario. Imagine we've a parameter sweep, divided into two arrays; `sweep_param_a` and `sweep_param_b`, which each need a quarter of the 16-core node (4 cores each). They're independent but for optimal throughput need to run concurrently.  I will again use `--nodes=1` and adjust `--cpus-per-task` accordingly.

Here is an example submission script, `parameter_sweep.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=parameter_sweep
#SBATCH --array=1-20
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:30:00
#SBATCH --output=parameter_sweep_%A_%a.out

echo "Starting parameter sweep task $SLURM_ARRAY_TASK_ID on node $SLURM_NODELIST"

# Simulate parameter sweeps
sleep 45
echo "Parameter sweep task $SLURM_ARRAY_TASK_ID completed."
```
Now, to run multiple instances of this array job concurrently on the same node, I would submit this job multiple times, perhaps like so: `sbatch parameter_sweep.sh; sbatch parameter_sweep.sh`. Slurm, seeing that each requests only 4 cores on a single node, can fit up to four instances concurrently on the same 16 core machine. Slurm can be also be used to submit these array jobs programmatically.

It’s important to be aware of potential conflicts and to tune parameters of your submission script. Here are some key points to consider:

*   **Memory Management:** While CPU cores are directly allocated, memory is shared. You need to ensure the total memory requests for all arrays concurrently running on a node do not exceed the node’s total physical memory. Over-subscription could lead to swapping and significant performance degradation.
*   **Resource Limits:** Slurm might have configured limits on per-node resource requests which may require adjustment via Slurm configuration. Familiarity with the Slurm configuration on your cluster is crucial to avoid hitting hard limits when attempting multi-array, single node workloads.
*   **Dependencies:** When arrays are interdependent, managing their order and timing correctly is crucial to prevent errors. Use Slurm dependency options (e.g., `afterany`, `afterok`, `afternotok`) precisely.
*   **Task Distribution:** While `sbatch` will schedule the tasks automatically, you might consider job step creation with `srun` within the submission script for finer-grained control over the allocation of resources.
*   **Debugging:** Always use array output file generation (e.g. via `%A` and `%a`) so as to easily analyze each array task individually. Monitor jobs using `squeue` and use tools like `scontrol show jobid <jobid>` to understand resource allocation.
*   **Node Specialization:** You can control array placement with more complex node selection rules, based on specific features or node types, by combining these array techniques with the `--constraint` option if your Slurm configuration supports it.

In summary, Slurm is perfectly capable of running multiple job arrays on the same node. However, success depends on careful resource planning, explicit definition of job dependencies, and understanding of your Slurm configuration. Over-subscription and poor resource management are common issues that can be mitigated with careful scripting and monitoring. For further learning, the official Slurm documentation and training resources available from various HPC institutions are excellent sources of information. Seek out materials on job arrays, dependencies, and resource management in Slurm for further study.
