---
title: "How can I run parallel jobs without hyperthreading in a Torque PBS script?"
date: "2025-01-30"
id: "how-can-i-run-parallel-jobs-without-hyperthreading"
---
Parallel processing within a Torque PBS environment, specifically when circumventing hyperthreading, requires careful management of resource allocation and job execution directives. This becomes crucial in high-performance computing (HPC) environments where the efficient use of physical cores maximizes computation speed while avoiding potential performance bottlenecks introduced by sharing a core’s resources. My experience managing several HPC clusters has shown that improper resource requests, particularly when dealing with multi-core nodes, often lead to suboptimal performance.

The default behavior of many job schedulers, including Torque, may not automatically respect the distinction between a physical core and a hyperthread, sometimes assigning jobs to hyperthreads on the same physical core. This can degrade performance when the underlying workload is CPU-bound. Therefore, we need to explicitly specify how many physical cores we intend to use and, importantly, ensure the scheduler places jobs onto dedicated cores. We accomplish this by employing node files, resource requests, and process binding commands within our Torque PBS script.

The fundamental approach involves reserving a specific number of cores and subsequently binding the execution of each parallel process to a separate, dedicated core. Node files become essential in this context. Torque allows us to request multiple nodes, often with a certain number of processors per node. However, without further guidance, the scheduler can distribute processes among hyperthreads. By generating node files, we can enumerate the specific cores to be used and then employ a method of process binding to ensure processes adhere to this arrangement.

In practice, generating the node file starts with our initial resource request to the scheduler. Here, we need to define the number of nodes and processes (or cores) requested using PBS directives, such as `-l nodes=X:ppn=Y`, where X is the number of nodes requested, and Y is the number of processes (or cores) requested *per node*. After this request, a dynamically generated file is created that stores the physical hostnames and allocated cores. This file can be accessed via the `$PBS_NODEFILE` environment variable. It is this variable which is used to build the list of cores used. We then parse this file, and use a tool like `taskset` (or `numactl` on systems with non-uniform memory access) to bind processes to specified core(s).

Here's a code example that illustrates requesting 4 cores (no hyperthreading) on a single node and using them to launch a parallel application:

```bash
#!/bin/bash
#PBS -l nodes=1:ppn=4
#PBS -N no_hyperthread_example
#PBS -j oe

# Get the path to the nodefile
NODE_FILE=$PBS_NODEFILE

# Initialize an array to hold the assigned cores
cores=()

# Read the nodefile line by line, populating the cores array with each CPU's ID
while IFS= read -r line; do
    node=$(echo "$line" | awk '{print $1}')
    cpus=$(echo "$line" | awk '{print $2}' | tr ',' ' ')
    for cpu in $cpus; do
      cores+=("$cpu")
    done
done < "$NODE_FILE"

# Check if we received the expected number of cores.
if [ ${#cores[@]} -ne 4 ]; then
    echo "ERROR: Expected 4 cores, but found ${#cores[@]}." >&2
    exit 1
fi

# Launch a parallel job. Here, using a hypothetical command 'my_parallel_app'.
# `taskset` binds each instance of the application to an assigned core.
my_parallel_app -n 4 \
--command "taskset -c ${cores[0]} my_task $PARAM1 &
taskset -c ${cores[1]} my_task $PARAM2 &
taskset -c ${cores[2]} my_task $PARAM3 &
taskset -c ${cores[3]} my_task $PARAM4 &
wait"
```

This script begins by requesting one node with four processors-per-node (`ppn=4`). The `-j oe` merges standard output and standard error to one output file. The node file path is stored in `NODE_FILE`. The subsequent loop processes each line of this file, extracting the list of cores assigned to that node. The core ID is added to the `cores` array. This array is then used to pass the appropriate cores to the `taskset` command. The example assumes a hypothetical `my_parallel_app` which takes as an argument the number of processes as `-n` and uses a command string to execute the individual tasks. The actual commands launched are controlled by a series of `taskset` commands to ensure each process is bound to the given core, with the `wait` at the end ensuring all background processes complete before the script exits.

Another scenario may involve an MPI application where the MPI launcher (e.g., `mpirun`) handles process placement and binding. This can be handled by carefully configuring MPI's resource management options in conjunction with the nodefile. For example:

```bash
#!/bin/bash
#PBS -l nodes=2:ppn=4
#PBS -N mpi_no_hyperthread
#PBS -j oe

NODE_FILE=$PBS_NODEFILE

# Initialize an array to hold the assigned cores
cores=()

# Read the nodefile line by line, populating the cores array with each CPU's ID
while IFS= read -r line; do
    node=$(echo "$line" | awk '{print $1}')
    cpus=$(echo "$line" | awk '{print $2}' | tr ',' ' ')
    for cpu in $cpus; do
      cores+=("$cpu")
    done
done < "$NODE_FILE"

# Check if we received the expected number of cores.
if [ ${#cores[@]} -ne 8 ]; then
    echo "ERROR: Expected 8 cores, but found ${#cores[@]}." >&2
    exit 1
fi


# Launch an MPI application binding to the cores.
# Modify the command based on MPI environment being used.
# This example assumes an environment that uses 'mpirun -bind-to core'
mpirun --bind-to core -np 8 \
  -x OMP_NUM_THREADS=1 \
  my_mpi_app
```

In this MPI-centric example, we are requesting two nodes, each with four cores, totaling eight cores. The cores are extracted in the same manner, and the number is again verified. The `mpirun` command then is used to start our MPI program, requesting 8 processes, each bound to a core. The exact syntax will vary depending on the installed MPI implementation, and so the `--bind-to core` option will need to be adapted accordingly, for example, `mpirun -map-by core` for OpenMPI based systems. Finally the `OMP_NUM_THREADS=1` environment variable is set, to ensure our program does not use internal threads. This prevents further hyperthreading from within our program itself.

Finally, consider the case where we are launching a series of tasks that will execute sequentially on separate cores. This could occur with a data-processing workflow. In this case, we again make use of `taskset` to bind tasks to specific cores as illustrated below:

```bash
#!/bin/bash
#PBS -l nodes=1:ppn=4
#PBS -N sequential_no_hyperthread
#PBS -j oe

NODE_FILE=$PBS_NODEFILE

# Initialize an array to hold the assigned cores
cores=()

# Read the nodefile line by line, populating the cores array with each CPU's ID
while IFS= read -r line; do
    node=$(echo "$line" | awk '{print $1}')
    cpus=$(echo "$line" | awk '{print $2}' | tr ',' ' ')
    for cpu in $cpus; do
      cores+=("$cpu")
    done
done < "$NODE_FILE"

# Check if we received the expected number of cores.
if [ ${#cores[@]} -ne 4 ]; then
    echo "ERROR: Expected 4 cores, but found ${#cores[@]}." >&2
    exit 1
fi

# Run four independent processes on separate cores using 'taskset'
taskset -c "${cores[0]}" process_command_1 &
taskset -c "${cores[1]}" process_command_2 &
taskset -c "${cores[2]}" process_command_3 &
taskset -c "${cores[3]}" process_command_4 &
wait
```

Here, we request four cores on a single node, and the core list is generated. Then, four distinct commands are launched, each bound to a separate core using `taskset`, and again a `wait` command ensures that all background processes have completed. This method is especially useful when the individual steps within a workflow are not designed for internal parallelization, but we need to ensure each task gets exclusive access to a CPU core.

When managing parallel jobs without utilizing hyperthreading, these principles provide a way to effectively control processor allocation. Key to successful implementation is understanding the environment variables and available commands, such as `taskset`, and adjusting them to fit the specific characteristics of the cluster. Further details concerning usage, and environment specific information, can often be found in the documentation provided with Torque, and the specific implementations of `taskset`, `numactl`, or `mpirun` on the respective system. Additionally, consult guides on process binding and NUMA-aware programming for more in-depth understanding of parallel performance optimization within HPC. Careful planning of core binding strategy and resource requests, combined with testing for performance validation, is key to achieving optimal performance in multi-core HPC environments.
