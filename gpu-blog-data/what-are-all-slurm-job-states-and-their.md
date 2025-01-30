---
title: "What are all SLURM job states and their corresponding sacct output?"
date: "2025-01-30"
id: "what-are-all-slurm-job-states-and-their"
---
SLURM's job state tracking mechanism is fundamentally reliant on a finite state machine.  Understanding this inherent structure is crucial for interpreting `sacct` output, as the reported states are direct reflections of the job's progression through this machine.  My experience managing large-scale HPC clusters, particularly those employing SLURM 17 and later, has reinforced the importance of precise state interpretation for efficient resource allocation and troubleshooting.  Misinterpreting a state can lead to incorrect conclusions about job failures or resource contention.


The `sacct` command provides a comprehensive view of a job's lifecycle, presenting the job's state at various points. While the exact output formatting can be customized, the core state identifiers remain consistent.  These states, representing distinct phases in a job's execution, are not always mutually exclusive; a job might transition between them multiple times during its lifecycle.  Let's examine the key states and their typical `sacct` representations:


**1. Pending States:**  These states reflect jobs waiting for resources.  The specific pending state often indicates the reason for the delay.

* **`PENDING`:** This is a general pending state, implying the job is queued but awaiting resources.  This is often the initial state upon submission.  `sacct` typically shows this as simply `PENDING`.

* **`PENDING:BOOT_FAIL`:** This denotes a failure to boot a node required by the job.  This might result from hardware issues or network problems on the target node(s).  `sacct` output would show `PENDING:BOOT_FAIL`.  Addressing this often requires examining node-specific logs and potentially restarting affected nodes.

* **`PENDING:CONFIGURING`:**  The job is awaiting configuration of its allocated resources, a less common state often associated with complex resource requests.  The `sacct` output will contain `PENDING:CONFIGURING`.  This usually resolves quickly, but prolonged instances could point to configuration issues within the cluster.

* **`PENDING:PREEMPTED`:**  The job was preempted, meaning it lost its allocated resources to a higher-priority job.  `sacct` will show `PENDING:PREEMPTED`. This indicates a scheduling policy conflict which might necessitate adjustments to job priorities or resource allocation strategies.

* **`PENDING:RESOURCE_UNAVAILABLE`:**  Indicates that the requested resources (e.g., specific nodes, memory, cores) are currently unavailable within the cluster. This is a frequent state, reflecting resource scarcity.  `sacct` displays `PENDING:RESOURCE_UNAVAILABLE`. Analyzing cluster resource utilization through tools like `scontrol show nodes` helps diagnose these bottlenecks.



**2. Running States:** These represent active job execution.

* **`RUNNING`:**  The job is currently executing on allocated resources.  `sacct` shows a simple `RUNNING`.  Monitoring this state usually involves performance metrics obtained from other tools, not directly from `sacct`.

* **`RUNNING:REQUEUE`:**  Although technically running, this implies the job has been requeued, possibly due to a minor interruption that didn't result in a complete failure.  `sacct` output will show `RUNNING:REQUEUE`.  Careful analysis is needed to determine the root cause of the requeue event.


**3. Completed States:**  These states signify successful or unsuccessful job termination.

* **`COMPLETED`:** The job finished successfully.  `sacct` simply indicates `COMPLETED`.  This is the desired final state.

* **`CANCELLED`:** The job was explicitly cancelled by the user or the system.  `sacct` shows `CANCELLED`.  Inspecting job submission scripts and SLURM logs might reveal the cancellation reason.

* **`FAILED`:** The job terminated abnormally, indicating an error during execution. `sacct` displays `FAILED`.  Examining the job's standard error output (`stderr`) is critical for identifying the failure cause.


**4. Suspended States:**  These states indicate temporary job pauses.

* **`SUSPENDED`:**  The job has been paused, usually due to system intervention or user action. `sacct` indicates `SUSPENDED`.  This requires investigating the reason for suspension through SLURM logs or administrative notices.

* **`TIMEOUT`:** The job exceeded its allocated runtime and was terminated.  `sacct` shows `TIMEOUT`.  Adjusting the time limit in the job submission script might be necessary.


**5. Other States:**  Additional states might be encountered, dependent on SLURM configuration and plugins.


**Code Examples and Commentary:**


**Example 1: Basic `sacct` Usage**

```bash
sacct -u $USER -j <job_id>
```

This command displays the accounting information for a specific job identified by `<job_id>`, filtered by the current user.  The output will include the job's state at different stages.  Replacing `<job_id>` with the actual job ID is crucial.  The `-u $USER` option restricts the output to jobs submitted by the current user.  This simple command is ideal for quick checks of individual job statuses.


**Example 2: Filtering for Failed Jobs**

```bash
sacct -u $USER -j <job_id> --format State
```

This command uses the `--format` option to only display the state of a specific job. This is beneficial when dealing with numerous jobs. Focusing on the `State` field allows easy identification of failures or other non-COMPLETED states.  It streamlines the process of isolating problematic jobs within a large workload.


**Example 3:  Retrieving States for All Jobs Submitted in the Last Day**

```bash
sacct -u $USER --starttime=yesterday --format State,JobID,User
```

This command retrieves information about all jobs submitted by the current user within the last 24 hours. The use of `--starttime` refines the search, and the `--format` option selects relevant columns: `State`, `JobID`, and `User`. This command aids in monitoring recently submitted jobs and identifying any potential issues within a specific timeframe.


**Resource Recommendations:**

SLURM documentation, specifically the sections covering job states and the `sacct` command.  The SLURM administrator's guide provides valuable insights into cluster management and troubleshooting.  Consulting with system administrators familiar with your cluster's configuration is recommended for complex issues.  Understanding the underlying scheduling algorithms used by SLURM is also highly beneficial.
