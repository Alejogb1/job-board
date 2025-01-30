---
title: "How can SLURM script configuration parameters be saved to the output file?"
date: "2025-01-30"
id: "how-can-slurm-script-configuration-parameters-be-saved"
---
The core challenge in capturing SLURM script configurations within the output file stems from the inherent separation of script execution and environment persistence. SLURM, by design, orchestrates jobs as distinct processes, and while the script itself defines parameters, these are typically evaluated at submission time and are not inherently preserved beyond the initial sbatch command. To bridge this gap, we must explicitly capture and record the desired configurations within the script itself, directing them to standard output or a specified file. This requires careful scripting, using techniques beyond direct SLURM commands. Over my years of managing HPC clusters, I've found this approach essential for reproducible research and debugging.

**Explanation**

The standard behavior of SLURM does not automatically include the script’s configuration parameters within the job's output files. The script is interpreted by the system to determine allocation requirements (nodes, tasks, memory), and these specifications are processed by the SLURM scheduler. The script variables and other parameters such as options defined via the `#SBATCH` directives are used during job scheduling and are not directly printed in standard output or saved in any persistent manner automatically. 

Therefore, direct manipulation is needed to capture this information. We achieve this by:

1.  **Extracting Parameters from SLURM Environment Variables:** SLURM exposes various parameters through environment variables, such as `$SLURM_JOB_ID`, `$SLURM_NNODES`, `$SLURM_NTASKS`, and others. These can be explicitly printed to standard output or redirected to a file.

2.  **Printing `#SBATCH` Directives:** The most robust approach is to echo the content of the script itself, focusing on the `#SBATCH` directives. We can use `grep` or similar tools to filter these directives, ensuring we capture the precise parameters defined by the user at submission.

3.  **Capturing Custom Script Variables:** Any variables that the user sets within the script and is important to record, should be explicitly outputted to the standard output.

4.  **Redirection of Output:** Standard output can be redirected to a specific file within the script. This allows for keeping the standard output clean, in case it is needed for processing the program or to directly record the configuration and other information.

Using these methods, we gain the capability to explicitly save the relevant details of a job's configuration, contributing to greater transparency and reproducibility. This data becomes invaluable during post-hoc analyses and when revisiting results.

**Code Examples**

*   **Example 1: Outputting SLURM environment variables:**

```bash
#!/bin/bash
#SBATCH --job-name=env_vars_example
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=00:10:00
#SBATCH --output=env_vars_out.txt

echo "--- SLURM Environment Variables ---"
echo "Job ID: $SLURM_JOB_ID"
echo "Number of Nodes: $SLURM_NNODES"
echo "Number of Tasks: $SLURM_NTASKS"
echo "Task per node: $SLURM_TASKS_PER_NODE"
echo "Job Submission Directory: $SLURM_SUBMIT_DIR"
echo "Job Working Directory: $SLURM_JOB_WORKING_DIR"
echo "---"

# Placeholder computation
sleep 60
```

*Commentary:* This script utilizes environment variables set by SLURM to output to standard output, which because of the `#SBATCH --output=env_vars_out.txt` directive will be recorded in the file “env_vars_out.txt”. This provides a readily accessible log of crucial allocation details, including the job ID, allocated nodes, and tasks. The placeholders are important to ensure the job is scheduled to a cluster and the environment variables are available and exported to be seen by the output.

*   **Example 2: Extracting and outputting `#SBATCH` directives:**

```bash
#!/bin/bash
#SBATCH --job-name=sbatch_options_example
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=sbatch_options_out.txt

echo "--- SBATCH directives ---"
grep "^#SBATCH" $0 | sed 's/^#SBATCH //g'
echo "---"

# Placeholder computation
sleep 60
```
*Commentary:* This script leverages `grep` to locate all lines beginning with `#SBATCH` within the script itself (`$0`). Then `sed` is used to remove the `#SBATCH` prefix to output the directives only. The output is recorded in the sbatch_options_out.txt file.  This captures the exact directives set by the user, regardless of implicit defaults, making it very effective to know the exact parameters of the job at runtime.

*   **Example 3: Saving Custom Script Variables:**

```bash
#!/bin/bash
#SBATCH --job-name=custom_vars_example
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
#SBATCH --output=custom_vars_out.txt

# Custom variables
my_param="value123"
data_file="input_data.dat"
iterations=1000

echo "--- Custom Script Variables ---"
echo "my_param: $my_param"
echo "data_file: $data_file"
echo "iterations: $iterations"
echo "---"

# Placeholder computation
sleep 30
```
*Commentary:* This script shows how to output custom variables that may be needed for recording the configurations of the script. These values can be important for ensuring reproducibility of the data analysis or simulations performed by the script. They are not available from SLURM environment variables, or `#SBATCH` directives and need to be added explicitly as shown here. Similarly to previous examples, this output will be saved in the specified output file.

**Resource Recommendations**

Several resources can provide detailed information regarding SLURM and bash scripting techniques for handling these tasks.

*   **Official SLURM Documentation:** The formal SLURM documentation is indispensable for precise information on command options, environment variables, and scheduling behaviors. It’s typically available on any HPC cluster environment. Specific focus should be placed on the `sbatch` command documentation and the explanations of environment variables.

*   **Bash Scripting Tutorials:** Numerous online resources offer comprehensive bash scripting tutorials, covering topics such as string manipulation, variable usage, file I/O, and command redirection. This is useful for advanced manipulation of the output.

*   **HPC Cluster Support Guides:**  Most HPC facilities provide user guides specific to their systems. These guides often include practical advice, including examples of script configurations. These are essential for ensuring scripts are tailored to the local hardware.

By combining these techniques, one can reliably save crucial SLURM script configuration parameters to output files, which enhances transparency, traceability, and debugging capabilities, crucial aspects for scientific computing and data analysis.
