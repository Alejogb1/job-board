---
title: "How to run Python scripts using Slurm?"
date: "2025-01-30"
id: "how-to-run-python-scripts-using-slurm"
---
In my experience managing high-performance computing clusters, effectively using Slurm to execute Python scripts is crucial for scalable and reproducible research. The core challenge lies in bridging the gap between interactive Python development and the batch-oriented nature of Slurm job scheduling. Understanding Slurm's resource allocation and job submission process is paramount to achieving efficient execution of Python workloads.

Specifically, Slurm, a widely used workload manager, requires a batch script instructing it on how to allocate resources (CPU, memory, GPU) and execute specific commands. This script, often written in bash, serves as an intermediary between the user's intent and Slurm's resource management system. The Python script itself is generally unmodified, focusing purely on the computational task, while the batch script manages its environment and execution parameters.

The process typically involves several key steps: writing the Python script, crafting the Slurm batch script, submitting the job, and monitoring its progress. In cases where multiple Python scripts need to be executed simultaneously or as a pipeline, the batch script becomes increasingly complex, requiring careful consideration of dependencies and output management. Misconfigurations within the Slurm script can lead to failed jobs, resource contention, or underutilization of allocated resources, making it essential to adhere to best practices.

Here are three practical examples illustrating how to run Python scripts using Slurm, alongside explanations:

**Example 1: Basic Python Script Execution**

Imagine a simple Python script, `my_script.py`, that calculates a sum:

```python
# my_script.py
import sys

def calculate_sum(a, b):
  return int(a) + int(b)

if __name__ == "__main__":
  if len(sys.argv) != 3:
    print("Usage: python my_script.py <integer1> <integer2>")
    sys.exit(1)
  
  num1 = sys.argv[1]
  num2 = sys.argv[2]
  result = calculate_sum(num1, num2)
  print(f"The sum is: {result}")
```

This script accepts two integers as command-line arguments. The associated Slurm batch script, `run_my_script.sh`, would be:

```bash
#!/bin/bash
#SBATCH --job-name=python_sum
#SBATCH --output=output_%j.txt
#SBATCH --error=error_%j.txt
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=100M

module load python/3.9  # Or your desired Python version

python my_script.py 5 10
```

*   `#!/bin/bash`: Indicates that the script should be interpreted by bash.
*   `#SBATCH ...`: These lines specify Slurm directives.
*   `--job-name`: Assigns a name to the job.
*   `--output`: Specifies the path for standard output. `%j` is replaced with the job ID.
*   `--error`: Specifies the path for standard error.
*   `--time`: Sets the time limit for the job.
*   `--cpus-per-task`: Requests the number of CPUs.
*   `--mem`: Requests the amount of memory.
*   `module load python/3.9`: Loads the necessary Python environment. This assumes the cluster uses environment modules.
*   `python my_script.py 5 10`:  The command that executes the Python script, providing 5 and 10 as arguments.

To submit this job, one would execute `sbatch run_my_script.sh` in the terminal. The output and error files can then be checked for results and any issues.

**Example 2:  Python Script with Input File**

Consider a slightly more complex scenario where a Python script processes data from an input file. Let's assume `process_data.py`:

```python
# process_data.py
import sys
import csv

def process_csv(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        for row in reader:
            processed_row = [int(x) * 2 for x in row] # Sample processing logic
            writer.writerow(processed_row)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python process_data.py <input.csv> <output.csv>")
        sys.exit(1)
    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    process_csv(input_csv, output_csv)
```

Assume an input file called `input.csv` exists with comma-separated numbers. The corresponding Slurm script, `run_process_data.sh`, could look like:

```bash
#!/bin/bash
#SBATCH --job-name=process_data
#SBATCH --output=output_%j.txt
#SBATCH --error=error_%j.txt
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=500M

module load python/3.9 # Or your desired Python version

python process_data.py input.csv output.csv
```

This example shows how to pass file paths as command-line arguments. The Python script then reads from `input.csv`, processes the content, and writes to `output.csv`. Again, `sbatch run_process_data.sh` submits the job.

**Example 3: Python Script with Multiple Processes**

In certain cases, especially with computationally intensive tasks, it's beneficial to leverage Slurm's support for parallel processing. While Python's inherent global interpreter lock (GIL) can limit true multithreading within a single process, multiple Python instances can run simultaneously using Slurm's array jobs. Here's a modified script `process_array.py` (simplified for illustration) which now accepts an index:

```python
# process_array.py
import sys
import time

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python process_array.py <index>")
        sys.exit(1)
    index = int(sys.argv[1])
    print(f"Processing index: {index}")
    time.sleep(10) # Simulate some processing time
    print(f"Completed index: {index}")
```

The corresponding Slurm script, `run_array.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=process_array
#SBATCH --output=output_%j_%a.txt
#SBATCH --error=error_%j_%a.txt
#SBATCH --time=00:15:00
#SBATCH --array=1-10
#SBATCH --cpus-per-task=1
#SBATCH --mem=200M

module load python/3.9

python process_array.py $SLURM_ARRAY_TASK_ID
```

*   `#SBATCH --array=1-10`:  This indicates an array job with indexes from 1 to 10.
*   `%a` is substituted with the array task ID in the output and error files.
*   `$SLURM_ARRAY_TASK_ID` is a Slurm environment variable containing the current array task index.

Submitting this job with `sbatch run_array.sh` will execute ten instances of the `process_array.py` script concurrently, each with a different index.

For further exploration of effective resource management when using Slurm, the following resources are helpful. Publications focusing on high-performance computing best practices are valuable. Consider consulting user documentation for specific Slurm installations, as there can be subtle variations across different environments. Introductory texts on parallel computing and distributed systems will also build a solid foundation for efficient utilization. Specifically, documentation on job arrays and resource allocation settings will further improve your use of Slurm for running Python jobs. These, coupled with practical experience, are the key to mastering Python script execution on high performance clusters.
