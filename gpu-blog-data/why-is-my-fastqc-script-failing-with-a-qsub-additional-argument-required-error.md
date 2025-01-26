---
title: "Why is my FastQC script failing with a 'qsub additional argument required' error?"
date: "2025-01-26"
id: "why-is-my-fastqc-script-failing-with-a-qsub-additional-argument-required-error"
---

The "qsub additional argument required" error encountered when submitting a FastQC script via a queuing system like Sun Grid Engine (SGE), or a variant thereof, typically signals a deficiency in how the resource request is formatted for the queuing system, rather than an inherent issue with the FastQC application itself. I've encountered this exact situation multiple times while managing high-throughput sequencing pipelines, and it almost always stems from incomplete or missing directives within the `qsub` command. Essentially, the queuing system requires explicit instructions regarding the resources your job needs (e.g., memory, cores, runtime). If these are not provided correctly, `qsub` cannot properly schedule the job and throws this error. The error itself is non-specific because the required argument or arguments are dependent upon the system configuration.

The primary cause isn't that FastQC *needs* these arguments, but that the queuing system, which is intermediary between your script execution and the compute resources, needs them. FastQC, when invoked directly from the command line, functions based on the environment’s available resources; it will use what it finds. When submitting to a queuing system, this automatic allocation is bypassed. Instead, the scheduler needs to be told explicitly what resources are required by each job to ensure proper distribution and prevent over- or underutilization of available compute power. Think of the queuing system as a sophisticated resource manager rather than just a script launcher; it needs a complete set of instructions to function correctly.

Here's a breakdown of what typically goes wrong and how to address it, coupled with representative code examples:

**Problem 1: Missing Memory Allocation:**

A common oversight is forgetting to specify the amount of memory needed for the FastQC process. FastQC can be relatively memory intensive when processing larger datasets, especially paired-end reads. If you fail to tell the queuing system how much memory is required, it will often reject the job submission with this error, or, worse, allow it to start with insufficient allocation leading to process termination and job failure.

```bash
#!/bin/bash
# Incorrect: Missing memory allocation
#$ -cwd
#$ -S /bin/bash

FASTQC_BIN="/path/to/fastqc"
INPUT_FILE="$1"
OUTPUT_DIR="${INPUT_FILE%.*}"

mkdir -p "$OUTPUT_DIR"
"$FASTQC_BIN" -o "$OUTPUT_DIR" "$INPUT_FILE"
```

The above script will almost certainly fail when submitted via `qsub` because it doesn't provide any memory requirements. Here is a corrected version:

```bash
#!/bin/bash
# Corrected: Memory allocation added
#$ -cwd
#$ -S /bin/bash
#$ -l mem_free=4G # Request 4 GB of memory

FASTQC_BIN="/path/to/fastqc"
INPUT_FILE="$1"
OUTPUT_DIR="${INPUT_FILE%.*}"

mkdir -p "$OUTPUT_DIR"
"$FASTQC_BIN" -o "$OUTPUT_DIR" "$INPUT_FILE"
```

Here, the line `#$ -l mem_free=4G` tells SGE to allocate at least 4 gigabytes of memory for this specific job. The specific flag (`mem_free` in this instance) and its value may vary based on the specific queuing system you are using. It is imperative to use the appropriate flag for your particular system.

**Problem 2: Insufficient Time Allocation:**

Another overlooked aspect is the time limit for the job. If a FastQC job runs for a longer time than what was allocated, it will be terminated by the system, but the initial error often lies in failing to provide this time limit. The 'qsub additional argument required' is a common response when time isn't specified. Consider this scenario:

```bash
#!/bin/bash
# Incorrect: Missing run time limit.
#$ -cwd
#$ -S /bin/bash
#$ -l mem_free=4G

FASTQC_BIN="/path/to/fastqc"
INPUT_FILE="$1"
OUTPUT_DIR="${INPUT_FILE%.*}"

mkdir -p "$OUTPUT_DIR"
"$FASTQC_BIN" -o "$OUTPUT_DIR" "$INPUT_FILE"
```

To address this, a time constraint should be added:

```bash
#!/bin/bash
# Corrected: Run time limit added
#$ -cwd
#$ -S /bin/bash
#$ -l mem_free=4G
#$ -l h_rt=01:00:00 # Request 1 hour of run time

FASTQC_BIN="/path/to/fastqc"
INPUT_FILE="$1"
OUTPUT_DIR="${INPUT_FILE%.*}"

mkdir -p "$OUTPUT_DIR"
"$FASTQC_BIN" -o "$OUTPUT_DIR" "$INPUT_FILE"
```

The line `#$ -l h_rt=01:00:00` instructs the queuing system to allocate a run time of one hour. Again, the flag and syntax will vary by system. If you routinely have long-running FastQC jobs, it's best practice to over-allocate time, rather than under, to prevent unexpected job terminations, though an overly large time allocation can negatively impact overall scheduling.

**Problem 3: Improper Core Request:**

FastQC is, by default, single-threaded. While it does not inherently require multicore processing, you might choose to utilize parallel processing using multiple FastQC instances to process multiple files simultaneously. You still must, however, request the resources from the queuing system if they are to be used. The 'qsub additional argument required' will still manifest if these are not included.

```bash
#!/bin/bash
# Incorrect: Missing core request
#$ -cwd
#$ -S /bin/bash
#$ -l mem_free=4G
#$ -l h_rt=01:00:00

FASTQC_BIN="/path/to/fastqc"
INPUT_FILE="$1"
OUTPUT_DIR="${INPUT_FILE%.*}"

mkdir -p "$OUTPUT_DIR"
"$FASTQC_BIN" -o "$OUTPUT_DIR" "$INPUT_FILE"
```

Let's assume we want to run multiple FastQC instances simultaneously on different files. In that case, you will need to request the number of cores required. It can also be advantageous to request a single core even for single-file analysis as this will explicitly make that resource available to the script via the queuing system. For this example, let's request 1 core:

```bash
#!/bin/bash
# Corrected: Requesting 1 core
#$ -cwd
#$ -S /bin/bash
#$ -l mem_free=4G
#$ -l h_rt=01:00:00
#$ -l threads=1

FASTQC_BIN="/path/to/fastqc"
INPUT_FILE="$1"
OUTPUT_DIR="${INPUT_FILE%.*}"

mkdir -p "$OUTPUT_DIR"
"$FASTQC_BIN" -o "$OUTPUT_DIR" "$INPUT_FILE"
```

The `#$ -l threads=1` line requests one core for this specific process. The nomenclature and flag are, as previously stated, system-dependent. For example, SGE systems may use `slots` while others use `ncpus`. If the job uses any form of parallelization, then the required number of cores must be requested.

In addition to memory, time, and cores, other common scheduling options exist such as defining specific queues, email notifications, or handling job dependencies.

**Resource Recommendations**

To fully understand and resolve this error, I advise consulting the following resources:

1. **Your local queuing system documentation:** This is the single most important resource. The syntax and arguments for the `qsub` command, as well as the specific flags needed for memory, time, and core allocation, are detailed in your system's documentation.
2. **The queuing system's administrator:** If the documentation is unclear or incomplete, consulting your system's administrator is often the most direct way to gain an understanding of how the scheduling system is configured and how to correctly request resources. They will be able to provide specifics on queue names, appropriate limits, and required system-specific parameters.
3. **Online forums and knowledge bases specific to your queuing system:** There are often community forums or knowledge bases for specific queuing system types. Users frequently post questions and solutions that can provide invaluable information when troubleshooting issues such as this.

By addressing the resource requirements of your FastQC job within the `qsub` submission script, and by consulting the relevant documentation and experts, you should be able to eliminate the "qsub additional argument required" error and successfully run your analyses. Failure to properly configure the queuing system's resource requests will lead to numerous errors and hinder the efficient use of the cluster’s compute capacity.
