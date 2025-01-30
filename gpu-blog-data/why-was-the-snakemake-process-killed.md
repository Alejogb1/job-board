---
title: "Why was the Snakemake process killed?"
date: "2025-01-30"
id: "why-was-the-snakemake-process-killed"
---
A frequent, often frustrating, occurrence in complex bioinformatics pipelines implemented with Snakemake is the unexpected termination of processes, leaving users with cryptic error messages or no message at all. These kills, commonly signaled by a `SIGKILL` or `SIGTERM`, are not necessarily indicative of a problem within the Snakemake workflow itself but rather a symptom of resource contention, primarily due to memory limitations or imposed time constraints from the underlying execution environment. Having managed several large-scale genomic analyses using Snakemake across various high-performance computing (HPC) clusters, I’ve consistently observed these resource-induced terminations and have developed strategies to mitigate them.

The underlying reason for a Snakemake process being killed is almost always tied to the limitations placed by the job scheduler or the operating system’s resource management. When Snakemake launches a job, it effectively wraps a shell command or Python function within a process. This process then competes for resources - primarily memory and CPU - with other processes running on the same node. If the process exceeds the defined limits, either implicitly (default system limits) or explicitly (through resource requests configured in the Snakemake file), the system will forcefully terminate the process to maintain overall system stability.

Specifically, a `SIGKILL`, often seen as `-9` in job logs, signals the operating system's kernel forcibly terminated the process. This usually indicates the process has consumed an unacceptably high amount of memory and is deemed a threat to the system's ability to execute other tasks. Conversely, a `SIGTERM`, usually `-15`, is a more graceful kill signal sent by a job scheduler. Here, the scheduler decided to terminate the job for a reason such as time expiration or resource limits that were approached but not yet breached. The difference is important: a `SIGKILL` indicates a sudden, unavoidable termination, whereas a `SIGTERM` allows a running process to perform some clean up tasks.

In my experience, several common scenarios lead to Snakemake process kills. Insufficient memory allocation during a compute-intensive step (e.g., sequence alignment or variant calling) is a prime culprit. Unrealistic time constraints can also prematurely terminate a job even though the code might run correctly given more time. Incorrect specification of resource needs within the Snakemake configuration leads to processes demanding more than the requested allocation. Also, errors in the actual Python/shell code within the job's execution can sometimes exacerbate resource consumption. For instance, an unexpected infinite loop in a script could lead to a rapid increase in memory usage, resulting in an out-of-memory condition.

Below are three examples with code snippets and detailed commentary, illustrating these points:

**Example 1: Insufficient Memory Allocation**

This scenario demonstrates a common error when aligning large sequence files using a memory-intensive aligner (e.g. `bwa-mem`). The default Snakemake settings often assume small datasets, leading to process termination when confronted with substantial memory needs.

```python
# snakemake_workflow/Snakefile

rule align_reads:
    input:
        fastq="reads.fastq"
        ref="genome.fasta"
    output:
        bam="aligned.bam"
    threads: 8
    shell:
        "bwa mem -t {threads} {input.ref} {input.fastq} | samtools view -Sb - > {output.bam}"
```

In this case, if `reads.fastq` is large, the `bwa mem` command might exceed default memory limits. Often, I’ve observed the jobs fail without proper exit messages and job logs will only indicate `SIGKILL` (`-9`). The issue isn't the `bwa mem` command itself but rather the limited memory granted to the job by default, especially on shared HPC environments.

To fix this, explicitly request memory.

```python
# snakemake_workflow/Snakefile

rule align_reads:
    input:
        fastq="reads.fastq"
        ref="genome.fasta"
    output:
        bam="aligned.bam"
    threads: 8
    resources:
        mem_mb=32000
    shell:
        "bwa mem -t {threads} {input.ref} {input.fastq} | samtools view -Sb - > {output.bam}"
```

Here, I explicitly define `resources: mem_mb=32000`, requesting 32GB of memory for this job. Depending on the specific HPC cluster or execution environment, the proper units and keywords might change (e.g., `mem_gb` or `mem`). This explicit specification allows the scheduler to allocate enough memory to the job, preventing termination due to memory over-consumption.

**Example 2: Insufficient Time Limit**

This situation arises frequently with complex analyses like genome assembly or variant calling, where execution time is unpredictable due to varied data sizes or inherent algorithmic complexities.

```python
# snakemake_workflow/Snakefile

rule variant_calling:
    input:
        bam="aligned.bam",
        ref="genome.fasta"
    output:
        vcf="variants.vcf"
    shell:
        "gatk HaplotypeCaller -R {input.ref} -I {input.bam} -O {output.vcf}"

```

Initially, a complex `gatk` workflow might appear to run indefinitely, until the scheduler kills it. Even though I request substantial memory, failure occurs due to an implicit time limit defined by the batch environment. This termination will be signaled with a `SIGTERM` (`-15`). This indicates that the job scheduler sent this kill signal because a defined job time limit had been reached.

To address this, it's essential to configure an adequate runtime limit:

```python
# snakemake_workflow/Snakefile

rule variant_calling:
    input:
        bam="aligned.bam",
        ref="genome.fasta"
    output:
        vcf="variants.vcf"
    resources:
        runtime= "72:00:00"
    shell:
        "gatk HaplotypeCaller -R {input.ref} -I {input.bam} -O {output.vcf}"
```

By specifying `resources: runtime= "72:00:00"`, I set a 72-hour time limit for the job. This is usually interpreted by job scheduler software. This explicit time request prevents premature job terminations, allowing for complex jobs to run to completion. The specific time format (HH:MM:SS) will vary slightly depending on the underlying scheduler, some might prefer a time expressed in minutes. The `runtime` parameter in Snakemake is ultimately a pass-through to the scheduler.

**Example 3: Incorrect Resource Requests**

Finally, resource-request errors are often linked to inconsistencies between what a job truly needs and what has been requested within a `cluster.json` file used by Snakemake when running in cluster mode.

```json
# snakemake_workflow/cluster.json
{
    "__default__":{
        "mem_mb":1000,
        "threads": 1
    }
}
```

If the `Snakefile` does not explicitly request resources, the default values in `cluster.json` are used. This is problematic when default values are insufficient. For example:

```python
# snakemake_workflow/Snakefile

rule process_data:
    input:
        data="input.txt"
    output:
        processed="output.txt"
    shell:
        "python process.py {input.data} {output.processed}"
```

In this case, If the `process.py` script is resource intensive, using the default values in `cluster.json` will result in out-of-memory errors. The job will be killed with `SIGKILL`.
To rectify this, it’s essential to override the defaults in the `cluster.json` at the rule level in the `Snakefile`.

```python
# snakemake_workflow/Snakefile

rule process_data:
    input:
        data="input.txt"
    output:
        processed="output.txt"
    resources:
        mem_mb=8000,
        threads= 4
    shell:
        "python process.py {input.data} {output.processed}"
```

Here, I overrode the default resource limits set in `cluster.json` by specifying a `mem_mb` of 8000 and 4 threads. This approach allows resource allocation to be handled on a rule-by-rule basis. If resources are not specified in the `Snakefile` a process running on a cluster with a `cluster.json` configuration will default to the defined resources in that file, which may not be optimal or even adequate.

Beyond the code and specific examples mentioned, it’s crucial to consult the job scheduler's documentation to fully understand how resources are defined and requested. This will involve an understanding of scheduling options that are specific to each type of HPC cluster such as SLURM, SGE, or LSF. These resources can involve more complex requests than just memory or CPU, but involve specific hardware on particular nodes (e.g. GPU’s). Additionally, understanding Linux system administration and monitoring tools can be beneficial for identifying resource issues as they arise. Tools like `top`, `htop`, or job monitoring systems within the HPC environment are critical in diagnosing resource use during code execution. Using `sacct` or `sstat` is also useful for understanding job resource usage in SLURM based clusters. In summary, process terminations are a direct consequence of resource limitations. Employing strategies to monitor resource usage, requesting sufficient resources in Snakemake, and having a clear understanding of underlying job schedulers and Linux system administration practices can mitigate these issues and ensure successful execution of large-scale workflows.
