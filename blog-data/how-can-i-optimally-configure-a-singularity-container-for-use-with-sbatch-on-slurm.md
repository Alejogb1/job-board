---
title: "How can I optimally configure a Singularity container for use with `sbatch` on SLURM?"
date: "2024-12-23"
id: "how-can-i-optimally-configure-a-singularity-container-for-use-with-sbatch-on-slurm"
---

Alright, let's tackle this. From my experience, especially back in the days when we were migrating our astrophysics simulations over to a new cluster, I remember the challenges involved in effectively marrying Singularity containers with SLURM's `sbatch`. It's not always as straightforward as one might hope. The crux of the matter lies in properly configuring the container environment to play nice with SLURM's resource management while maintaining portability and reproducibility. Let's break this down into practical steps, focusing on the aspects that really make a difference.

First, it's paramount to understand that when you invoke `sbatch`, you're essentially handing off a job to a scheduler that manages resources across a cluster. Your Singularity container needs to be configured to be aware of this environment and to seamlessly integrate with it. This boils down to a few key configuration points. The first, and possibly the most critical, is ensuring the container has access to the necessary resources and is executing within the confines of what SLURM has allocated. This often translates to passing the appropriate mount points and environment variables.

A common pitfall is failing to properly manage the scratch space within a SLURM job. By default, many Singularity containers don't inherently understand SLURM's scratch directory setup (often specified via `SLURM_TMPDIR`). This is where things can become complicated. If your container attempts to write to locations outside the allocated workspace, the job can fail or behave unexpectedly. To address this, you should configure the container to use the `SLURM_TMPDIR` variable if it exists within the SLURM environment.

Here's an example of a basic `sbatch` script coupled with a Singularity execution:

```bash
#!/bin/bash
#SBATCH --job-name=my_singularity_job
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --output=my_singularity_job_%j.out
#SBATCH --error=my_singularity_job_%j.err

# Detect if SLURM_TMPDIR is defined.
if [ -n "$SLURM_TMPDIR" ]; then
  export TMPDIR="$SLURM_TMPDIR"
  echo "SLURM_TMPDIR detected: $TMPDIR"
  # In this example the scratch is mounted to /scratch_space/
  SINGULARITY_BINDPATH="$TMPDIR:/scratch_space/"
else
  echo "SLURM_TMPDIR not detected; using default temp folder."
  SINGULARITY_BINDPATH="/tmp:/scratch_space/"
fi

# Execute the Singularity container
singularity exec --bind "$SINGULARITY_BINDPATH" my_container.sif my_application --input /input_data --output /scratch_space/output_data

```

In this script, we are examining if the environment variable `SLURM_TMPDIR` exists, and if so, it is used to set the container's working directory `/scratch_space/`. This ensures that all temporary data is written within SLURM's allocated resources. If it doesn't exist, you can default to a usual `/tmp`. The `--bind` option makes the designated space available to the container. It is critical to ensure the application is using `/scratch_space` as the working directory to write data.

A second, often overlooked aspect is dealing with environment variables. Just because a variable is set in your SLURM environment doesn't mean it will automatically propagate into the container. You need to explicitly define them to be carried over using `-e` or `--env`. Some variables, such as those related to GPU usage or MPI, are incredibly critical.

Let's illustrate with an example where you need to propagate the `CUDA_VISIBLE_DEVICES` variable for GPU-based workloads:

```bash
#!/bin/bash
#SBATCH --job-name=gpu_singularity_job
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=gpu_singularity_job_%j.out
#SBATCH --error=gpu_singularity_job_%j.err

# Propagate CUDA-related environment variables
SINGULARITY_ENV="CUDA_VISIBLE_DEVICES"

singularity exec --nv -e $SINGULARITY_ENV --bind "$TMPDIR:/scratch_space/" my_gpu_container.sif my_gpu_application --input /input_data --output /scratch_space/output_data
```

In the snippet above, the `-e CUDA_VISIBLE_DEVICES` portion ensures that the `CUDA_VISIBLE_DEVICES` environment variable, critical for directing computation to the right GPU on the node, is passed into the container. The `--nv` flag indicates that the container needs access to the NVIDIA drivers on the host node. This allows the application to access the designated GPU(s). Note how the `-e` could also be written as `--env CUDA_VISIBLE_DEVICES`, which produces the same result.

Finally, consider how your application within the container is interacting with files outside of the working directory. It's common that you need to share data between a user directory and the container's environment. This needs careful management and can be achieved via bind mounts as demonstrated in the first script. The problem here is that you must carefully manage these file paths inside your container, making sure that the application is reading and writing to the correct locations. You could also consider using `singularity build --sandbox my_container_sandbox` and building a container directly on the SLURM nodes for a more bespoke integration, but I suggest you only use that if the other suggestions aren't effective.

Here's a more complex example that demonstrates bind mounting of several key locations and demonstrates how to use a pre-existing working directory:

```bash
#!/bin/bash
#SBATCH --job-name=complex_singularity_job
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --output=complex_singularity_job_%j.out
#SBATCH --error=complex_singularity_job_%j.err

# Set the working directory based on $SLURM_SUBMIT_DIR
WORKING_DIR="$SLURM_SUBMIT_DIR"

# Define mount points, where 'input_data' is a folder
SINGULARITY_BINDPATH="$WORKING_DIR/input_data:/input_data,$TMPDIR:/scratch_space,$WORKING_DIR/output_data:/output_data"

# Execute the container
singularity exec --bind "$SINGULARITY_BINDPATH" my_analysis_container.sif  /my_analysis_application --config /input_data/my_config.cfg --output /output_data/results.dat
```

Here we mount an input data directory, the temporary scratch space, and an output data folder. The crucial thing to remember is that paths within the container are now completely relative to where we mounted the folders on the host system. By setting a `WORKING_DIR`, the script ensures it’s working with user specified folder locations.

For those seeking a deeper dive, I'd strongly recommend looking at the official Singularity documentation, which is meticulously maintained and is an excellent resource for the latest best practices. Also, the book "High Performance Computing: Programming and Applications," by David W. Walker, provides a solid foundation in parallel computing concepts that frequently tie into using containerized environments in HPC contexts. Moreover, it would be worth reviewing any documentation produced by your HPC cluster’s maintainers as they might have specific recommendations tailored to their infrastructure. Finally, for a deep look into Slurm’s configuration, explore the official SLURM documentation and related papers from the SchedMD group.

In short, effectively using Singularity containers with SLURM `sbatch` involves careful consideration of environment variables, bind mounts, and an understanding of SLURM's resource allocation strategy. It's a balancing act, ensuring both your application's needs and SLURM's requirements are met. Hopefully, these insights and examples point you in a more efficient direction.
