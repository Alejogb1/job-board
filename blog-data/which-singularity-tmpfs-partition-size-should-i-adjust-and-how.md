---
title: "Which Singularity tmpfs partition size should I adjust and how?"
date: "2024-12-23"
id: "which-singularity-tmpfs-partition-size-should-i-adjust-and-how"
---

Alright, let's tackle this. The issue of tmpfs partition sizing within Singularity containers is one that’s tripped me up a few times, particularly in those high-performance computing environments where unexpected resource limitations can send your job straight to the bit bucket. It’s not as straightforward as just throwing some arbitrary number at it. From what I've seen, and trust me, I've debugged enough Singularity-related issues to fill a small book, the optimal size hinges on several factors specific to the container's workload.

Before we dive into the ‘how’, let’s establish ‘what’ we're dealing with. A tmpfs filesystem resides entirely in memory, meaning it is blazing fast for read/write operations. When Singularity starts a container, by default it creates a tmpfs mount at `/tmp`. This provides a volatile workspace for the container that avoids cluttering the underlying host’s filesystem, a practice born from necessity when dealing with multi-user systems. Now, the default size often isn't sufficient for intensive tasks that generate intermediate files, large temporary datasets, or that require a lot of short-term data manipulation. This limitation is what we need to address when adjusting the size.

The core issue is that a container running out of space within its `/tmp` (or a custom tmpfs mount location) will likely result in errors ranging from applications crashing to jobs failing silently, and this can be maddening when trying to pinpoint the source of a problem. So, how do you figure out the appropriate size, and how do you actually adjust it? There isn't a one-size-fits-all answer, but a methodical approach typically leads to a good outcome.

The methodology I've found most useful involves a bit of iterative testing. Initially, monitor the `/tmp` usage when running your application. If you are using `singularity exec`, simply log into the container environment and periodically check the usage using `df -h /tmp`. If you are using a scheduler, you can incorporate a similar check at the start of a job. This baseline will show you the scale of resources the container needs. From there, we incrementally increase the tmpfs size during container launch, until we see that the container no longer runs out of space.

So, let's talk specifics with concrete examples. We adjust the tmpfs size using the `--tmpdir-size` option when launching Singularity containers using `singularity exec` or `singularity run`.

**Example 1: Basic Increase**

Let's say you've determined via monitoring that the default `/tmp` space is insufficient. Here's how you'd launch a container with a specified size of 8 GB.

```bash
singularity exec --tmpdir-size 8192 image.sif /bin/bash -c "my_application --some-flag --input my_large_data"
```

In this example, the `--tmpdir-size 8192` argument will configure a tmpfs partition with a size of 8192 MB (or 8 GB). It is crucial to note that the value you provide is in *megabytes*. This approach will increase the size of the /tmp directory and its related mount. The rest of the command executes `my_application` within the container, passing it the relevant arguments.

**Example 2: Custom Mount Point**

Sometimes, it's necessary to use a custom mount point. For instance, some applications expect temporary files in locations other than `/tmp`. In such cases, we can create our own tmpfs mounts. This also helps with container organization and management. Let’s assume we want to mount a 4 GB tmpfs at `/scratch` within the container. This requires using `--bind` to mount the tmpfs onto a specific path.

```bash
singularity exec --bind /scratch:$TMPDIR --tmpdir-size 4096 image.sif /bin/bash -c "my_application --some-flag --scratch-dir /scratch --input my_data"
```

In this instance, we are using the command-line equivalent of the singularity option `TMPDIR`, and specifying that the `scratch` folder in the container will be mounted as a tmpfs at 4096 MB (4 GB). The application then should expect the `/scratch` folder to be available and to use it for storing its intermediate files.

**Example 3: Using environment variables**

Finally, let’s explore using environment variables to set the tmpfs size. This can be useful when dealing with batch processing systems or when automating container deployments. This is particularly handy within slurm scripts or similar job managers.

```bash
export SINGULARITY_TMPDIR_SIZE=16384
singularity exec image.sif /bin/bash -c "my_application --another-flag --data input_file"
```

Here, the environment variable `SINGULARITY_TMPDIR_SIZE` dictates the size of the default `/tmp` tmpfs partition when the container starts. In this instance it will be set to 16384 MB or 16 GB. This is often cleaner than specifying the argument to every execution, especially if the application will always need that amount of tmp space. Note that the value set this way will only apply to `singularity exec` and `singularity run` commands within the scope of the shell where the variable is declared.

It is important to understand that tmpfs is a volatile filesystem. This implies that the content of the mount will be erased upon container shutdown. You should never rely on tmpfs as a mechanism for persistent storage.

Now, as for resources, rather than providing specific web links, I recommend the following authoritative references. Start with the official Singularity documentation, it is very comprehensive, although it doesn't go into much detail about the `tmpdir` mount options. To understand tmpfs from the linux perspective, the kernel documentation found on kernel.org is crucial; particularly look for the section about *mount options*, where options like `size` and other relevant details are described. The *man page* for the command `mount` is also highly recommended; it can be accessed from the terminal with the command `man mount`. It includes details about tmpfs mount behavior that are important to know for advanced scenarios. Furthermore, for a deep dive into the technical nuances of memory management within Linux, consider reading *Understanding the Linux Kernel* by Daniel P. Bovet and Marco Cesati. While not solely focused on tmpfs, it provides indispensable context for understanding how the Linux kernel allocates and manages resources which in turn will give a better understanding of how memory allocations affect tmpfs mounts. For a more focused reading, you might want to explore *Linux Kernel Development* by Robert Love which gives a bit more information about filesystem drivers and other related low-level concepts that are fundamental to understanding tmpfs behavior.

In conclusion, adjusting tmpfs size in Singularity requires a clear understanding of application requirements, a methodical approach to testing and adjustment, and the correct use of Singularity's options, whether on the command line or environment variables. I've encountered scenarios where misconfigured tmpfs settings led to significant time wasted debugging, so doing it proactively based on workload is essential. Monitoring, careful experimentation, and reference to the authoritative sources I've recommended will lead you to the optimal configuration for your containerized applications. Remember that tmpfs is volatile; it isn't a long-term storage solution, but rather an excellent high-performance scratch space.
