---
title: "Why is singularity container B not found?"
date: "2025-01-30"
id: "why-is-singularity-container-b-not-found"
---
The root cause of a "Singularity container B not found" error often stems from inconsistencies between the requested container name or path and the actual stored container image location or name within the file system or the configured Singularity cache. My experience managing HPC clusters has shown this issue is less about inherent Singularity failure and more about precise container identification and environment configuration. Let's delve into the specifics.

When executing a `singularity run` or similar command, Singularity needs to resolve the given container specifier to a concrete container image. This process involves several layers of lookups: the local file system, any configured container registries, and potentially a local cache. Failure at any of these stages leads to the "not found" error. The error message itself is a relatively generic signal that the resolution process has failed; it doesn’t always pinpoint exactly where the problem resides. It's important to approach this error methodically, examining the input, the environment, and expected outcomes to identify the disconnect.

The simplest case involves a typo in the container name or path provided in the Singularity command. For instance, if you intended to run a container called `my_container.sif`, but mistyped it as `mycontainer.sif`, you'd encounter this error. Likewise, if the specified path, perhaps `/home/user/containers/my_container.sif`, is incorrect due to a misconfiguration or incorrect relative path, Singularity won’t find the image. Absolute and relative paths are interpreted literally, so discrepancies there are very often the problem. It’s important to confirm that the path exists exactly as it is specified.

Beyond simple typos, another frequent cause relates to container location and caching. Singularity, by default, caches container images downloaded from container registries like Docker Hub. If you download a container named `docker://ubuntu:latest` and it is cached locally, it's stored based on a hashing algorithm, and not directly as 'ubuntu:latest'. Subsequent runs using the `docker://ubuntu:latest` specifier will use this cached image unless otherwise configured or invalidated. However, manipulating this cache directly, for example manually deleting a cached image, can cause the "not found" error. Even when a cached image exists, Singularity may be configured to only look at the local file system in a particular execution environment or it might not find the cached image because of permissions or because a different user or the root user downloaded it. In addition, singularity has settings, like using `--no-cache`, that can affect whether it would use a cache at all. This is a less likely scenario for simple "not found" cases but crucial to consider if the user is working in environments with complex caching setups or custom Singularity configurations.

The third commonly encountered scenario pertains to named or path-specified images that Singularity cannot access. If the container image is not present at the specified location, the program will throw this error. In particular, if the container image is stored on a network file system, permission issues can lead to this error. Likewise, a container image on a mounted volume which becomes unmounted will produce the same behavior. In such cases, it isn't that the container image was never found, rather the current state of system does not include the given path and its associated file. The same thing occurs if the user has specified a container image as a uri, but that uri is not accessible. Similarly, a container image located on an external storage device which has become detached would also not be found.

Let's examine some specific examples.

**Example 1: Simple Typo**

```bash
# Correct command using the correct file name
singularity run my_container.sif

# Incorrect command with a typo in the filename, which will fail
singularity run myconainer.sif
```

Here, the second command will likely lead to the “not found” error, assuming no container image named `myconainer.sif` exists. The error message is a direct consequence of the misspelling. This simple example underscores the importance of meticulous command-line input. The user has made no other assumptions here. It is solely a naming issue and has no other cause, if only one container image exists within the working directory.

**Example 2: Incorrect Path**

```bash
# Assuming the container exists at /home/user/containers/my_container.sif
# This command will work, assuming the path is correct and the user has permission.
singularity run /home/user/containers/my_container.sif

# This command will fail, because no container image exists in the current directory
singularity run my_container.sif

# Assuming a current working directory is /home/user/tmp, and the following will fail, 
# because my_container.sif does not exist in /home/user/tmp/containers/
singularity run containers/my_container.sif

# This will also fail, assuming the directory home/user/my_containers does not exist.
singularity run /home/user/my_containers/my_container.sif
```

In this example, the first command will execute, assuming all preconditions are met, while the rest will throw a "not found" error since the path to the container is not correct. Either a relative path is wrong or an absolute path does not match the actual location, or the container has been moved, deleted or renamed. The first example will run as long as the container image exists at that location, and that the user has permission to access the given location. The next three commands will fail, as the program will not find the image file, even if it was present in another location in the system.

**Example 3: Cache Issues**

```bash
# Initial pull of the container, it will be cached and work the first time
singularity pull docker://ubuntu:latest

# Subsequent run will rely on the cached image and will work
singularity run docker://ubuntu:latest

# Removing cache for demonstration purposes, not recommended for actual workflows
# This command is a placeholder, as the commands for deleting cache vary
# This assumes that user is deleting an image from the cache that matches the hash of ubuntu:latest, and this will create error if not done with caution
#  singularity cache clean --all
#   the actual command would vary on the system
# Now the next run will generate "container not found" error because cache is deleted
# singularity run docker://ubuntu:latest
```

This example demonstrates how caching behavior can influence the "not found" error. The first `singularity pull` downloads and caches the Ubuntu image. The subsequent `singularity run` uses this cached image. However, if the cache is somehow deleted or corrupted, a "container not found" error will result, even though a container image is expected based on previous activity. The commands to delete the cache vary depending on singularity version and local configuration.  I’ve seen issues occur when the cached image’s underlying file or directory is removed or altered by some other system process, or by incorrect user intervention which has been applied without proper knowledge of the underlying structure. The last run will fail due to the cache being cleaned, or an image within the cache being deleted.

When faced with "Singularity container B not found," a methodical debugging approach is crucial. First, double-check the container image name or path. Verify the file exists and that the user possesses the appropriate file permissions. If the container was pulled from a registry, confirm that the local Singularity cache hasn’t been modified or cleared unintentionally. Check user permissions for the cache directory. Lastly, examine the environment for custom Singularity configuration that might be influencing how Singularity resolves container locations.

For additional knowledge, I’d recommend researching Singularity's documentation, particularly the section on container image storage and caching mechanisms. Also, familiarization with the specific system's configuration is beneficial. Online forums and user groups can provide practical advice from other users. System administrator documentation often contains specific details that are pertinent to a particular cluster or computing environment, including permission settings, storage locations, and network access information. This information can greatly assist with diagnosing the root causes of such errors.
