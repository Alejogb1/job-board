---
title: "How do Docker and Singularity working directories differ?"
date: "2025-01-30"
id: "how-do-docker-and-singularity-working-directories-differ"
---
The core operational difference between Docker and Singularity working directories stems from their fundamentally different approaches to containerization: Docker relies on a daemon and shared host filesystem access by default, whereas Singularity operates within a more isolated, security-focused environment. This distinction significantly impacts how working directories behave within each containerization technology, particularly concerning persistent data management and security implications.


My experience working with high-performance computing clusters and deploying sensitive scientific applications highlighted these differences repeatedly.  I've witnessed firsthand how Docker's flexibility, while advantageous in certain development scenarios, can lead to unexpected security vulnerabilities when improperly configured.  Singularity's design, on the other hand, proved invaluable for ensuring reproducibility and preventing unintended host system modifications within these HPC environments.


**1. Clear Explanation of Working Directory Differences**

In Docker, the container's working directory is typically mapped to a directory on the host filesystem.  This mapping is configured using the `-v` or `--volume` flag during container launch or within the Dockerfile itself.  This means that files created or modified within the container's working directory are directly reflected on the host machine.  This behavior provides flexibility for data sharing and persistent storage but introduces security risks if not managed carefully.  An attacker compromising the container might gain unauthorized access to the host's filesystem through this shared volume.

Singularity, conversely, operates with a more isolated model. The working directory within a Singularity container is, by default, entirely contained within the container's image.  Changes made within the container's working directory do not affect the host filesystem unless explicitly bound through a bind mount similar to Docker's `-v` flag. However, even with bind mounts, Singularity offers better control and explicit definition of the access points, mitigating unintended side effects.  Crucially, the default behavior inherently enhances security by preventing accidental or malicious modification of the host system.


**2. Code Examples and Commentary**

**Example 1: Docker with Volume Mount**

```dockerfile
FROM ubuntu:latest

WORKDIR /app

COPY . /app

CMD ["echo", "Working directory: $(pwd)"]
```

```bash
docker build -t my-docker-image .
docker run -v $(pwd):/app -it my-docker-image
```

*Commentary:* This Docker example creates a container with the working directory `/app` mapped to the current host directory using `-v $(pwd):/app`.  Any changes made to files within `/app` inside the container will directly impact files in the current host directory. This behavior facilitates development workflows but compromises security if not properly managed.


**Example 2: Singularity with Bind Mount**

```singularity
Bootstrap: docker
From: ubuntu:latest

%environment
    WORKDIR /app

%runscript
    echo "Working directory: $(pwd)"
```

```bash
singularity build my-singularity-image.sif Singularityfile
singularity exec -B $(pwd):/app my-singularity-image.sif sh -c "echo 'test' > test.txt"
```

*Commentary:*  This Singularity example uses a similar approach to map a host directory, but uses the `-B` flag for explicit binding.  The `Singularityfile` defines the container environment and the `%runscript` section specifies the command to execute.  This showcases how Singularity leverages a more controlled binding mechanism compared to Docker's implicit behavior.  Note that unlike Docker, the `Singularityfile` is used to define the container, rather than a separate `Dockerfile`.

**Example 3: Singularity without Bind Mount**

```singularity
Bootstrap: docker
From: ubuntu:latest

%environment
    WORKDIR /app

%runscript
    touch myfile.txt
    echo "Working directory: $(pwd)"
    ls -l /app
```

```bash
singularity build my-singularity-image.sif Singularityfile
singularity exec my-singularity-image.sif sh -c "./runscript.sh"
```


*Commentary:* This Singularity example runs without any bind mount.  The `myfile.txt` file created within `/app` will only exist inside the container and will not persist on the host.  This inherently enhances security and data integrity.  After the `singularity exec` command completes, the `myfile.txt` file will not be present in the host's filesystem. This demonstrates the core security advantage of Singularity’s default behavior.



**3. Resource Recommendations**

For a deeper understanding of Docker's functionalities, I highly recommend exploring the official Docker documentation and tutorials. They offer comprehensive guides on various aspects of Docker, including volume management and security best practices.

To effectively utilize Singularity,  refer to the official Singularity documentation. It contains in-depth information about the build process, bind mounts, and security features. This documentation is crucial to understanding the differences and mastering Singularity's unique approach to containerization.  Furthermore, several advanced tutorials covering HPC-specific use cases will be invaluable.  Finally, studying container security best practices in general is crucial for both Docker and Singularity deployments, irrespective of the specific working directory behavior.  Understanding the principles of least privilege and secure configuration are vital to both systems.
