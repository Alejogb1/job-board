---
title: "Why does the singularity container fail locally with a libicui18n.so.66 error?"
date: "2024-12-23"
id: "why-does-the-singularity-container-fail-locally-with-a-libicui18nso66-error"
---

Let’s tackle this libicui18n.so.66 issue. It's a scenario I've certainly encountered in my past, specifically during a large-scale genomic analysis project where we relied heavily on singularity containers for reproducible research. We were pulling our hair out trying to get things to run locally before they could be deployed on our HPC cluster. The core problem here stems from version mismatches between the libraries inside your container and those available on your host system. `libicui18n.so.66` is part of the International Components for Unicode (ICU) library, which handles tasks like text collation, date/time formatting, and number formatting. Its version 66 is not that old and relatively common, but conflicts arise surprisingly often when working with containers.

When a singularity container starts, it attempts to use libraries from the host system for efficiency, when appropriate. However, if the container was built with a specific version of the ICU library (let's say, version 66 in your case) and the host system has a different version (say, 67, or even 64), the containerized application might not find the required library or the API may be incompatible. This can manifest as the error you are seeing, essentially causing the application to crash because it cannot load the correct library. The specific error message is a good indicator of this library incompatibility.

A critical detail to note here is that singularity attempts to resolve library dependencies at runtime. Unlike some other container technologies that bundle all necessary libraries, singularity’s approach tries to optimize resource utilization by leveraging host system libraries. While this works well in many cases, it can introduce these versioning issues when the container and the host diverge in their library dependencies.

There are a few ways to navigate this, and my experience dictates that a combination of approaches often yields the most robust solution. I’ll outline three methods with accompanying code examples to illustrate how to address the situation:

**Method 1: Statically Link the Library in the Container**

The most straightforward, yet often the least resource-efficient solution, is to statically link the correct version of the `libicui18n.so.66` library into your container. This ensures the container will not depend on the host system for this particular library. Here's how one can proceed. I'm presenting a basic `Dockerfile` equivalent here, assuming you build your singularity container from a Dockerfile:

```dockerfile
FROM ubuntu:20.04

# Install necessary build tools and dependencies
RUN apt-get update && apt-get install -y wget build-essential

# Download the correct ICU library version
RUN wget http://download.icu-project.org/files/icu4c/66.1/icu4c-66_1-src.tgz && \
    tar -xzf icu4c-66_1-src.tgz

# Configure and build ICU from source (this is a simplified version, you may need flags specific to your system)
WORKDIR icu/source
RUN ./configure --prefix=/usr/local/icu66 && make && make install

# Now, add a dummy application that needs ICU
RUN apt-get update && apt-get install -y g++
RUN echo '#include <iostream>\n#include <unicode/unistr.h>\n\nint main() {\n    icu::UnicodeString s("Hello, ICU!");\n    std::cout << s << std::endl;\n    return 0;\n}' > test.cpp

RUN g++ -std=c++11 -o test test.cpp -I /usr/local/icu66/include -L /usr/local/icu66/lib -licuuc -licudata -licui18n

# Set the library path
ENV LD_LIBRARY_PATH=/usr/local/icu66/lib:$LD_LIBRARY_PATH

# Final entry point (for singularity)
CMD ["./test"]
```

In this example, we are downloading, compiling, and installing version 66.1 of the ICU library directly into the container's filesystem, thereby removing any dependency on the host's library. When you build this docker image and then convert it to singularity, the `libicui18n.so` within the `/usr/local/icu66/lib` directory will be used.

**Method 2: Using `singularity.conf` to Control Library Search Paths**

Another way, sometimes more flexible, is to control the library search path within singularity using the `singularity.conf` file. This allows you to influence where the container looks for shared libraries on the host system, potentially overriding system defaults and pointing to the correct ICU version. This approach assumes you have access to a controlled environment with a suitable version of ICU installed, and you simply wish to tell singularity where to look. In the following example, I'm demonstrating how you could add a path to your singularity configuration file.

First, you would need to modify the default `singularity.conf` file. The location of this configuration file is typically `/usr/local/etc/singularity/singularity.conf` or `/etc/singularity/singularity.conf`, depending on your installation. You need root privileges for this modification.
Inside `singularity.conf` you would then modify, or if needed, add, a section to influence library paths.
```
[librarypath]
  # add your custom library directory
  path = /path/to/your/specific/icu/lib:/usr/lib64:/lib64:/usr/lib:/lib
  
```
**Note**:  Replace `/path/to/your/specific/icu/lib` with the actual path where your system's ICU library version 66 (or a compatible version) is located. After you edit the configuration, you will need to restart your singularity services and rebuild your singularity image (if it has library paths configured inside the container image). This modification ensures that the first place singularity looks is the directory you provided, before searching the host's system's default library paths. This change affects all singularity containers on that host.
This solution is advantageous when you have multiple containerized applications that all depend on the same non-default library location. However, modifying `singularity.conf` requires root privileges and affects all users on that machine, so proceed cautiously.

**Method 3: Creating a Custom Singularity Image with Specified Libraries**

Alternatively, one could take a hybrid approach of using a container build environment and selectively including the required libraries. If neither of the above solutions work for your scenario, I've found that making a very minimal container with only the libs you need can be a viable strategy.
We can create our own container from scratch. This is a simplified example; we will include the required `libicui18n.so.66` within the container.

```dockerfile
FROM scratch
# Copy your lib
COPY path/to/your/libicui18n.so.66 /lib64/libicui18n.so.66
# Add any other libs as needed
COPY path/to/your/other.so /lib64/other.so
# Add an entry point
COPY your_app_binary /
CMD ["./your_app_binary"]
```
You will then need to build the container using tools such as docker or podman, and convert this image into singularity container image with `singularity build <container_name>.sif docker-daemon://<image_name>`. This way the image already has all its dependencies.

**Recommendation for Further Learning:**

To deepen your understanding of shared library loading, I recommend reading "Linkers and Loaders" by John R. Levine. It provides an exhaustive treatment of how shared libraries are resolved and loaded. Additionally, the ICU project website (icu-project.org) provides excellent documentation regarding the library and version specific details. For a detailed discussion of container technology, "Docker Deep Dive" by Nigel Poulton offers detailed explanations of the inner workings of container technologies and may help to understand how singularity, in turn, operates.

In summary, the `libicui18n.so.66` error during singularity container usage stems from a mismatch between the library versions inside the container and on the host system. Statically linking the library, adjusting the `singularity.conf`, or creating a custom container with pre-packaged dependencies are all valid methods for solving this problem. Each approach has its trade-offs, and choosing the best one often comes down to the specific requirements and constraints of your environment. Through experience, I've learned that understanding these core mechanisms is vital for robust and portable container deployments.
