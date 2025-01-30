---
title: "How do I resolve the missing libnvidia-fatbinaryloader.so.384.90 shared library?"
date: "2025-01-30"
id: "how-do-i-resolve-the-missing-libnvidia-fatbinaryloaderso38490-shared"
---
The absence of `libnvidia-fatbinaryloader.so.384.90` typically signals a mismatch between the NVIDIA driver version expected by an application and the version actually installed on the system, a scenario I've encountered frequently while optimizing GPU compute workflows. This particular shared library is a critical component of the NVIDIA driver stack, responsible for loading and processing "fat" binary files, which contain compiled code targeting various GPU architectures within a single file. Without it, applications relying on CUDA or other NVIDIA acceleration technologies will fail to launch or execute correctly. Resolving this involves a systematic approach, verifying installed driver versions, ensuring consistent dependencies and, if necessary, performing driver reinstalls.

The core of the issue resides in NVIDIA’s driver release cycle. Every new driver often introduces or modifies internal interfaces. Applications are, therefore, compiled and linked against a specific driver version's API and libraries, including `libnvidia-fatbinaryloader.so`. When this dependency is not satisfied, the dynamic linker, responsible for resolving shared library dependencies at runtime, cannot find the required version of the library, producing an error that cites the missing `libnvidia-fatbinaryloader.so.384.90`. The "384.90" segment signifies the specific driver release that the application was targeting at compile time. If a system has a different driver version installed – newer or older – this mismatch triggers the error.

To diagnose and address this, the first step is to determine both the driver version required by the application and the driver version currently installed on the system. The former is not always readily apparent, but it can often be inferred from the application's documentation or the environment in which the application was compiled. For the installed version, we can query the system. On a Linux system utilizing the standard NVIDIA driver distribution, the command `nvidia-smi` provides detailed information about the current driver installation. Here's a typical example of its output and how to interpret the relevant sections:

```bash
nvidia-smi
```

**Example Output Snippet:**
```
NVIDIA-SMI 535.129.03           Driver Version: 535.129.03   CUDA Version: 12.2
```

The "Driver Version" field directly reveals the installed driver version. If this value is not compatible with the application's expected version, which in this specific case is implicitly 384.90, that's the root of our problem. If the installed driver is much newer or older, it's unlikely to provide backward compatibility for an older fatbinaryloader.

Another approach to determining the installed driver version is to query the system's package manager. For distributions using `apt`, such as Debian and Ubuntu, the following command reveals the installed package details:

```bash
dpkg -l | grep nvidia-driver
```

**Example Output Snippet:**

```
ii  nvidia-driver-535           535.129.03-0ubuntu1~22.04.1    amd64        NVIDIA driver metapackage
```

Again, the package name here contains the driver version, in this example 535.129.03. Similarly, distributions using `yum`, such as Red Hat and CentOS, can use the command:

```bash
yum list installed | grep nvidia-driver
```

This provides similar information regarding the installed driver package. Once we have determined that there is an incompatibility, resolution entails either updating the driver to a version that provides the required libraries or downgrading to the specific version required by the application.

A common scenario involves an application that expects an older driver version, while the system has been upgraded to a newer driver. Upgrading or downgrading the drivers requires careful planning. If the existing driver is provided via the system's package manager, we would ideally utilize the same tools to perform the changes. The steps are, generally:

1. **Identify the target driver version:** From application documentation or specific build configuration information, note the exact version expected.
2. **Uninstall the existing driver:** Using the appropriate package manager, remove the current driver package. For example, `sudo apt remove nvidia-driver-535` or `sudo yum remove nvidia-driver-535`. Note that exact package names can vary.
3. **Install the desired driver version:** After carefully removing the previous driver, install the new one, ensuring you are using a repository that contains the appropriate package.
    *   For example, using `apt`: `sudo apt install nvidia-driver-384` might be used, if the appropriate package is available within the enabled repositories.
    *   For example, using `yum`: `sudo yum install nvidia-driver-384` may be used.

It's essential to consult the relevant distribution's documentation for specific procedures related to package management, since methods and package names can deviate from the examples. Furthermore, after installing or removing driver packages, a system reboot is usually required for the changes to take effect.

A less common, but still possible, scenario is that an application may expect a very specific version of the driver, and that the desired driver may no longer be available through typical distribution channels. In such instances, it may become necessary to install the driver directly from NVIDIA's website. This usually entails downloading the correct installer and then running the installation script. This method bypasses the system's package manager and requires that the user carefully follows NVIDIA's installation instructions. Moreover, direct driver installations can be less robust when compared to package-manager driven installations, as conflicts with system-wide dependencies can more easily arise. Careful consideration must be given to its use.

Finally, in some intricate situations, an application might require driver components that are not included within a typical driver install, such as specific developer tools or legacy libraries. If the above methods do not resolve the issue, it might be necessary to manually extract or copy the required `libnvidia-fatbinaryloader.so.384.90` from a driver archive that specifically contains it and place it in a location where the system's dynamic linker will find it, such as `/usr/lib` or `/usr/local/lib` or within the same directory as the executable which needs it. However, this method is significantly more advanced and carries a higher risk of introducing system instability. It is advisable only after consulting with experts or exhaustively reviewing application documentation and system configurations.

In summary, the `libnvidia-fatbinaryloader.so.384.90` error usually signifies a driver version mismatch. Resolving this involves determining the correct driver version, carefully uninstalling the existing version and installing the appropriate one from system repositories using tools like `apt` and `yum`, or in extreme cases from NVIDIA directly. The specific method required depends on the specific software and the user's technical expertise.

For further study, I recommend reviewing the documentation for the specific Linux distribution involved, along with NVIDIA's official website's driver documentation and release notes. The NVIDIA developer forum can be a valuable resource to identify similar issues experienced by others. The specific application's documentation is also key as there might be a recommended or required driver version, or workarounds for compatibility issues. While specific website links cannot be provided, searching the relevant resources will prove useful.
