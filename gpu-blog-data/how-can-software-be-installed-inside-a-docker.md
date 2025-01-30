---
title: "How can software be installed inside a Docker Windows container?"
date: "2025-01-30"
id: "how-can-software-be-installed-inside-a-docker"
---
Windows containers, unlike their Linux counterparts, do not natively support a full package management system like apt or yum within the container image itself. This presents unique challenges when attempting to install software after a container has been created from an image. I've personally spent considerable time debugging situations where a seemingly straightforward installation failed silently in a Windows container context, leading me to adopt more explicit methods. The approach deviates significantly from the common practice of simply using a shell and a package manager during the build process. The core issue lies in the fundamental design differences between Windows and Linux, particularly the management of shared system resources.

Generally, the installation of software within a Windows Docker container occurs in one of two primary phases: **during the image build process** or **post-container creation**. The former is the preferred and often only practical route for most common software requirements. Performing installations post-creation is more akin to managing software on a bare-metal Windows instance, something that should be done judiciously and only under very specific circumstances, as it complicates reproducibility. Let’s explore the nuances of each method.

The **image build-time approach** leverages the Dockerfile instructions to incorporate the necessary software within the container image before the container even exists. This involves using `RUN` instructions that execute commands within the context of the build. However, unlike Linux, Windows requires careful orchestration of executable installers (.msi, .exe, etc.) or direct file manipulation. The standard “installer.exe /quiet” rarely works flawlessly, due to issues such as required user interaction, lack of awareness of the Windows installation context, or even the container's limited interaction with its host environment. We address these challenges by carefully selecting the appropriate installer arguments and sometimes pre-staging necessary dependencies. In my experience, it's essential to understand the specific installer’s command-line parameters.

Here's how the build-time approach might be structured:

```dockerfile
# escape=`
FROM mcr.microsoft.com/windows/servercore:ltsc2022

# Set a working directory
WORKDIR /app

# Copy the installer executable to the container
COPY MyInstaller.exe .

# Execute the installer with appropriate command-line arguments
RUN Start-Process -Wait -FilePath "MyInstaller.exe" -ArgumentList "/quiet", "/norestart"

# Copy the application files into the container
COPY ApplicationFiles .

# Set the entry point for the container
ENTRYPOINT ["Application.exe"]
```

**Commentary:**

This Dockerfile example illustrates the process of installing an application using an executable file. Note the critical use of `Start-Process -Wait` to ensure that the installer completes before the next instruction is executed. The arguments `/quiet` and `/norestart` are commonly used in silent installations, however, it is important to consult the installer documentation for the appropriate flags. Without the `-Wait` parameter, the installer might be started asynchronously, causing build errors if the container build proceeds before installation completes. I've encountered instances where this led to inconsistent image builds that worked erratically. The `COPY` command transfers the application's executable and associated files into the specified working directory. The `ENTRYPOINT` sets the application as the default executable when the container starts. This is a typical example for relatively simple scenarios, but can become considerably complex if prerequisites must be installed or if the installation itself requires multiple steps.

Now let's consider a scenario where the installer is an MSI package:

```dockerfile
# escape=`
FROM mcr.microsoft.com/windows/servercore:ltsc2022

# Set a working directory
WORKDIR /app

# Copy the MSI installer file
COPY MyInstaller.msi .

# Execute the MSI installer using msiexec
RUN msiexec /i "MyInstaller.msi" /qn /norestart

# Copy the application files into the container
COPY ApplicationFiles .

# Set the entry point
ENTRYPOINT ["Application.exe"]
```

**Commentary:**

Here, the Windows `msiexec` utility is used to execute the MSI installer. Again, the parameters `/qn` (quiet mode) and `/norestart` ensure a silent, non-interactive installation.  The critical point here is recognizing the subtle difference in how executables and MSI installers are handled by Windows. Relying on common Linux patterns like `apt install` is not viable. This example uses absolute path for the .msi file, as `msiexec` requires the full file path. In my experience, failing to specify the full path can lead to unexpected failures, since relative paths can sometimes be misinterpreted or not resolved in the Docker build context, leading to an installer not being found.

Finally, let's explore installing a component through a PowerShell script, which is invaluable for managing complex installations or configurations.

```dockerfile
# escape=`
FROM mcr.microsoft.com/windows/servercore:ltsc2022

# Set a working directory
WORKDIR /app

# Copy the PowerShell installation script
COPY install.ps1 .

# Execute the PowerShell script using pwsh.exe or powershell.exe
RUN powershell.exe -NoProfile -ExecutionPolicy Bypass -File install.ps1

# Copy the application files into the container
COPY ApplicationFiles .

# Set the entry point
ENTRYPOINT ["Application.exe"]
```

**Commentary:**

This example employs a PowerShell script (`install.ps1`) to manage installation steps, offering considerable flexibility. The `-NoProfile` argument prevents PowerShell from loading user profiles, ensuring consistent behavior in the containerized environment. The `-ExecutionPolicy Bypass` allows the script to run, regardless of the system's policy, though this should be used with caution.  In many cases, this method is preferable because it allows for more intricate installation processes including conditional checks, registry modifications, and interaction with Windows services. PowerShell's robust capabilities enable us to handle edge cases and complex scenarios that might prove challenging with simple RUN instructions alone. The contents of the `install.ps1` would contain the specific installation and configuration logic required. For example, a script could retrieve packages from the internet using `Invoke-WebRequest` or install Chocolatey packages, if needed.

**Post-container creation** software installation, while technically feasible, is not usually recommended for several reasons.  Firstly, it makes container deployments less reproducible and deterministic.  The installed state of a container will depend on operations performed after it's created, and this could be different for each instantiation. Additionally, it blurs the lines between container immutability and traditional server management. While tools like Docker’s interactive mode, `docker exec`, allow us to shell into a running container and execute commands (including installers), any changes made are not persisted in the image, meaning the changes will only exist in the running container, and they will be lost when the container is stopped or removed. If software needs to be installed after the container is created, a process such as container orchestration software like Kubernetes should be responsible for provisioning the container with the correct runtime environment, usually through re-building the base container image itself.

When choosing between these two methods, favor the build-time approach for almost all cases. It leads to more reproducible, predictable, and immutable containers. Treat post-creation modification with extreme caution, resorting to it only for specific needs where build-time installation is truly unachievable or impractical. In such situations consider creating a separate “management” container if a tool is needed for post installation.

For further learning, consult the official Microsoft documentation on Windows container images, particularly focusing on Dockerfile syntax and command-line parameters for standard installers (`msiexec`, `setup.exe`, etc.). The PowerShell documentation provides comprehensive information on how to use it for automated tasks, and the Chocolatey documentation can provide examples on managing Windows packages. These resources helped me substantially in developing reliable and manageable Windows Docker containers. In summary, the key is understanding the limitations of the Windows environment within a Docker context and adapting traditional installation procedures accordingly.
