---
title: "Is it possible to run a Windows container with GUI support?"
date: "2024-12-23"
id: "is-it-possible-to-run-a-windows-container-with-gui-support"
---

Let's tackle this one head-on. It's a question I've grappled with extensively, particularly when migrating legacy applications during a rather complex infrastructure modernization project a few years back. The need to package older Windows apps with their full graphical interfaces into containers for improved scalability and portability came up, and it definitely wasn't straightforward. So, yes, it *is* possible to run a Windows container with gui support, but it’s a journey, not a simple toggle switch.

The challenge primarily stems from the fundamental nature of containerization. Traditionally, containers are designed to isolate applications and their dependencies at the process level, focusing on lightweight, command-line driven services. Graphic user interfaces, however, interact heavily with the underlying operating system's windowing system, which is deeply integrated into the kernel. Windows containers, specifically, by default, run in process isolation, meaning they don’t have direct access to the host's graphic resources.

To get a gui application working inside a windows container, we essentially need a mechanism to bridge the gap between the container’s isolated environment and the host's display system. This often involves a combination of technologies, most notably using Remote Desktop Protocol (rdp) or specialized virtualization techniques such as those found in hyper-v isolated containers. The solution chosen depends largely on the required use case and performance considerations. Let’s break down the common approaches:

**RDP within a container:**

This is arguably the most prevalent approach for enabling gui within windows containers. The core idea is to run an rdp server within the container itself. Then, you use a regular rdp client to connect to that container and display its gui on your host machine or on another remote computer. This approach avoids complex virtualization setups and is reasonably straightforward to implement. However, it adds the overhead of rdp traffic, which might impact performance in some latency-sensitive applications.

Here’s a basic example of how to set this up, focusing on a `dockerfile` and a powershell script, as they are the bread and butter of windows container management:

```dockerfile
# escape=`
FROM mcr.microsoft.com/windows/servercore:ltsc2022

# Install RDP services
RUN powershell -Command `
    Install-WindowsFeature Remote-Desktop-Services -IncludeAllSubFeature `
     | Out-Null;`
    New-Item -ItemType "Directory" -Force -Path "C:\rdp" ; `
    New-Item -ItemType File -Force -Path "C:\rdp\start_rdp.ps1";`
    'Start-Service TermService' | Out-File -FilePath "C:\rdp\start_rdp.ps1"
    
#configure user for rdp
RUN powershell -Command `
    $password = ConvertTo-SecureString -String 'P@ssword123' -AsPlainText -Force ; `
    $user = "ContainerAdmin" ;`
    New-LocalUser -Name $user -Password $password  ; `
    net localgroup "Remote Desktop Users" $user /add
    
EXPOSE 3389
ENTRYPOINT ["powershell", "-ExecutionPolicy", "Bypass", "C:\\rdp\\start_rdp.ps1"]

# you'd normally copy your application in here, but lets skip this for brevity in this example.
```
In the example above, I'm using windows server core as my base image. The important part is the `install-windowsfeature` command that installs the required rdp services. We create a simple ps1 script to start the rdp service on container startup. The second `run` instruction creates a user and adds it to the rdp group to allow rdp connections.  Finally, we expose port 3389 (the default rdp port) and set the entry point to execute our script. After building this image you could run the container like so: `docker run -d -p 3389:3389 <your image name>` and connect to it using a rdp client using the ip address of your docker host and the user credentials defined in the `dockerfile`.

**Hyper-V isolated containers:**

This approach is less common for simple gui applications but extremely powerful in specific scenarios where strong isolation is paramount. Here, the container runs within a minimal Hyper-V virtual machine. This provides enhanced security and isolation compared to process isolation but adds substantial overhead. Importantly, when using hyper-v isolation, the container *does* have its own instance of the windows display driver, allowing for more robust compatibility with applications that make direct calls to win32 apis related to graphic rendering. While you still would often need to leverage an rdp solution to interact with the gui of your application inside the container, the graphics subsystem is isolated.

This is a more complex setup and would require the following general workflow: first you'd use a full windows server image as a base, since servercore does not include all the needed resources for full display interaction, then inside your container you would install hyper-v support features, including the virtual display adapter, and finally you would still need to install an rdp server as seen in the previous example to connect to your container.

To provide a more practical example, let us show the dockerfile code for the hyper-v container (again skipping app install for brevity):

```dockerfile
# escape=`
FROM mcr.microsoft.com/windows:ltsc2022

# Install Hyper-V features
RUN dism /online /Enable-Feature /All /FeatureName:Microsoft-Hyper-V /quiet /norestart

# Install RDP services
RUN powershell -Command `
    Install-WindowsFeature Remote-Desktop-Services -IncludeAllSubFeature `
     | Out-Null;`
    New-Item -ItemType "Directory" -Force -Path "C:\rdp" ; `
    New-Item -ItemType File -Force -Path "C:\rdp\start_rdp.ps1";`
    'Start-Service TermService' | Out-File -FilePath "C:\rdp\start_rdp.ps1"
    
#configure user for rdp
RUN powershell -Command `
    $password = ConvertTo-SecureString -String 'P@ssword123' -AsPlainText -Force ; `
    $user = "ContainerAdmin" ;`
    New-LocalUser -Name $user -Password $password  ; `
    net localgroup "Remote Desktop Users" $user /add
    
EXPOSE 3389
ENTRYPOINT ["powershell", "-ExecutionPolicy", "Bypass", "C:\\rdp\\start_rdp.ps1"]
```

Here the main difference is the addition of `dism /online /enable-feature` which will enable hyper-v features in the container (remember, that hyper-v containers must be run as *isolated hyper-v* containers). As before, you would use a rdp client to connect to this container. To run it using docker you would need to specify that this is a hyper-v isolated container using the `--isolation=hyperv` switch: `docker run -d -p 3389:3389 --isolation=hyperv <your image name>`.

**VDI-like Solutions**

There are also solutions that, while not technically running *a* single gui application in *a* container, allow you to run many applications across multiple containers and connect to them via a web browser or a specific application client. These are the likes of apache guacamole or citrix virtual apps and desktops. I'm mentioning these for completeness, since they often come up as potential solutions, even though they are more of a 'desktop as a service' solution, not a container specific solution.

**Important considerations:**

When dealing with gui applications in containers, several factors come into play:

*   **Image size:** windows images can be quite large. If you have a simple app, consider using servercore, as seen in my first example, and building as minimal an image as possible. Be selective with your installed features.
*   **Performance:** rdp based solutions can introduce performance bottlenecks, especially with complex applications or over high-latency networks. If you can, try to minimize the need to move large chunks of graphic data.
*   **Security:** ensure your rdp connections are secured using tls and consider enabling network firewalls to further limit access to your container. When using hyper-v, take into account that you now have a vm-like environment to secure as well.
*   **Licensing:** remember that if your application requires licensing based on a specific machine, that licensing will be tied to the container, which means that you might have to re-activate your application after the container has been recreated.

**Recommended Resources**

For those wanting to dive deeper, I’d strongly suggest checking out these resources:

*   **Microsoft's official documentation on Windows containers:** This is the go-to source for all things related to windows containers, from basic setup to more advanced configurations, including information on hyper-v isolation.
*   **The book “Docker Deep Dive” by Nigel Poulton:** This book is a phenomenal general resource on docker and containers and offers a good section on Windows containers and how they work at a deeper level.
*   **Research papers on container security:** exploring the isolation mechanisms of process-isolated versus hyper-v isolated containers can be beneficial for understanding the security implications.

To summarize, while it certainly takes more work than running a standard, headless application in a container, achieving gui support in windows containers is absolutely feasible. The exact approach will depend on the specific requirements of your application and the constraints of your infrastructure. The most important thing to remember is to understand the tradeoffs that each approach brings and to select the solution that suits your needs the most. It’s definitely a challenge that I’ve encountered in my career, and having the correct foundations definitely made the process smoother.
