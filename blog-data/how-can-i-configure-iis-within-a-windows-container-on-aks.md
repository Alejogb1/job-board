---
title: "How can I configure IIS within a Windows container on AKS?"
date: "2024-12-23"
id: "how-can-i-configure-iis-within-a-windows-container-on-aks"
---

Right then, let's tackle configuring IIS inside a Windows container on Azure Kubernetes Service (AKS). I’ve seen this setup more than a few times – usually involving migration scenarios or legacy applications – and while it’s not always the most elegant solution, it’s often a necessary one. The key here isn’t just *getting it to work*, it’s about doing so in a manner that’s manageable, secure, and robust within the kubernetes context.

The first thing to acknowledge is that we’re dealing with a stateful application, something kubernetes traditionally isn't natively designed for. IIS, by its nature, tends to write to disk, manages configuration through the registry, and assumes a somewhat consistent environment. Kubernetes, on the other hand, favors stateless, immutable deployments. So, our challenge is to bridge this gap effectively. We're not trying to shoehorn a square peg into a round hole; rather, we’re carefully carving a new shape that allows both technologies to work in tandem.

My experience has largely involved pre-configuring the IIS instance within the container image itself. This approach minimizes runtime dependencies and makes deployments significantly more consistent. I typically begin with a Dockerfile derived from the `mcr.microsoft.com/windows/servercore/iis` base image. I then use powershell commands within the Dockerfile to enable specific features, configure application pools, and deploy my application's artifacts. Think of this as building a 'golden image' that already contains all the necessary configuration for your application.

Here’s an illustrative example of what a basic Dockerfile might look like:

```dockerfile
#escape=`
FROM mcr.microsoft.com/windows/servercore/iis:windowsservercore-ltsc2022

# Install needed features
RUN powershell -Command `
    Install-WindowsFeature Web-Asp-Net45; `
    Install-WindowsFeature Web-Net-Ext45; `
    Install-WindowsFeature Web-Static-Content;

# Create the website directory
RUN mkdir C:\inetpub\mywebsite

# Copy the application files
COPY . C:\inetpub\mywebsite

# Configure the website
RUN powershell -Command `
    Import-Module WebAdministration; `
    New-Website -Name 'MyWebsite' -PhysicalPath 'C:\inetpub\mywebsite' -Port 80;

# Expose port 80
EXPOSE 80

# Set default startup command
CMD [ "C:\\inetpub\\wwwroot\\iisstart.htm" ]
```

In this simplified example, we're adding necessary web server features, creating a directory for our website, copying application files (represented by the `.` which should be replaced by the application folder) into that directory, then configuring a new website to point to the directory. We also expose port 80 and use a placeholder html page for startup.

Now, deploying this container to AKS requires some specific configurations in your kubernetes manifests. The key is to ensure that the container is able to map port 80 correctly and that the appropriate health checks are in place. Remember, AKS doesn't automatically map ports. You'll need to specify that in your Kubernetes service definition.

Here’s an example of a deployment and service manifest:

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: iis-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      app: iis-app
  template:
    metadata:
      labels:
        app: iis-app
    spec:
      containers:
      - name: iis-container
        image: <your-image-repo>/iis-image:<your-image-tag>
        ports:
        - containerPort: 80
        resources:
            requests:
              cpu: "250m"
              memory: "512Mi"
            limits:
              cpu: "500m"
              memory: "1Gi"
```

```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: iis-service
spec:
  selector:
    app: iis-app
  ports:
    - protocol: TCP
      port: 80
      targetPort: 80
  type: LoadBalancer
```

This deployment creates three replicas of our IIS container with resource requests and limits, and the service then exposes them via a load balancer on port 80. Crucially, observe the container port, target port and exposed port all match.

A critical aspect often overlooked is persistent storage for configurations or application data. While the container image itself should be immutable, your application might need to store data. To achieve that, you'll need to employ persistent volumes and volume mounts. In my own projects, I’ve utilized azure file shares for simple file persistence and used volume mounts in the kubernetes deployment to link a directory inside the container with a persistent volume. I avoid using local storage whenever possible to guarantee the resilience of the deployments across node failures.

Here’s how you can add a volume mount using an azure file share as an example:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: iis-deployment-with-storage
spec:
  replicas: 3
  selector:
    matchLabels:
      app: iis-app-storage
  template:
    metadata:
      labels:
        app: iis-app-storage
    spec:
      containers:
      - name: iis-container
        image: <your-image-repo>/iis-image:<your-image-tag>
        ports:
        - containerPort: 80
        volumeMounts:
        - name: persistent-storage
          mountPath: C:\inetpub\mywebsite\appdata
        resources:
            requests:
              cpu: "250m"
              memory: "512Mi"
            limits:
              cpu: "500m"
              memory: "1Gi"
      volumes:
      - name: persistent-storage
        azureFile:
           secretName: azure-secret
           shareName: myshare
           readOnly: false
```

In this example, an `azureFile` volume has been added which references a secret containing connection information for the file share, and this is mounted to `C:\inetpub\mywebsite\appdata` inside the container. Any writes to that directory will now be persisted across pod restarts, and accessible across all replicas.

Furthermore, ensure your applications health checks are accurately configured, and avoid excessive logging. IIS logs can quickly overwhelm container filesystems. Instead, forward your logs to an external system, such as Azure Monitor or a logging solution tailored for Kubernetes.

For a deeper understanding, I recommend reviewing the official Microsoft documentation on Windows container support, specifically with regard to IIS. The book "Programming Microsoft Windows with C#" by Charles Petzold provides a very strong foundation on the underpinnings of windows, if your application uses .net this will be useful. I also advise studying the kubernetes documentation focusing on deployments, services, and persistent volumes. Specifically, look at the [Kubernetes documentation on workload resources](https://kubernetes.io/docs/concepts/workloads/) as that covers the deployment aspects well. Finally, "Containers for Everyone" by Ricardo Sueiras, Chris Noring, and Brendan Burns provides a very digestible explanation of containerisation in general and may be worth your time to understand the underlying concepts more fully.

In summary, configuring IIS inside Windows containers on AKS is certainly achievable with careful planning and execution. Pre-baking the configuration into the image, properly configuring kubernetes deployments and services, and using persistent storage are crucial elements of a successful setup. Remember that these examples are starting points and you'll need to adjust configurations to fit your specific application's requirements and security constraints.
