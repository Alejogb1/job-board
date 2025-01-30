---
title: "Why isn't my VS Code dev container running on Docker?"
date: "2025-01-30"
id: "why-isnt-my-vs-code-dev-container-running"
---
The most common reason a VS Code Dev Container fails to launch within Docker stems from improperly configured or missing Dockerfiles.  My experience debugging hundreds of these issues across various projects points consistently to discrepancies between the `devcontainer.json` file's specifications and the Dockerfile's implementation, or a fundamental misunderstanding of the layering and context within the Docker build process.  This often manifests as seemingly innocuous errors, masking the true underlying problem.

Let's clarify the expected workflow.  VS Code's Dev Container feature relies on a Docker image. This image is typically built from a `Dockerfile` located within your project directory.  The `devcontainer.json` file acts as a configuration file, instructing VS Code on how to interact with that Docker image, including specifying the Dockerfile's location, setting ports, configuring volumes, and defining the environment. The failure often arises from a disconnect between these two crucial components.

**1. Clear Explanation of the VS Code Dev Container and Docker Interaction:**

The Dev Container feature leverages Docker's containerization capabilities to provide a consistent and isolated development environment. When you open a folder with a `devcontainer.json` file, VS Code attempts to:

1. **Identify the Dockerfile:** It looks for a `Dockerfile` in the location specified in the `devcontainer.json` (defaulting to the `.devcontainer` folder).
2. **Build the Docker Image (if necessary):** If a matching image doesn't exist locally, VS Code instructs Docker to build the image from the `Dockerfile`.
3. **Create and start the Container:**  Using the built image, it creates a Docker container, applying any mount points (volumes) or port mappings defined in `devcontainer.json`.
4. **Connect to the Container:** Finally, it connects your VS Code instance to the running container, allowing you to work within the isolated environment.

Failure points typically arise in steps 2 and 3.  A failed build indicates problems within the `Dockerfile`, while a failure to start the container often results from incorrect configuration within `devcontainer.json` or mismatches between it and the `Dockerfile`.  Insufficient privileges, network issues, and Docker daemon problems can also contribute, but these are generally less frequent than the configuration discrepancies.


**2. Code Examples and Commentary:**

**Example 1:  Incorrect Dockerfile Base Image**

```dockerfile
# Incorrect:  Using a non-existent image
FROM non-existent-image:latest

# ...rest of the Dockerfile...
```

```json
{
  "name": "My Dev Container",
  "image": "my-custom-image:latest", // This would fail because the build failed
  "workspaceFolder": "/workspace"
}
```

**Commentary:** This example demonstrates a common error. The `Dockerfile` attempts to use a non-existent base image, leading to a failed build.  VS Code's `devcontainer.json` subsequently fails to create the container because the image `my-custom-image:latest` (assumed to be built from this Dockerfile) is missing. Ensure the base image specified (e.g., `ubuntu:latest`, `node:16`, `python:3.9`) is readily available from a Docker registry (like Docker Hub).

**Example 2:  Port Mapping Mismatch**

```dockerfile
# Correct Dockerfile, exposes port 3000
FROM node:16
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
EXPOSE 3000
CMD ["npm", "start"]
```

```json
{
  "name": "My Dev Container",
  "dockerfile": "Dockerfile",
  "ports": [
    { "publisher": 8080, "target": 3000 } // Mismatched ports
  ],
  "workspaceFolder": "/app"
}
```

**Commentary:** The `Dockerfile` correctly exposes port 3000, but `devcontainer.json` attempts to map port 8080 on the host to port 3000 inside the container. This discrepancy leads to the application being inaccessible, not a container startup failure.  Verify that the ports specified for mapping are consistent between both files.  Note that, while in this example port 3000 is exposed, it is not published - this should be adjusted to meet application requirements.  To publish it you may need to run something like `docker run -p 3000:3000 my-image`

**Example 3:  Incorrect Volume Mounting**

```dockerfile
# Correct Dockerfile, defines the working directory.
FROM python:3.9
WORKDIR /app
COPY . /app
CMD ["python", "main.py"]
```


```json
{
  "name": "My Dev Container",
  "dockerfile": "Dockerfile",
  "workspaceFolder": "/app",
  "mounts": [
    "source=/path/to/project,target=/wrong/path", // Incorrect target path
    "source=${localWorkspaceFolder},target=/app" // Correct mount
  ]
}
```

**Commentary:** This example shows a potential problem with volume mounting. The first mount attempts to map a source path to an incorrect target path inside the container. This can lead to the container starting but your local changes not being reflected inside, or to an application failure due to incorrect file paths. Always ensure the target paths in `mounts` accurately correspond to the paths expected within your container’s application. The second mount is correct given that the `Dockerfile` `WORKDIR` is `/app`.  Pay close attention to path consistency, especially when using variables like `${localWorkspaceFolder}`.



**3. Resource Recommendations:**

Consult the official VS Code documentation on Dev Containers.  Review Docker's documentation on building images and managing containers.  Familiarize yourself with the structure and functionality of `Dockerfile` instructions. Understanding the basics of container networking will be extremely helpful.  Thorough familiarity with the Linux command line will allow for easier inspection and debugging.

Through my extensive experience, I've found that a methodical approach to troubleshooting, starting with careful examination of the `Dockerfile` and `devcontainer.json` for inconsistencies, almost always reveals the root cause of such failures.  Remember to check Docker logs for detailed error messages and use the Docker CLI to inspect the container's state after attempting a launch. These strategies, combined with a strong grasp of fundamental Docker concepts, are key to resolving the vast majority of VS Code Dev Container issues.
