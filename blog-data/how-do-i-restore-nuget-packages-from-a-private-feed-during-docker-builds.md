---
title: "How do I restore NuGet packages from a private feed during Docker builds?"
date: "2024-12-16"
id: "how-do-i-restore-nuget-packages-from-a-private-feed-during-docker-builds"
---

Okay, let’s tackle this. It’s something I’ve had to refine over several projects, and it always requires a bit more thought than simply pointing docker to the public feed. We're talking about integrating private nuget feeds, usually within a secure environment, into your docker build processes. This isn’t just about slapping a `-restore` command into a dockerfile; there are security considerations and subtle configurations that need careful attention. I remember a project years back where we had a near-catastrophe because of improperly configured credentials for a private feed within the docker environment – a good lesson learned the hard way.

First off, the core issue arises because, during docker builds, your environment is typically isolated from your local setup. Dockerfiles don't inherently have access to your private nuget credentials stored locally or in your development environment. Therefore, we need to explicitly pass these credentials and configure the nuget client within the docker build context. There are several ways to accomplish this, each with its own pros and cons, but let’s break down what I consider the most reliable methods.

The most straightforward, but potentially least secure, method involves injecting the nuget.config directly into the docker image along with any needed credentials. This means having the credentials stored somewhere accessible during the build. We'll use the docker build's `--build-arg` flag and an environment variable for this. This approach works, but remember, that means these values will be present in your image layers, so proper security procedures (secret management) are critically important for production scenarios.

Here’s an example illustrating this approach. Let’s assume your private nuget feed is at `https://my.private.nuget.server/v3/index.json` and you've got an api key you manage securely.

```dockerfile
# Start with a base image appropriate for your project
FROM mcr.microsoft.com/dotnet/sdk:7.0 AS build

# Set a build argument for the API key
ARG NUGET_API_KEY

# Create the nuget.config file
RUN mkdir -p /root/.nuget

RUN echo "<?xml version='1.0' encoding='utf-8'?>" > /root/.nuget/nuget.config \
  && echo "<configuration>" >> /root/.nuget/nuget.config \
  && echo "  <packageSources>" >> /root/.nuget/nuget.config \
  && echo "    <add key='MyPrivateFeed' value='https://my.private.nuget.server/v3/index.json' />" >> /root/.nuget/nuget.config \
  && echo "  </packageSources>" >> /root/.nuget/nuget.config \
    && echo "   <apikeys>" >> /root/.nuget/nuget.config \
    && echo "     <add key='https://my.private.nuget.server/v3/index.json' value='${NUGET_API_KEY}' />" >> /root/.nuget/nuget.config \
    && echo "   </apikeys>" >> /root/.nuget/nuget.config \
  && echo "</configuration>" >> /root/.nuget/nuget.config

# Copy the project files
COPY . .

# Restore nuget packages
RUN dotnet restore

# Build the application
RUN dotnet build -c Release -o /app

# Publish the application
RUN dotnet publish -c Release -o /publish

# Create the final runtime image
FROM mcr.microsoft.com/dotnet/aspnet:7.0 AS runtime
WORKDIR /app
COPY --from=build /publish .
ENTRYPOINT ["dotnet", "YourApplication.dll"]
```

To build this image, you would execute:

`docker build --build-arg NUGET_API_KEY=<your_actual_api_key> .`

This method works, but be absolutely certain that you’re managing the api key securely; avoid embedding it directly in your Dockerfile or shell scripts. Ideally, use a CI/CD environment’s secrets management, injected during build time only. Remember that docker image layers are persisted after a build, so even if you overwrite it in a later stage, an earlier layer might contain the secret. This is why I typically recommend separating build and runtime environments.

A more robust and secure alternative involves using a multi-stage build and not injecting credentials directly into the initial build step, instead utilizing a 'restore-stage' that discards the nuget.config with secrets. In this version, we configure nuget during the restore stage, but leave it out of the final image.

Here's how we approach that:

```dockerfile
# Stage 1: Restore Stage with Credentials
FROM mcr.microsoft.com/dotnet/sdk:7.0 AS restore-stage

ARG NUGET_API_KEY

RUN mkdir -p /root/.nuget
RUN echo "<?xml version='1.0' encoding='utf-8'?>" > /root/.nuget/nuget.config \
  && echo "<configuration>" >> /root/.nuget/nuget.config \
  && echo "  <packageSources>" >> /root/.nuget/nuget.config \
  && echo "    <add key='MyPrivateFeed' value='https://my.private.nuget.server/v3/index.json' />" >> /root/.nuget/nuget.config \
  && echo "  </packageSources>" >> /root/.nuget/nuget.config \
    && echo "   <apikeys>" >> /root/.nuget/nuget.config \
    && echo "     <add key='https://my.private.nuget.server/v3/index.json' value='${NUGET_API_KEY}' />" >> /root/.nuget/nuget.config \
    && echo "   </apikeys>" >> /root/.nuget/nuget.config \
  && echo "</configuration>" >> /root/.nuget/nuget.config

COPY . .
RUN dotnet restore

# Stage 2: Build Stage (no secrets)
FROM mcr.microsoft.com/dotnet/sdk:7.0 AS build-stage
COPY --from=restore-stage . .
RUN dotnet build -c Release -o /app

# Stage 3: Publish Stage (no secrets)
FROM mcr.microsoft.com/dotnet/sdk:7.0 AS publish-stage
COPY --from=build-stage /app .
RUN dotnet publish -c Release -o /publish

# Stage 4: Final Runtime Image (no secrets)
FROM mcr.microsoft.com/dotnet/aspnet:7.0 AS runtime
WORKDIR /app
COPY --from=publish-stage /publish .
ENTRYPOINT ["dotnet", "YourApplication.dll"]
```
As with the first example, build using the docker build command and appropriate credentials: `docker build --build-arg NUGET_API_KEY=<your_actual_api_key> .`

In this setup, our nuget config with credentials only exists within `restore-stage` and is not copied over to the subsequent build, publish, or runtime stages. This significantly improves the security posture of your final image.

Finally, another approach, which I find particularly useful for managing complex credential setups, is utilizing docker secrets. You could store the nuget config as a file in docker secrets, then mount that secret during the build phase. This ensures no sensitive data is baked into the layers and requires a docker swarm or equivalent to handle the secret management. This is probably the most robust method but adds some complexity to the docker orchestration setup. While a full example would be quite involved, here is a snippet that demonstrates where it would fit:

```dockerfile
# Stage 1: Restore Stage with Credentials (using Docker Secrets)
FROM mcr.microsoft.com/dotnet/sdk:7.0 AS restore-stage

# Mount the nuget.config secret
RUN mkdir -p /root/.nuget
COPY --from=secrets /run/secrets/nuget.config /root/.nuget/nuget.config
COPY . .
RUN dotnet restore

# Stage 2: Build Stage (no secrets)
... and so on ...
```
To use this approach, you'd first create a docker secret called `nuget.config`, then, during your build, use the `--secret` argument to mount it (using docker swarm):

`docker build --secret id=nuget.config,src=path/to/your/nuget.config .`

or using a docker compose file, for example:

```yaml
version: '3.9'
services:
  my-app-build:
    build:
      context: .
      dockerfile: Dockerfile
      secrets:
        - nuget.config
secrets:
  nuget.config:
    file: ./nuget.config
```

Regarding further reading, i’d suggest these resources. First, for a solid overview of .net and nuget, consult “pro .net 6” by Andrew Troelsen and Phillip Japikse – it has sections dedicated to package management that are very useful, especially combined with the official Microsoft nuget documentation. For security and container management, “docker in practice” by Ian Miell and Aidan Hobson is a great read, it has good material on managing docker secrets and ensuring secure builds. Understanding best practices for multi-stage builds is key to securing your docker build environment, and both resources touch on this.

In summary, properly restoring nuget packages from private feeds within docker builds is a multi-faceted issue, not simply a command that needs to be invoked. It requires a deep understanding of how docker builds operate and how to manage secrets securely. We've covered several practical approaches, from embedding credentials directly to utilizing docker secrets, each with different security trade-offs. Choose the technique that fits your needs and the security constraints of your specific scenario, and always prioritize the security of your build process. This is especially true when dealing with private packages. The cost of overlooking these details can be quite high.
