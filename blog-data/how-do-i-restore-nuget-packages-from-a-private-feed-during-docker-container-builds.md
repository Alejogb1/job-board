---
title: "How do I restore NuGet packages from a private feed during Docker container builds?"
date: "2024-12-23"
id: "how-do-i-restore-nuget-packages-from-a-private-feed-during-docker-container-builds"
---

,  It’s a problem I’ve certainly encountered more than once, particularly during those early cloud-native migrations. Handling NuGet packages in Docker builds, especially from private feeds, requires a solid understanding of both Dockerfile mechanics and NuGet configuration. I’ve seen it go sideways in several ways, mostly revolving around authentication and caching. Here’s how I approach it, drawing from a few hard-earned lessons.

First off, consider the fundamental issue: by default, Docker build processes are isolated and, crucially, lack the context of your development environment’s configured NuGet sources. They don't inherently *know* about your internal package feeds, nor do they possess any credentials to access them. This is deliberate, for security reasons. So, we have to explicitly tell the container build process where to look for packages and how to authenticate if necessary.

The approach hinges on two main aspects: configuring NuGet within the Docker build context and, often, handling authentication via environment variables or some other secure mechanism, especially if it's a protected feed. We can achieve this fairly systematically.

The initial step always involves setting up a `nuget.config` file, or modifying an existing one if you're already using one in your project. This file, placed at the root of your project directory, needs to define your private package source. The fundamental structure might resemble something like this:

```xml
<?xml version="1.0" encoding="utf-8"?>
<configuration>
  <packageSources>
    <add key="my-private-feed" value="https://my.private.nuget.feed/v3/index.json" />
  </packageSources>
  <packageSourceCredentials>
    <my-private-feed>
      <add key="Username" value="%NUGET_USERNAME%" />
      <add key="ClearTextPassword" value="%NUGET_PASSWORD%" />
    </my-private-feed>
  </packageSourceCredentials>
</configuration>
```

Notice the use of environment variables `%NUGET_USERNAME%` and `%NUGET_PASSWORD%`. I tend to favor this approach because it separates credentials from the configuration itself, which is critical for security. This `nuget.config` file will eventually get copied into our build context. Remember to avoid committing actual credentials to your version control system. It's also important to use https feeds, not http.

Now, let's translate this into a Dockerfile structure. Here’s a straightforward example, assuming we're working with a .net application:

```dockerfile
FROM mcr.microsoft.com/dotnet/sdk:7.0 AS build
WORKDIR /app

COPY nuget.config ./
COPY *.csproj ./
RUN dotnet restore --configfile nuget.config

COPY . ./
RUN dotnet publish -c Release -o /out

FROM mcr.microsoft.com/dotnet/aspnet:7.0 AS runtime
WORKDIR /app
COPY --from=build /out ./
ENTRYPOINT ["dotnet", "YourApp.dll"]
```

Here’s what’s happening, step by step:

1.  **`FROM mcr.microsoft.com/dotnet/sdk:7.0 AS build`**:  We start with a build image containing the .net SDK, as we're publishing an application.
2.  **`WORKDIR /app`**: We establish a working directory within the container.
3.  **`COPY nuget.config ./`**:  We copy the NuGet configuration file from the build context into the container's `/app` directory.
4.  **`COPY *.csproj ./`**:  Copy over the project files.
5.  **`RUN dotnet restore --configfile nuget.config`**: This critical line invokes `dotnet restore`, explicitly instructing the restore operation to use the provided `nuget.config`. The `--configfile` option is paramount here. This will pull down dependencies from your private feed, provided the credentials are set.
6.  **`COPY . ./`**: Copy the rest of your code into the container.
7.  **`RUN dotnet publish -c Release -o /out`**: Finally we publish the code to a separate folder.

We then switch to a runtime image in the subsequent part of the Dockerfile.

8.  **`FROM mcr.microsoft.com/dotnet/aspnet:7.0 AS runtime`**: This pulls the runtime image.
9.  **`WORKDIR /app`**: Sets up a working directory for the application.
10. **`COPY --from=build /out ./`**: Copies only the published code to the runtime image
11. **`ENTRYPOINT ["dotnet", "YourApp.dll"]`**: Sets the entry point of the container.

Now, during the Docker build, you'll need to supply the environment variables. This can be achieved in a few ways. For local development, you might set them in your shell:

```bash
export NUGET_USERNAME="your-username"
export NUGET_PASSWORD="your-password"
docker build -t my-app .
```

However, for continuous integration pipelines or production scenarios, it's much more secure to use secrets management within your CI/CD platform or a dedicated secrets store. This is something I advocate strongly for.

A slightly more complex scenario might involve authenticating with an Azure Artifacts feed, where the credentials usually involve an access token rather than a username/password. In that case, your `nuget.config` might look something like this:

```xml
<?xml version="1.0" encoding="utf-8"?>
<configuration>
  <packageSources>
    <add key="my-azure-artifacts-feed" value="https://your-org.pkgs.visualstudio.com/_packaging/your-feed/nuget/v3/index.json" />
  </packageSources>
  <packageSourceCredentials>
    <my-azure-artifacts-feed>
      <add key="AccessToken" value="%NUGET_PAT%" />
    </my-azure-artifacts-feed>
  </packageSourceCredentials>
</configuration>
```

And the Docker build command would involve setting the `NUGET_PAT` environment variable instead, something like:

```bash
export NUGET_PAT="your-personal-access-token"
docker build -t my-app .
```

The key point here is the consistent use of environment variables, ensuring that credentials aren't baked into the Dockerfile.

Another optimization I would mention is to use multi-stage builds (as shown above) to keep the final image slim, as the build tools are no longer required in the final container image. We use the build stage to retrieve and process everything, and then copy only what we require.

Lastly, it is useful to examine the output of your builds. When encountering issues it’s often helpful to see the console output from the `dotnet restore` command. Errors often indicate issues with the configuration file or the environment variables.

To dive deeper into these areas, I highly recommend the following:

*   **"Pro NuGet" by Kurt Dillard**: Provides a comprehensive look at NuGet configurations. It's slightly older but still very relevant for understanding the intricacies of NuGet.
*   **The Docker documentation**: Crucial for understanding build contexts and the mechanics of Dockerfiles. Pay particular attention to sections on build optimization and working with secrets.
*   **Official Microsoft documentation on .NET CLI tools and package management**: Details on how to use `dotnet restore`, and the `nuget.config` format.
*   **Documentation for your specific private package feed provider**: This often has specific requirements, particularly around authentication methods.

Working with private NuGet feeds in Docker builds might seem challenging initially, but, as I’ve demonstrated, with the correct configuration and authentication practices, the process is quite manageable and can be integrated into a robust development and deployment workflow. It’s all about understanding the context each layer operates in, and how to bridge the gap by providing the necessary information.
