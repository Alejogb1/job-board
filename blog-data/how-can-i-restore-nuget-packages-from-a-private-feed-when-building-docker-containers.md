---
title: "How can I Restore NuGet Packages from a Private Feed when building Docker Containers?"
date: "2024-12-23"
id: "how-can-i-restore-nuget-packages-from-a-private-feed-when-building-docker-containers"
---

Okay, let's tackle this one. I've bumped into this exact scenario more times than I care to remember, especially when migrating legacy .net projects into containerized environments. The core issue, restoring nuget packages from a private feed within a docker build, isn’t inherently complex, but it does require careful orchestration to avoid common pitfalls. It’s about securely providing credentials and ensuring the build process has access to those feeds, all while maintaining a clean and efficient docker image. I'll walk you through my go-to methods, highlighting where things can often go sideways.

Firstly, the challenge originates from the fact that docker builds typically occur in an isolated environment, where the containerization engine has limited access to your development machine or your corporate network's private nuget server. Directly referencing private feeds within a `.csproj` file will fail during a docker build if no authentication mechanism is in place or if your container’s network configurations are incorrect. You can’t simply assume the docker build context will magically inherit your development environment’s configuration.

Over my years, I’ve found three primary approaches that handle this well. Each has its strengths and weaknesses, so the best choice hinges on your specific circumstances and security posture.

**1. Passing Credentials as Build Arguments**

This approach involves passing the necessary credentials, typically a username/password or an api key, as build-time arguments. This is relatively straightforward to implement, and it’s often my starting point for rapid prototyping. The critical detail here is that you must be extremely careful not to inadvertently embed these credentials into the final docker image.

Here’s a code snippet for a `Dockerfile` demonstrating this:

```dockerfile
# syntax=docker/dockerfile:1
FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
WORKDIR /app
COPY *.csproj .
RUN dotnet restore -s "https://my-private-feed.example.com/v3/index.json" -u $NUGET_USERNAME -p $NUGET_PASSWORD

COPY . .
RUN dotnet publish -c Release -o out

FROM mcr.microsoft.com/dotnet/aspnet:8.0
WORKDIR /app
COPY --from=build /app/out .
ENTRYPOINT ["dotnet", "MyApplication.dll"]
```

and here’s an example of how you might invoke the docker build command:

```bash
docker build -t my-app --build-arg NUGET_USERNAME="my_username" --build-arg NUGET_PASSWORD="my_password" .
```

In this snippet, we’re:

*   Using a multi-stage docker build, so the credentials are only present in the build stage. They do not contaminate the final image.
*   Passing `NUGET_USERNAME` and `NUGET_PASSWORD` as build arguments.
*   Using these arguments in the `dotnet restore` command within the build stage.

This method is functional and relatively easy to implement, but *never* store credentials directly in your dockerfile. Additionally, it is not best practice to expose credentials directly on the command line. Instead, utilize an environment file, or a secrets manager. You must use build args only within your `FROM … AS build` stage and use `COPY --from=build` for any subsequent stage. This ensures security of your credentials.

**2. Using a `nuget.config` File**

A slightly more sophisticated approach involves utilizing a `nuget.config` file to manage feed configurations and authentication, including credentials. It's a better practice to use a `.nuget/nuget.config` file which allows for a more organized configuration and avoids the direct exposure of credentials in a docker command or in your build script. You also gain the ability to configure multiple feeds and credential providers.

Here's how that looks inside your `Dockerfile`

```dockerfile
# syntax=docker/dockerfile:1
FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
WORKDIR /app
COPY nuget.config .nuget/
COPY *.csproj .
RUN dotnet restore

COPY . .
RUN dotnet publish -c Release -o out

FROM mcr.microsoft.com/dotnet/aspnet:8.0
WORKDIR /app
COPY --from=build /app/out .
ENTRYPOINT ["dotnet", "MyApplication.dll"]

```

And your `nuget.config` file would look something like this:

```xml
<?xml version="1.0" encoding="utf-8"?>
<configuration>
  <packageSources>
    <add key="my-private-feed" value="https://my-private-feed.example.com/v3/index.json" />
  </packageSources>
 <packageSourceCredentials>
    <my-private-feed>
        <add key="Username" value="my_username" />
        <add key="ClearTextPassword" value="my_password" />
    </my-private-feed>
 </packageSourceCredentials>
</configuration>

```

Key takeaways here:

*   We copy the `nuget.config` file into the container at a well-known location before the `dotnet restore` command. The default location is at `.nuget/nuget.config` next to the solution or project file.
*   The `nuget.config` file is typically excluded from version control and passed to the container build as a build arg, or copied via the build context or mounted volume.
*   Note the use of `ClearTextPassword`. While this works, for production systems it’s far better to use encrypted credentials or another credential provider (e.g., Windows Credential Manager, Azure DevOps service connections), where available. You can then use the 'password' key instead.

This method is more flexible for managing multiple feeds and simplifies the dockerfile, but also needs careful credential management to avoid storing your credentials in source control.

**3. Leveraging a Credential Provider Plugin or Environment Variables**

The final, and what I generally recommend for more robust scenarios, is employing a credential provider plugin. A credential provider plugin enables a more secure means of providing credentials during the build process. It interacts with the `dotnet` cli's credential acquisition workflow without embedding secrets directly in configuration files or build args. This approach also allows for more advanced auth scenarios such as Azure Active Directory authentication using managed identities.

For example, the Azure Artifacts Credential Provider is a .NET Tool that handles the acquisition of Azure Artifacts feeds using Azure Active Directory and managed identities.

Here's a revised `Dockerfile` that uses a credential provider (this example assumes Azure DevOps). We'll also leverage environment variables for the nuget feed url.

```dockerfile
# syntax=docker/dockerfile:1
FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
WORKDIR /app
COPY *.csproj .
RUN dotnet tool install -g Azure.Artifacts.CredentialProvider
RUN dotnet restore -s $NUGET_PRIVATE_FEED_URL
COPY . .
RUN dotnet publish -c Release -o out

FROM mcr.microsoft.com/dotnet/aspnet:8.0
WORKDIR /app
COPY --from=build /app/out .
ENTRYPOINT ["dotnet", "MyApplication.dll"]

```

Then invoke the command with the feed URL:

```bash
docker build --build-arg NUGET_PRIVATE_FEED_URL="https://my-private-feed.example.com/v3/index.json"  -t my-app .
```

Key points:

*   Here, we install the `Azure.Artifacts.CredentialProvider` .NET tool during the build phase.
*   The `dotnet restore` command uses the provided feed url from the environment variable.
*   The plugin automatically detects and uses managed identities or other configured credentials, depending on the environment.
*   You'll need to ensure that the container's execution environment has the correct permissions and configuration to successfully authenticate using the plugin. This is typically achieved by deploying the container with a managed identity.

This is my preferred method for production builds. It minimizes credential exposure, provides more complex authentication solutions, and integrates nicely with modern cloud-based build pipelines.

**Resource Recommendations**

For more in-depth information, I suggest the following resources:

1.  **"Docker in Practice" by Ian Miell and Aidan Hobson Sayers**: This book offers a solid understanding of practical docker usage, including multi-stage builds and best practices. I would recommend this for a broad understanding of container best-practices.
2.  **Microsoft’s Documentation on NuGet**: The official nuget documentation on their site is an invaluable resource for all things related to configuring package sources and credentials, particularly for more advanced setups. Specifically, the articles on the configuration file and credential providers are worth exploring.
3.  **The official .net documentation**: Specifically, explore the documentation on the `dotnet restore` command and its various options, including configuration and authentication.

In summary, restoring nuget packages from private feeds during docker builds demands careful attention to security. Avoid storing credentials directly within your dockerfiles or using command line arguments. Leverage `nuget.config` for structured configuration, use credential providers for secure authentication, and remember to follow the principle of least privilege when configuring access to your private feeds. While all three of the above methods work, I would recommend using method three when you are running in cloud environments that have a built in credential provider. Otherwise, the second method (with secure credentials) may work fine for smaller use cases.
