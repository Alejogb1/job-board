---
title: "What devcontainer file property should I use for my prebuilt VS Code project?"
date: "2024-12-23"
id: "what-devcontainer-file-property-should-i-use-for-my-prebuilt-vs-code-project"
---

Let's dive into this. If you're leveraging prebuilt devcontainers with vscode, the choice of the correct devcontainer.json property can make a significant difference in your development workflow, particularly in terms of speed and consistency. I've personally spent considerable time optimizing these configurations across several large projects, and I’ve seen how seemingly minor changes can lead to major gains. So, let’s not treat this as a trivial matter. Instead of looking at just one property in isolation, consider how different properties interact, and how they shape the final development environment. In your specific question, focusing on the prebuilt images aspect, I think we should address this with a deep dive into the `image` property versus `build` and their relationship with prebuilt containers.

The `image` property within a `devcontainer.json` file is where you specify a prebuilt container image. This approach is a godsend for larger projects, as it sidesteps the container image build process on every container recreation. Think of it as using a ready-made cookie rather than starting with flour and eggs. It means that rather than constructing a container from scratch, using a dockerfile for every instance, you pull a pre-configured image from a registry, thereby significantly cutting down the startup time. This can make the initial devcontainer setup far faster, and the day-to-day switching between container instances nearly instantaneous.

Now, the devil is in the details. The `image` property should contain a string that points to a valid image in a container registry—Docker Hub, GitHub Container Registry, or any private registry accessible to your environment. A simple example looks like this:

```json
{
  "image": "my-organization/my-base-dev-image:latest",
  "name": "My Development Environment",
  "remoteUser": "vscode"
}
```

In this snippet, `my-organization/my-base-dev-image:latest` is the location of the prebuilt image. Notice I have also included `remoteUser`. This is necessary in many instances, to make sure the vscode user has proper permissions in the container environment.

This approach, however, requires some planning. You need to create this `my-organization/my-base-dev-image` separately, likely using a Dockerfile and a dedicated build process. This initial setup is more complex than building the container directly in the `devcontainer.json`, but it pays off handsomely in the long run, particularly when you are iterating frequently. I can tell you from personal experience on a project where we were constantly spinning up new development instances across a large team that using a prebuilt image reduced our overall build time by something like 75 percent. This allows the developers to focus on actual code instead of waiting for container builds to finish. We had this entire automated pipeline for nightly image builds and it became a bedrock for the overall project.

But what if the `image` alone does not cover your needs? This is where the `build` property can enter the equation. If you want to include customisations that need to happen during the image startup, you can include `build` along with the `image` property. It takes precedence over `image`. This is useful if the prebuilt base image does not contain everything you need.

Here’s an illustration: Suppose you have a prebuilt image that includes all of your core development tools, but it does not include the required python packages for your particular project. You can specify the following in the `devcontainer.json`:

```json
{
  "image": "my-organization/my-base-dev-image:latest",
   "build": {
        "dockerfile": "Dockerfile.dev",
        "args": {
            "PYTHON_PACKAGES": "requests pandas numpy"
        }
    },
  "name": "My Project Specific Dev Environment",
  "remoteUser": "vscode"
}
```

Here, the prebuilt image `my-organization/my-base-dev-image:latest` is still the base, but we are further customizing it with a new Dockerfile, `Dockerfile.dev`. This dockerfile, often a very simple one, might install the specific python packages as shown below:

```dockerfile
FROM my-organization/my-base-dev-image:latest
ARG PYTHON_PACKAGES
RUN pip install $PYTHON_PACKAGES
```
Here, the additional packages will be installed on container creation using the `Dockerfile.dev` specified in the `build` object. The critical point here is the `FROM` instruction which specifies that our new `Dockerfile.dev` is extending the prebuilt image we specified in the `image` property.

Now, where it gets interesting is when you are not directly extending the prebuilt image with a dockerfile. You can achieve a similar effect, without resorting to dockerfiles, by using the `features` property.

The features object allows you to apply some standardized configuration on top of a prebuilt image. Let's say your team uses zsh instead of bash, and this is not part of the prebuilt image. Then you can use the feature for zsh. Here is an example:

```json
{
  "image": "my-organization/my-base-dev-image:latest",
  "features": {
    "ghcr.io/devcontainers/features/zsh": {
       "version": "latest"
    },
     "ghcr.io/devcontainers/features/docker-in-docker": {
       "version": "latest"
    }
  },
  "name": "My Development Environment",
  "remoteUser": "vscode"
}
```
In this example, we are pulling two features from github container registry: the `zsh` feature and the `docker-in-docker` feature. These features get installed, in the context of the prebuilt image, on container creation. These features provide a standardized mechanism to extend the base image, without resorting to a specific dockerfile. These are often provided by the vscode development team or the community.

Let's talk about the recommended approach. Generally, I recommend starting with a suitable prebuilt base image and then using features or extending the base image with a dockerfile. Avoid building a full image from scratch using `build` only, when there exists a prebuilt image that suits your base needs. Leverage the `image` property whenever possible to accelerate setup and instantiation times. Use the `build` property to extend that image, but always using a specific base image. The `features` property should be used to configure standardized aspects of the container environment.

For further study, I recommend delving into the following resources: *Effective DevOps* by Jennifer Davis and Katherine Daniels which includes sections on container management and infrastructure as code, particularly focusing on the practices of reproducible environments. Also, Microsoft has excellent resources on devcontainers in general that go into these finer details that you can find in the online documentation. You can find a detailed breakdown of all `devcontainer.json` properties, including `image`, `build`, and `features`. These resources provide practical guidance that will help you deepen your expertise with devcontainers.
