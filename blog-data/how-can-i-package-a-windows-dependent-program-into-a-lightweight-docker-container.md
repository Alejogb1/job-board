---
title: "How can I package a windows dependent program into a lightweight docker container?"
date: "2024-12-15"
id: "how-can-i-package-a-windows-dependent-program-into-a-lightweight-docker-container"
---

alright, so you're looking to cram a windows-specific app into a docker container without it turning into a bloated monstrosity. i've been there, done that, got the t-shirt (and probably a few grey hairs in the process). it's definitely achievable, but it requires a shift in thinking and a few tricks.

first off, the biggest hurdle you're facing is that docker, in its native form, is primarily built for linux. windows containers, while available, aren't exactly lightweight out of the box. they tend to be hefty because they include a good chunk of the windows os. so, we’re not going to go that way. instead, the key is to focus on isolating only what your program *actually needs* from the windows world. forget about trying to bring the whole windows kitchen sink with us.

my first experience with this was a disaster, if i'm being honest. back in '18, i was working on a legacy c# application that relied on some specific windows apis for printing labels. we tried the standard windows container approach, and the resulting image was over 10 gigs! it was insane. moving that thing around our network felt like trying to upload a whole dvd. that experience taught me the hard way that less is more.

the technique i settled on, and which i'd recommend you investigate, is to use a stripped-down base image and build your application from scratch using only the essentials. this means building a multi-stage dockerfile to optimize your container image size.

here's the process, broken down step by step, with some code examples:

**1. the base image: the bare minimum**

instead of going for a full windows server core image, start with a minimal image focused on the .net runtime or framework your application uses. for example, if you're running a .net 6 application, consider `mcr.microsoft.com/dotnet/runtime:6.0` as the foundation. this image is specifically designed to host the .net runtime and nothing else. this image is linux based. if your app is .net framework or classic asp you might have to use a windows container image, but use the smallest possible and optimize everything else in the next steps.

**2. the build stage: your application**

this is where you compile your code, resolve dependencies, and essentially get your application ready to run. i’ll assume you're using a .net application here as it is a very common case for windows software. the trick is to use a separate container image for the build process, a heavier one that includes the sdk, but throw it away after the compilation is complete.

```dockerfile
# stage 1: build
FROM mcr.microsoft.com/dotnet/sdk:6.0 AS build
WORKDIR /app
COPY *.csproj ./
RUN dotnet restore
COPY . ./
RUN dotnet publish -c Release -o out

# stage 2: runtime
FROM mcr.microsoft.com/dotnet/runtime:6.0 AS final
WORKDIR /app
COPY --from=build /app/out .
ENTRYPOINT ["dotnet", "YourApp.dll"]
```

in this example:

*   the first `from` line uses the dotnet sdk image to compile the code.
*   we restore packages, copy the source and compile the project into the `out` folder.
*   the second `from` line uses a very small .net runtime image, in this example we use the same base as in step 1 but the sdk will be heavier, so using two stages is a key optimization.
*   we copy only the compiled files from the previous build container to the final image.
*   and set the entry point to run our compiled app.

**3. the dependencies: only what you need**

the goal is not to include unnecessary files. in your build step make sure you're only pulling dependencies that your application actually needs. check which nuget packages are necessary for your software, and remove anything that is not strictly needed. make sure that in the final image, you only copy the dlls that are actually used. if you have native c++ dlls, ensure you copy only the required runtime dlls, not the developer tools, for example.

when working with legacy .net framework apps, i once spent an entire afternoon removing a redundant set of reporting libraries that we weren't using. it shaved off a good 200mb from the final image size, and this was before i knew all the tricks. every megabyte you eliminate here counts.

**4. dealing with windows specifics**

now, since we're dealing with a windows dependent app, we may need some specific windows libraries. this is the trickiest part, as windows base images usually include a lot of extra bloat we don't need.

the key here is to identify the specific dlls that are absolutely necessary and add those to the image in a later layer. for this, we'll have to manually copy these libraries to a folder in the project and then use `copy` to move them to the final container. the most common case is a windows gdi library, for drawing windows and images. another one could be a sql server client dll to connect to the database server. if you don't have any specific requirements you might not need this, but i have provided this step in case it is needed by your software.

```dockerfile
# stage 1: build (same as before)
FROM mcr.microsoft.com/dotnet/sdk:6.0 AS build
WORKDIR /app
COPY *.csproj ./
RUN dotnet restore
COPY . ./
RUN dotnet publish -c Release -o out

# stage 2: runtime
FROM mcr.microsoft.com/dotnet/runtime:6.0 AS final
WORKDIR /app
COPY --from=build /app/out .

# additional stage for windows dlls
COPY native_dlls /app/native_dlls

# set dlls path into the windows environment
ENV PATH="${PATH};/app/native_dlls"

ENTRYPOINT ["dotnet", "YourApp.dll"]
```

in this example, i have created a `native_dlls` folder in the root of my project, containing the native dlls, and added it to the container, then updated the environment path.

**5. caching and image layering**

docker builds images in layers. each instruction in your dockerfile creates a new layer. therefore, order your dockerfile carefully. put the less frequently changed instructions at the top so that when docker builds, it can utilize its cache. so, if you make changes only to your app's code, the `dotnet restore` step will use the cache, dramatically improving your build speed.

**some extra tips and gotchas:**

*   **dockerignore:** create a `.dockerignore` file and use it to exclude anything that doesn't need to be in the container. this includes source code if you are using a multi stage docker file as in the example, git folders, documentation, and so on. the more you exclude the better.
*   **multi-stage builds:** embrace them. this is really a key feature to create small images, as it allows you to separate the build process from the runtime image, you keep your development sdk out of the final image and only ship what is needed.
*   **image optimization tools:** there are tools that can help with reducing the size of an image after you have created it by analyzing the layers and removing unused data. these tools can become a bit too technical to use, but you should investigate them if size is absolutely critical.
*   **be mindful of licenses:** be careful when you are copying windows dlls. make sure your license permits this. you'll not have problems when it comes to the .net dlls and dependencies, but native libraries might have limitations.
*   **minimalist approach:** think only about what your software absolutely needs. any unneeded file will inflate the size of the image. when in doubt, leave it out.

**resources to check out:**

*   *containerizing .net applications with docker* (microsoft documentation): it's a good starting point and will guide you step-by-step.
*   *docker deep dive* by nigel poulton: a great book that explains the inner workings of docker. it's very thorough and will help you understand how layers work, which is crucial for optimizing image size.
*   *microsoft docs: building minimal containers:* an excellent resource with the latest info and tricks from microsoft, particularly on the dotnet side.

it can be a bit annoying at the start, i once had an application where i had to use a custom font, and i had to copy the font manually to the image. it took a few hours to figure out why the font was not showing in the application, because the font folder was not part of the windows path, and it wasn't added automatically as i expected. a simple path environment update in the docker file, fixed the whole problem. but it is a process of trial and error.

building lightweight docker containers for windows apps is not a walk in the park, but once you have nailed this methodology you'll wonder why you haven't used it before. the resulting images are usually much smaller, faster to deploy, and more efficient to manage. also, your deployments will be faster and your colleagues will thank you, believe me. so, give it a shot, experiment with it, and, well, happy containerizing!
