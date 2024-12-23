---
title: "Why am I unable to evaluate symlinks in a Dockerfile path on an MB Pro M1 Max?"
date: "2024-12-23"
id: "why-am-i-unable-to-evaluate-symlinks-in-a-dockerfile-path-on-an-mb-pro-m1-max"
---

Let’s tackle this. I recall a similar conundrum during a large-scale microservices deployment on a hybrid architecture, where we extensively used symlinks to manage code versions. It became quite apparent, particularly on arm64 architectures like the M1 Max, that symlink behavior within docker build contexts can introduce some frustrating inconsistencies. The core issue isn't usually a bug, but rather a consequence of how docker manages the build context and how symlinks are resolved at different stages of the process.

Specifically, what you're likely experiencing isn't a direct inability of the Dockerfile parser to handle symlinks on an M1 Max. The parser itself isn't particularly architecture-dependent. The challenge emerges from the interaction between your host machine's filesystem and the isolated build context that docker creates. When you instruct docker to build an image (e.g., `docker build .`), docker first copies the entire *context* – typically the contents of the directory where you issue the build command – into a temporary area, isolated from your host filesystem. This copy process is not simply a one-to-one mirroring of your files and folders, and this is where symlinks cause complications.

Docker typically 'follows' symlinks during the build context copy, effectively resolving the link and copying the *target* of the symlink into the build context, instead of creating a new symlink. This means that at the time your Dockerfile executes, any command that references the symlink will not operate on the link itself, but on the files it points to. This resolution behavior is generally desirable, but it becomes problematic when you specifically *want* the symlink to remain a symlink within the container. This is especially critical if you're attempting to create symlinks within the image during the build process using instructions such as `RUN ln -s ...` within your Dockerfile, and those symlinks depend on files existing within the build context – files that are already dereferenced and no longer available as symlinks. The difference on the M1 Max stems from subtleties in the underlying file system and how docker’s build context copy mechanisms interact with it.

To illustrate this, let's examine three scenarios with accompanying code examples:

**Scenario 1: Initial Dereferencing**

Let's say you have this structure on your host machine:

```
project/
├── original_file.txt
├── link_to_original.txt -> original_file.txt
└── Dockerfile
```

And your Dockerfile contains:

```dockerfile
FROM ubuntu:latest

COPY . /app
RUN ls -l /app
```

When you build using `docker build .`, the output of `ls -l /app` inside the container will show *two* files – `/app/original_file.txt` and `/app/link_to_original.txt`. The crucial detail is that `/app/link_to_original.txt` will *not* be a symlink but rather a copy of the content of `original_file.txt`. This is because Docker copied *content* rather than preserving the symlink nature.

**Scenario 2: Symlink Creation Inside the Dockerfile**

Now consider a setup where we expect to operate on a file using a symlink created *inside* the docker build process. We have the same project structure as in Scenario 1 and the following `Dockerfile`:

```dockerfile
FROM ubuntu:latest

COPY . /app
RUN ln -s /app/original_file.txt /app/link_created_in_docker.txt
RUN ls -l /app
RUN cat /app/link_created_in_docker.txt
```

Here, the `ln -s` command creates a symlink within the container’s file system, `/app/link_created_in_docker.txt`, which points to `/app/original_file.txt`. Inside the running container, this will indeed show a symlink after the `RUN ls -l /app` operation, and `cat` will successfully read the original file through that symlink. This works because the symlink creation happens entirely inside the isolated environment and all referenced paths are within that context, and all paths correctly exist. However, it is important to know that any symlinks that are part of the build context are not copied as symlinks, they are resolved, and copied as files, therefore, operations relying on the existence of symlinks in the build context will not work as expected.

**Scenario 3: The Problematic Case (Path Evaluation)**

Now let's address the scenario that's probably causing your issue. Suppose your project structure looks like this:

```
project/
├── source_dir/
│   └── actual_file.txt
├── symlink_dir -> source_dir
└── Dockerfile
```

And you have a `Dockerfile` that attempts to access the file through the symlink:

```dockerfile
FROM ubuntu:latest

COPY . /app
RUN cat /app/symlink_dir/actual_file.txt
```

You might anticipate this would work, but it will likely fail. The output during the `docker build` will likely produce an error stating that `/app/symlink_dir/actual_file.txt` does not exist. This is because when docker copies the build context into the build environment, `/symlink_dir` is not a symlink, but a folder copied from what that link resolved to, hence `source_dir`. As a result, at build time `/app/symlink_dir` will have been copied as a directory called `symlink_dir` inside the container, but the symlink that pointed to it from your host machine is lost. The resulting path, `/app/symlink_dir/actual_file.txt` resolves to what was inside your source folder. In summary: the file is present, but the path used in the dockerfile is inconsistent with what exists in the docker context.

**Resolution Approaches**

There aren’t simple switches to alter how docker handles symlinks in the build context copy process. Instead, you need to work *around* this behavior, depending on your specific needs:

1.  **Avoid Symlinks in the Build Context (Where Possible):** If you can, structure your project so that the Dockerfile doesn't depend on symlinks in the build context. Restructuring can often make your setup more portable and resilient.

2.  **Use Absolute Paths Inside the Container:** As seen in scenario 2, if you're creating symlinks inside the Dockerfile, ensure all referenced paths are absolute, and exist within the image. Avoid relying on relative paths that depend on how the initial context is structured.

3.  **Explicitly Create Symlinks in Dockerfile:** If you need the symlink inside your image, use `RUN ln -s ...` in your Dockerfile, explicitly creating the required symlinks *after* the context is copied. The symlink target should exist inside the container file system.

**Resource Recommendations**

For a deeper understanding, I’d highly recommend exploring these resources:

*   **"Docker Deep Dive" by Nigel Poulton:** This book covers the core concepts of docker, including the intricacies of build contexts and image layering. It provides a very thorough background.
*   **The official Docker documentation:** Focus specifically on the `Dockerfile` reference and the `docker build` command. These are essential to understand the precise behavior of these mechanisms, and they get updated frequently as docker matures.
*   **Advanced Linux Programming by Mark Mitchell, Jeffrey Oldham and Alex Samuel**: this book does not speak directly about docker, but contains essential information about file systems and symlinks, that is essential to understanding why certain behaviors occur.

In my experience, the most reliable method is often restructuring projects to avoid the dependency on symlinks within the initial build context. Docker’s build process prioritizes deterministic behavior and the current behavior, while seemingly cumbersome, makes the process more predictable across different development environments. Addressing issues by changing the structure of your project and being explicit in the `Dockerfile` leads to more resilient builds in the long run. I hope this detailed explanation helps clarify your situation and provides you with actionable ways to proceed.
