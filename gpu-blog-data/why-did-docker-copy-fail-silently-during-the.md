---
title: "Why did Docker COPY fail silently during the build?"
date: "2025-01-30"
id: "why-did-docker-copy-fail-silently-during-the"
---
Docker’s `COPY` instruction, when it fails silently during a build process, typically points to a mismatch between the source path specified in the Dockerfile and the actual file structure within the build context. I've personally encountered this several times, usually when rapidly iterating on Dockerfiles or after making changes to directory layouts, and it’s never immediately obvious. The silent failure stems from Docker’s design to proceed with the build even if a `COPY` instruction finds no matching files or directories at the source, opting instead for a warning level log message (often easily missed) rather than an error. This behavior, while seemingly permissive, is intentional to accommodate scenarios where a Dockerfile is designed to be flexible across different build environments that might not always have the exact same set of source files available.

The core reason for the silent failure is that the Docker build process operates within a specific context, which is the directory or tar archive provided to the `docker build` command (e.g., `docker build .` uses the current directory as the build context). The source path in the `COPY` command is relative to this build context, not the location of the Dockerfile itself or your host's filesystem. If the specified source path does not exist within this build context or if it’s mistyped or an incorrect relative path, Docker will proceed as if the instruction has completed successfully without modifying the target image, leading to an application that lacks the required resources.

Let’s consider a few scenarios where this can occur, along with practical examples to demonstrate the problem.

**Scenario 1: Incorrect Relative Path**

Imagine a Dockerfile in a project structure like this:

```
project/
├── Dockerfile
└── src/
    └── app.py
    └── requirements.txt
```

If the Dockerfile contains the following `COPY` instruction, the build will proceed, but the files will not be copied correctly:

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY src . 
CMD ["python", "app.py"]
```

The Dockerfile is at the `project` root level, and the intention is to copy the entire `src` directory and its contents into the `/app` directory of the Docker image. However, in this case, the silent failure arises because the build context is set as `project/`, therefore the Docker engine sees `src` as a directory that's available to copy. However, if the build command was executed from a parent directory, such as using the command `docker build project/`, then the root of the build context is `project`, and the `COPY src .` would work as intended. However, if the command `docker build .` was executed from within `project` the copy operation would also work as intended. To demonstrate the issue I'll build the image in a way that replicates an incorrect build context and show the resulting image does not have the contents expected.

Here's an example of how that could happen. The build command is executed from the root of a folder that contains the `project` directory, like this:

`docker build project/ -t my-faulty-image`

The contents of my Dockerfile for this scenario are:

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY src . 
CMD ["python", "app.py"]
```

And the files are arranged like this:

```
my-root/
├── project/
│   ├── Dockerfile
│   └── src/
│       ├── app.py
│       └── requirements.txt
```

The following script `verify.sh` will be used after building the docker image to test if the copy command was successful:

```bash
#!/bin/bash

docker run --rm my-faulty-image ls /app
```

After executing the commands above I will run the `verify.sh` script. The result is a directory listing of `/app` that shows nothing but the root folder, indicating the copy failed:
```
root@my-computer:/my-root# ./verify.sh
```

This confirms that the silent failure resulted in the `src` folder not being copied. The solution for this issue is to execute the docker build command from within the `project` directory: `docker build . -t my-fixed-image` or to move the Dockerfile to a higher directory within your projects file structure, and use the correct source path. If I use the following `COPY` instruction instead `COPY ./src .` it will also copy the contents as expected.

**Scenario 2: Typographical Error**

Consider the same directory structure, but this time a subtle typo has crept into the `COPY` instruction:

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY scr/ .
CMD ["python", "app.py"]
```

Here, 'src' has been misspelled as 'scr'. Because the build context does not have a directory named 'scr', `COPY` does not find anything to copy, but still continues with the build. This results in an image without the expected source files, and the docker build command provides no immediate indication of this issue. Again, this silent error is easy to overlook, especially when reviewing lengthy Dockerfiles. This example was provided in the context of the first example. If I use the same file structure and modify the docker file to have a typo, then the output of `verify.sh` using the resulting image will be identical to the previous example. If a typo was made using an absolute path the result would be identical. This issue can be solved with increased attention to detail and using a linting tool.

**Scenario 3: Excluding Files or Directories from the Context**

Docker allows exclusion of files and directories using a `.dockerignore` file within the build context. If files or directories specified in the `COPY` command are unintentionally included in the `.dockerignore` file, Docker won't copy them, but also doesn't report an error.

For example, with the same directory structure, a `.dockerignore` file could be added like this:

```
project/
├── .dockerignore
├── Dockerfile
└── src/
    └── app.py
    └── requirements.txt
```

Where the `.dockerignore` contains:

```
src/
```

And the Dockerfile contains:

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY src .
CMD ["python", "app.py"]
```

The result of building this image will also be an empty `/app` folder and a silent failure of the `COPY` command. Even if the `COPY` instruction is correct, the build context is correct and no typo has been made, the files will still be ignored because they were excluded from the build context. This issue can be difficult to identify because the `.dockerignore` file can exist at different levels of the project.

**Recommendations for Troubleshooting:**

1.  **Explicitly Check Build Context**: Always ensure you are aware of what directory is being used as the build context. When in doubt, build from the root of your project directory.
2.  **Double-Check Source Paths**: Verify the correctness of all source paths specified in the `COPY` command against your actual project structure. A tool like tree can be helpful for this, or manually browsing to ensure the file structure matches your expectations.
3.  **Review `.dockerignore`**: Scrutinize the `.dockerignore` file to confirm no unintentionally excluded files. Use the `--verbose` flag in conjunction with `docker build` for more detailed output, including which files and folders are being ignored.
4.  **Utilize Docker Buildkit**: The Docker Buildkit, enabled by setting the `DOCKER_BUILDKIT=1` environment variable, offers an improved build process, including more granular logging. However, it won't fix a typo, it will simply provide verbose information about the docker build.
5. **Test in Isolated Steps**: Do not make multiple changes in a Dockerfile. Implement a single change and rebuild. In this way, when an issue like this arises you know it is related to the most recent change you have made.
6. **Implement Automated Testing:** Create an automated script that tests if the copy command was successful. This was demonstrated earlier with the `verify.sh` script. This script can be run each time the docker image is built to ensure all dependencies were copied correctly.

These steps, in conjunction with a clear understanding of the build context and the silent failure characteristics of the `COPY` instruction, will significantly improve the troubleshooting process. Remembering to start from the basics, checking simple typos, and confirming the build context will allow you to avoid unnecessary headaches.
