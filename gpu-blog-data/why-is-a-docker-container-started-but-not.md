---
title: "Why is a Docker container started but not running?"
date: "2025-01-30"
id: "why-is-a-docker-container-started-but-not"
---
Docker containers transitioning from a ‘created’ to a ‘running’ state, and then seemingly failing, is a common point of friction when beginning with containerized applications. The crucial aspect lies not just in the container’s creation itself, but in the execution of the primary process within that container. A container, in its essence, is an isolated process; it's not a virtual machine with a running operating system. If the process specified by the Dockerfile’s `CMD` or `ENTRYPOINT` instruction either exits immediately, encounters an unhandled error, or is never reached due to a misconfiguration, the container will appear started but will not be running. The Docker daemon reports the container as having existed with a specific exit code, though not in a continuous execution state.

The lifecycle of a Docker container is straightforward, yet often misunderstood. When `docker run` is invoked, Docker first creates a container based on the specified image. This involves setting up the necessary namespaces and cgroups to isolate the container's process and resources. Next, the command specified in the image's metadata (either `CMD` or `ENTRYPOINT`) is executed. A container is considered “running” only as long as this primary process, and any processes it spawns as descendants, is alive. The moment this primary process terminates, the container exits, rendering it “not running,” despite existing. It’s critical to understand that Docker does not keep a container running in the absence of its designated application process.

Let’s consider a situation where you've specified a shell command as the `CMD` instruction in a Dockerfile, designed to print a string. This could manifest as an example of a container that starts, prints its message and exits as demonstrated below:

**Example 1: A simple echo command**

```dockerfile
FROM alpine:latest
CMD ["echo", "Hello, Docker!"]
```

Building this image, and then running a container, reveals how a container can exit rapidly:

```bash
docker build -t my-echo-image .
docker run my-echo-image
```

Upon running this, “Hello, Docker!” will print to your terminal, followed by the container exiting. It's not actively "running" in the way one might expect. The `docker ps -a` command would show this container with an exit code of 0, signifying a successful command execution but not a persistent process.

This is frequently a point of misunderstanding; if the `CMD` or `ENTRYPOINT` instructions involve a command that only executes once, and then terminates, the container's lifecycle will reflect this brief execution. If we wanted a long-running process, we would need to specify a service that continuously runs such as web service, or a long looping program.

In contrast, consider a scenario where the intended application within the container is a web server. If this server throws an uncaught exception, the server process will terminate, causing the container to immediately exit. This might occur due to various reasons, such as configuration errors, incorrect file paths, or missing dependencies.

**Example 2: A problematic web server**

Imagine an attempt to run a simple Python Flask server, but the application code contains an import error. The Dockerfile is below. Note how there is a missing file in our virtual directory.

```dockerfile
FROM python:3.9-slim-buster
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "app.py"]
```

The `requirements.txt` file would specify something like `Flask`. Below is the `app.py` file:

```python
from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return "Hello from Flask!"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
```

This will function perfectly, but if the import statement was changed to `from flsk import Flask`, you would encounter an error.

Executing the below commands show this behaviour.

```bash
docker build -t my-broken-web .
docker run -p 5000:5000 my-broken-web
```

When this container is run, the `docker logs <container_id>` command would reveal a traceback indicating an `ImportError`, and the container would stop with an error code. This situation mirrors the previous scenario: the container starts by executing the `CMD`, but that process immediately crashes, thus ending the container's active status. Debugging here would involve careful examination of the logs for errors and a check of all dependencies.

Finally, the `ENTRYPOINT` instruction, a close counterpart to `CMD`, presents another opportunity for confusion. If an `ENTRYPOINT` script contains an error or does not execute the intended command properly, the container will also fail to keep running. The `ENTRYPOINT` differs slightly from `CMD`. It provides a primary, immutable entry point for the containerized process. Let’s examine a scenario where an entrypoint script fails by attempting to execute a nonexistent binary:

**Example 3: A faulty entrypoint script**

Here is a Dockerfile that creates a simple entry point script called `run.sh`.

```dockerfile
FROM alpine:latest
COPY run.sh /
RUN chmod +x /run.sh
ENTRYPOINT ["/run.sh"]
```

Below, the `run.sh` script is where the error occurs:

```bash
#!/bin/sh
nonexistent-command
```

Building and running the image as before using below commands demonstrate what is happening.

```bash
docker build -t my-entrypoint-image .
docker run my-entrypoint-image
```

The container will immediately exit because the `nonexistent-command` cannot be located and executed. This again indicates that, in a similar way to the `CMD` instruction example, the process specified by the `ENTRYPOINT` must be executing successfully in order to keep the container running. Debugging here would require checking the script logs or if no error codes or output are found, manually executing the commands to find the issue.

In summary, a Docker container starting but not running typically points to the primary process within that container having terminated. This can be due to multiple reasons ranging from incorrectly configured commands, application crashes, import errors, and issues within a custom entry point scripts. Debugging begins with thoroughly examining the container logs using the `docker logs` command, and systematically reviewing the commands specified in the `Dockerfile` within the `CMD` and `ENTRYPOINT` instructions. It is good practice to carefully inspect the application’s logs and ensure that the underlying service or command is operating correctly, and to ensure that you are aware of what the application’s intended behaviour is.

For individuals seeking more knowledge in containerization, several books offer thorough insights into the Docker ecosystem. Titles concerning “Docker in Practice,” or books on “Kubernetes” (which often assumes Docker knowledge) provide detailed explanation of these concepts. Docker’s official documentation also provides a comprehensive reference that details the lifecycle of a container, and best practices.
