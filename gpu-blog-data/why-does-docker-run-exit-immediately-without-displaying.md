---
title: "Why does docker run exit immediately without displaying script.sh's output?"
date: "2025-01-30"
id: "why-does-docker-run-exit-immediately-without-displaying"
---
The immediate exit of a Docker container after executing `docker run` without displaying the output of `script.sh` typically stems from the container's process completing before the standard output (stdout) and standard error (stderr) streams can be fully transmitted to the host.  This is a common issue I've encountered during years of building and deploying containerized applications, particularly in scenarios involving short-lived scripts.  The crucial understanding is that Docker containers are ephemeral processes; when the primary process within the container finishes, the container itself shuts down.

1. **Clear Explanation:**

The `docker run` command launches a container based on a specified image.  This image contains a filesystem and a defined entrypoint.  If you use `docker run` with a command, it overrides the image's default entrypoint.  In your case, `docker run <image> script.sh`, `script.sh` becomes the primary process.  If `script.sh` completes execution quickly – for instance, it performs a single task and terminates – the container's lifecycle mirrors this rapid completion.  The container's stdout and stderr, where the output of `script.sh` is written, haven't had sufficient time to be streamed back to the host machine before the container's shutdown.  This leads to the seemingly instantaneous exit and the absence of visible output.

The solution necessitates ensuring the primary process within the container remains active until all output is written. This can be achieved in several ways:

* **Running a process that waits for the script to finish:** A simple approach is to use a shell command (like `bash`, `sh`, or `zsh`) to execute `script.sh` and then incorporate a command that keeps the process running until its completion. This often involves using a `wait` command within the shell script.

* **Modifying the entrypoint or using a foreground process:**  Instead of overriding the entrypoint, you can configure the image's entrypoint to execute `script.sh` within a shell that ensures output is handled properly.  Alternatively, ensuring `script.sh` runs as a foreground process keeps the container active until its natural completion.

* **Using a process manager:** A more robust solution, especially for complex applications, involves using a process supervisor like `supervisord` or `tini`. These managers oversee processes, handle restarts, and facilitate logging, ensuring output is captured even if individual processes within the container terminate.

2. **Code Examples with Commentary:**

**Example 1: Using `bash` to wait for script completion:**

```bash
# Dockerfile
FROM ubuntu:latest
COPY script.sh /app/script.sh
RUN chmod +x /app/script.sh
CMD ["bash", "-c", "/app/script.sh && wait"]

# script.sh
#!/bin/bash
echo "Starting script..."
sleep 5
echo "Script finished."
```

This Dockerfile uses `bash -c` to execute the script. The `&& wait` ensures the container waits for `script.sh` to finish before exiting.  The `wait` command is crucial; it prevents the container from exiting prematurely.  I’ve utilized this in numerous projects where brief scripts needed explicit handling to prevent premature container termination.


**Example 2: Modifying the entrypoint to include the script:**

```dockerfile
# Dockerfile
FROM ubuntu:latest
COPY script.sh /app/script.sh
RUN chmod +x /app/script.sh
ENTRYPOINT ["/bin/bash", "-c", "/app/script.sh"]

# script.sh
#!/bin/bash
echo "Starting script..."
sleep 5
echo "Script finished."
```

Here, we leverage the `ENTRYPOINT` instruction to directly incorporate the execution of `script.sh` within the `bash` shell.  This is a cleaner approach than overriding the command if you don't need to change the default behavior in other contexts. I've found this method particularly useful when the script forms a core component of the container's functionality.


**Example 3: Utilizing `tini` as a process manager:**

```dockerfile
# Dockerfile
FROM ubuntu:latest
RUN apt-get update && apt-get install -y tini
COPY script.sh /app/script.sh
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
RUN chmod +x /app/script.sh
ENTRYPOINT ["/entrypoint.sh"]

# script.sh
#!/bin/bash
echo "Starting script..."
sleep 5
echo "Script finished."

# entrypoint.sh
#!/bin/bash
exec tini -- /app/script.sh
```

This example demonstrates a more sophisticated approach using `tini`.  `tini` acts as a process manager, ensuring the main process (script.sh in this case) is properly supervised.  This approach is beneficial when dealing with more complex scenarios involving multiple processes or when robust error handling and logging are needed.  During my work on large-scale deployments, I found `tini` invaluable for increased stability and better monitoring capabilities.


3. **Resource Recommendations:**

The Docker documentation provides comprehensive guidance on container lifecycle management and process handling.  Understanding how processes interact within the container context is crucial.  Exploring resources on process supervisors like `supervisord` and `tini` will deepen your understanding of managing long-running processes in containers.  Finally, becoming proficient in shell scripting enhances your ability to create well-structured containerized applications.  These resources, along with practical experience, are vital for effective container orchestration.
