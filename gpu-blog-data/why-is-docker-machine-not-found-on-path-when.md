---
title: "Why is docker-machine not found on PATH when using Testcontainers?"
date: "2025-01-30"
id: "why-is-docker-machine-not-found-on-path-when"
---
Testcontainers, while facilitating seamless integration testing with containerized applications, sometimes presents a hurdle related to `docker-machine` not being discoverable. This isn't an intrinsic issue with Testcontainers, but rather a consequence of its reliance on Docker environment variables and the potential mismatch with the user's local setup, specifically when `docker-machine` was previously employed for Docker management.

The core of the problem stems from how Testcontainers locates the Docker daemon. It attempts to interact with the Docker daemon using a combination of environment variables, such as `DOCKER_HOST`, `DOCKER_TLS_VERIFY`, and `DOCKER_CERT_PATH`, and a default socket if none of these are set. When a `docker-machine` environment is in play, those environment variables are precisely what `docker-machine env` would set to establish a connection to the specific virtual machine housing the Docker daemon. However, the problem arises when these variables are not active *in the context where Testcontainers is executing*. Consider, for example, when tests are being run directly from an IDE or within a CI/CD pipeline; these contexts might not have the `docker-machine` environment activated. This results in Testcontainers attempting to connect via the defaults or perhaps to a different, unintended Docker environment.

The error "docker-machine not found on PATH" is a misleading symptom. The issue is not that Testcontainers requires the `docker-machine` executable to be on the path, per se. It's that Testcontainers expects a functional Docker daemon connection configured via environment variables and cannot reach it when those variables are not set to point to the right `docker-machine` environment, or when they are absent and the default socket configuration is not viable. Testcontainers essentially expects these to be configured beforehand, relying on the user or their environment to do the setup.

To elaborate, before Docker Desktop became a popular, streamlined option, `docker-machine` was common for managing Docker on macOS and older Windows versions. Each `docker-machine` instance would run a separate Docker daemon inside a virtual machine. This mechanism required activation for the command-line `docker` executable to communicate with the correct daemon. This activation is achieved via sourcing the `docker-machine env <machine-name>` shell command and setting environment variables accordingly. These variables enable the `docker` client, and Testcontainers in turn, to talk to the correct daemon. If the variables are missing, Testcontainers falls back to defaults, resulting in the error and inability to connect.

The solution often involves one of two options: either ensuring the `docker-machine` environment is active where the tests execute, or directly specifying a valid Docker socket to Testcontainers (if you're bypassing Docker Machine entirely, for instance if Docker Desktop is being used).

Here are code examples demonstrating these concepts:

**Example 1: Activating `docker-machine` Environment Within Test Run (Not Recommended)**

While functional, sourcing within the test context is fragile. It introduces unnecessary system dependencies and can fail due to unforeseen environmental issues. Here I'll demonstrate a Java approach using `ProcessBuilder` which has the same principles in other languages.

```java
import org.junit.jupiter.api.Test;
import org.testcontainers.containers.GenericContainer;
import org.testcontainers.utility.DockerImageName;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

public class DockerMachineTest {
   @Test
    void testWithDockerMachine() throws IOException, InterruptedException {
      // Assume 'default' is your docker-machine name
      String machineName = "default";
      List<String> envCommand = new ArrayList<>(List.of("/bin/bash", "-c", "docker-machine env "+ machineName));

       ProcessBuilder processBuilder = new ProcessBuilder(envCommand);
       Process process = processBuilder.start();
       process.waitFor();

       // Read output
       BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
       String line;
       List<String> dockerEnvVars = new ArrayList<>();
        while ((line = reader.readLine()) != null) {
            if(line.startsWith("export")) {
                dockerEnvVars.add(line);
            }
        }

       //Set variables in test run
        dockerEnvVars.forEach(envVar -> {
            String[] parts = envVar.replace("export ", "").split("=");
            System.setProperty(parts[0], parts[1].replace("\"", ""));
        });


      try (GenericContainer<?> container = new GenericContainer<>(DockerImageName.parse("alpine:latest"))) {
          container.start();
         //Container works because of docker-machine settings from shell
          container.execInContainer("sh", "-c", "echo Hello from Docker");
          container.stop();

       }

    }
}
```

The above code first executes the `docker-machine env` command and extracts the necessary `export` lines from its output and then sets these as system properties which are then used by the Testcontainers library.

**Example 2: Utilizing `DOCKER_HOST` Directly**

This method works when one is actively using a `docker-machine` environment or knows the socket address to communicate with the daemon and provides a more robust solution if not dynamically changing docker environments. This method is more resistant to accidental environment changes.

```java
import org.junit.jupiter.api.Test;
import org.testcontainers.containers.GenericContainer;
import org.testcontainers.utility.DockerImageName;
import org.testcontainers.DockerClientFactory;

public class DockerHostTest {
    @Test
    void testWithDockerHost()  {
      //Set DOCKER_HOST
        System.setProperty("DOCKER_HOST", "tcp://192.168.99.100:2376");
       try (GenericContainer<?> container = new GenericContainer<>(DockerImageName.parse("alpine:latest"))) {
           //Container works because of manually set docker host
           container.start();
          container.execInContainer("sh", "-c", "echo Hello from Docker");
          container.stop();

       }

    }
}

```

In this case, I've directly specified `DOCKER_HOST`. You'd obtain the IP via `docker-machine ip <machine-name>` or the output of `docker-machine env <machine-name>`. This directly provides the address of the daemon instead of relying on the test environment to setup the correct variables, but does require knowing these values ahead of time.

**Example 3: Explicitly specifying the socket.**

If you're not using docker machine, but Docker Desktop, or have a remote daemon, it is often better to explicitly specify the socket location for clarity.

```java
import org.junit.jupiter.api.Test;
import org.testcontainers.containers.GenericContainer;
import org.testcontainers.utility.DockerImageName;

public class DockerSocketTest {

    @Test
    void testWithDockerSocket()  {
      //Set socket location. This could also be a tcp socket in case of a remote daemon
        System.setProperty("TESTCONTAINERS_DOCKER_SOCKET_OVERRIDE", "/var/run/docker.sock");
        try (GenericContainer<?> container = new GenericContainer<>(DockerImageName.parse("alpine:latest"))) {
            //Container works because of explicitly set socket
          container.start();
          container.execInContainer("sh", "-c", "echo Hello from Docker");
          container.stop();
       }

    }

}

```

Here, the Docker socket path is set using the `TESTCONTAINERS_DOCKER_SOCKET_OVERRIDE` property. This ensures that Testcontainers connects directly to the specified socket and bypasses any environment variable-related misconfigurations.

**Resource Recommendations:**

*   **Docker Documentation:** The official Docker documentation thoroughly explains Docker daemon architecture, client communication, and usage of environment variables. Specifically, pay attention to sections concerning `docker-machine` if you are still utilizing it.
*   **Testcontainers Documentation:** The Testcontainers documentation provides detailed information on its connection process, supported environment variables, and advanced configuration options. Understanding their internal mechanics will help in debugging related errors.
*   **Stack Overflow and Similar Forums:** Examining questions and answers regarding Testcontainers and Docker connection issues provides real-world scenarios and solutions from fellow developers. A wide range of cases can broaden the understanding of the issue's scope.

In summary, the `docker-machine not found on PATH` error with Testcontainers is generally not about the binary's path, but rather, stems from a broken or missing connection to the Docker daemon, typically due to misconfigured or absent environment variables from `docker-machine` or an incorrect socket path. Either explicitly setting the connection parameters or using a modern Docker setup, avoiding docker-machine where possible, solves this problem. Proper environment variable management, coupled with a grasp of Testcontainers' inner workings, are key to a smooth test suite integration.
