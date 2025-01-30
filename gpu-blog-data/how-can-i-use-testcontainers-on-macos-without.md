---
title: "How can I use Testcontainers on macOS without Docker Desktop?"
date: "2025-01-30"
id: "how-can-i-use-testcontainers-on-macos-without"
---
Testcontainers, designed for integration testing, typically relies on a running Docker daemon. However, the reliance on Docker Desktop on macOS presents resource constraints and, for some, cost considerations. Therefore, alternatives using lightweight virtualization are imperative. I've extensively explored this space, encountering various challenges and nuances in establishing a viable Testcontainers environment on macOS without Docker Desktop. The key lies in leveraging Podman, specifically its machine functionality, which emulates Docker’s core functionality.

My initial encounter involved evaluating Colima, which is a valid option but introduced complexities in network configuration that ultimately proved less streamlined than Podman's approach for my specific use case involving multiple interconnected containers. Podman Machine provides a more direct drop-in replacement, and for projects already structured around Docker Compose, it offers a smoother transition. The primary conceptual difference to grasp is that Podman does not operate via a centralized daemon; rather, it creates virtual machines where containers run. These machines act as isolated, virtualized Docker environments.

Setting up a Podman Machine environment requires, primarily, installing Podman and its accompanying `podman-machine` tool, commonly available via Homebrew or their official website. I typically create a dedicated machine for Testcontainers to avoid interference with other container activities. The fundamental command to initialize a new machine is:

```bash
podman machine init testcontainers-machine
```

This command creates a virtual machine. Further customization, like assigning specific memory and CPU resources can be specified through command line parameters during initialization if needed, though the default configurations tend to be sufficient for most testing scenarios. Post-initialization, one needs to start this machine:

```bash
podman machine start testcontainers-machine
```

Crucially, the next step involves configuring your environment to point to this new machine. Podman relies on environment variables to direct its CLI. The command to retrieve these configuration variables and set them for the active session is:

```bash
eval $(podman machine env testcontainers-machine)
```

This command ensures that subsequent `podman` commands are executed against the defined virtual machine instead of a non-existent daemon. Failing to execute this step is a common source of confusion and errors. This setup also changes the default socket path for Docker. This altered path must also be accounted for by Testcontainers.

Now that a suitable Podman environment is running, consider the integration with Testcontainers itself. Testcontainers automatically detects a Docker environment at runtime, assuming it conforms to Docker’s API. Podman, being API-compatible, does not necessitate any major alteration in the Testcontainers code itself. However, it’s sometimes necessary to explicitly guide Testcontainers to locate the Docker socket exposed by the Podman machine. This can be achieved with environment variables or through direct configuration in your test setup.

Here's a first example demonstrating the core concept in a Java environment, a language where I’ve historically spent most of my time implementing integration tests with Testcontainers. Assume a basic JUnit 5 test:

```java
import org.junit.jupiter.api.Test;
import org.testcontainers.containers.PostgreSQLContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;

import java.sql.*;

import static org.junit.jupiter.api.Assertions.assertTrue;

@Testcontainers
public class DatabaseTest {

  @Container
  private PostgreSQLContainer<?> postgres = new PostgreSQLContainer<>("postgres:15-alpine");

  @Test
  void testDatabaseConnection() throws SQLException {
    String jdbcUrl = postgres.getJdbcUrl();
    String user = postgres.getUsername();
    String password = postgres.getPassword();

    try (Connection connection = DriverManager.getConnection(jdbcUrl, user, password)) {
      assertTrue(connection.isValid(5));
    }
  }
}
```

Here, `Testcontainers` automatically uses the configured Docker socket provided by the evaluated `podman machine env` command. The `PostgreSQLContainer` spins up inside the Podman virtual machine, allowing for database integration testing. No specific Testcontainers settings are needed because Podman inherently tries to comply with the expected socket structure. This highlights Podman’s primary utility as a drop-in replacement.

The second example showcases how to directly configure Testcontainers in cases where the automatic detection fails, or if multiple environments are in play. This requires manual specification of the Docker host:

```java
import org.junit.jupiter.api.Test;
import org.testcontainers.containers.GenericContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;
import org.testcontainers.utility.DockerImageName;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;

import static org.junit.jupiter.api.Assertions.assertEquals;

@Testcontainers
public class HttpTest {

  private static final String DOCKER_HOST = System.getenv("DOCKER_HOST");

  @Container
  private GenericContainer<?> webserver = new GenericContainer<>(DockerImageName.parse("nginx:latest"))
          .withExposedPorts(80);

    @Test
  void testHttpConnection() throws IOException, InterruptedException {
      HttpClient client = HttpClient.newHttpClient();
      String host = webserver.getHost();
      int port = webserver.getMappedPort(80);
      String url = String.format("http://%s:%d", host, port);
        HttpRequest request = HttpRequest.newBuilder().uri(URI.create(url)).build();
        HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());

        assertEquals(200, response.statusCode());

    }

}
```

In this setup, the `DOCKER_HOST` environment variable has been used to instruct Testcontainers to communicate with the correct Podman-managed virtual machine. The code itself remains standard, but the execution environment is adjusted to leverage Podman. Explicit configuration like this becomes crucial when utilizing CI/CD pipelines or different target environments. When using Testcontainers with multiple environments, an environment variable or a configuration class specifying the `DOCKER_HOST` may be required to ensure tests run against the proper docker host.

Finally, it's sometimes beneficial to utilize Docker Compose files for a more complex test setup, and Podman supports this directly. The following example demonstrates this concept:

```java
import org.junit.jupiter.api.Test;
import org.testcontainers.containers.DockerComposeContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;

import java.io.File;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertTrue;

@Testcontainers
public class ComposeTest {
    @Container
    private static DockerComposeContainer compose = new DockerComposeContainer(new File("src/test/resources/docker-compose.yml"))
            .withExposedService("web", 80);

    @Test
    void testDockerComposeUp() {
         Map<String, List<Integer>> exposedPorts = compose.getExposedPorts();
        assertTrue(exposedPorts.containsKey("web"));

    }

}

```
This code presumes the existence of `src/test/resources/docker-compose.yml` containing your service definitions. When executed, Testcontainers employs Podman to spin up the entire compose environment defined, thus offering a flexible solution for complex integration testing. Note: the environment variable `DOCKER_HOST` would also need to be configured to ensure compose starts in the intended virtual machine.

In summary, while Docker Desktop provides a convenient containerization experience, Podman Machine presents a robust alternative for Testcontainers on macOS. It requires an upfront setup, encompassing machine creation and environment configuration, but it subsequently facilitates seamless integration tests without reliance on Docker Desktop. For users seeking to escape the resource footprint or licensing implications of Docker Desktop, or for more complex CI/CD scenarios, a Podman based approach offers considerable merit.

For further information, I'd recommend exploring the official Podman documentation, which provides in-depth insights into Podman’s functionalities, and researching the Testcontainers website, particularly regarding its environment variable and configuration options. Additionally, community forums and tutorials dedicated to containerization best practices are valuable resources for solving more nuanced, edge-case scenarios that may arise.
