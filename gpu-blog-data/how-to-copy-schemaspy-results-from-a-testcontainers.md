---
title: "How to copy SchemaSpy results from a Testcontainers container?"
date: "2025-01-30"
id: "how-to-copy-schemaspy-results-from-a-testcontainers"
---
SchemaSpy's output, by default, resides within the container's filesystem. Accessing this data requires understanding how Testcontainers manages containers and their ephemeral nature.  My experience working with integration tests and database migrations has frequently necessitated extracting SchemaSpy reports generated within isolated containers.  This necessitates a multi-step process leveraging container functionalities and potentially external tools.

**1. Understanding the Challenge**

The core difficulty lies in the transient nature of Testcontainers.  Containers are spawned, utilized, and then destroyed after the test suite completes.  Simply attempting to access the filesystem directly after container shutdown will invariably fail. The solution, therefore, involves retrieving the data *before* the container is removed.  This can be achieved through several methods, each with its own trade-offs concerning complexity and resource utilization.

**2.  Solution Strategies**

The most robust approach involves copying the generated files directly from the container using Testcontainers' built-in capabilities, or leveraging `docker cp` from the command line.  Alternatively, for more complex scenarios or larger datasets, mounting a host directory as a volume within the container allows for shared access. The final, less efficient approach, is to stream the report content directly from the container.

**3. Code Examples and Commentary**

**Example 1: Using Testcontainers' `copyFileFromContainer` (Recommended)**

This method directly utilizes Testcontainers' functionality for efficient and contained data retrieval.  It avoids external dependencies and keeps the solution within the testing framework.

```java
import org.testcontainers.containers.GenericContainer;
import org.testcontainers.containers.wait.strategy.Wait;
import org.testcontainers.utility.MountableFile;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class SchemaSpyExtraction {

    public static void main(String[] args) throws IOException, InterruptedException {

        GenericContainer<?> container = new GenericContainer<>("mysql:8")
                .withEnv("MYSQL_ROOT_PASSWORD", "password")
                .withExposedPorts(3306)
                .waitingFor(Wait.forLogMessage(".*mysqld: ready for connections.*\\s", 1));

        container.start();

        // ... (Your database setup code, including SchemaSpy execution within the container) ...

        // Assuming SchemaSpy outputs to /tmp/schemaspy.html within the container
        Path hostPath = Paths.get("target/schemaspy.html");
        String containerPath = "/tmp/schemaspy.html";

        container.copyFileFromContainer(containerPath, hostPath.toString());

        System.out.println("SchemaSpy report copied to: " + hostPath.toAbsolutePath());

        container.stop();
    }
}
```

This example leverages `copyFileFromContainer`.  Critical aspects here are ensuring the correct path within the container (`containerPath`) and specifying a local destination (`hostPath`) for the copied file.  Error handling (e.g., using a `try-catch` block) should be included for robust production use.  The database setup and SchemaSpy execution are represented by "... (Your database setup code, including SchemaSpy execution within the container) ...". This would involve using JDBC to interact with your database, running your schema generation scripts, and invoking the schemaspy command inside the container using `ExecInContainer`.

**Example 2: Utilizing `docker cp` (Requires Docker CLI)**

This approach uses the Docker command-line interface directly. It offers flexibility but necessitates a direct interaction with the Docker daemon.

```java
import org.testcontainers.containers.GenericContainer;
import org.testcontainers.containers.wait.strategy.Wait;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Path;
import java.nio.file.Paths;

public class SchemaSpyExtractionDockerCp {

    public static void main(String[] args) throws IOException, InterruptedException {
        GenericContainer<?> container = new GenericContainer<>("mysql:8")
                .withEnv("MYSQL_ROOT_PASSWORD", "password")
                .withExposedPorts(3306)
                .waitingFor(Wait.forLogMessage(".*mysqld: ready for connections.*\\s", 1));

        container.start();

        // ... (Your database setup code, including SchemaSpy execution within the container) ...

        String containerId = container.getContainerId();
        String containerPath = "/tmp/schemaspy.html";
        Path hostPath = Paths.get("target/schemaspy.html");

        ProcessBuilder processBuilder = new ProcessBuilder("docker", "cp", containerId + ":" + containerPath, hostPath.toString());
        Process process = processBuilder.start();

        BufferedReader reader = new BufferedReader(new InputStreamReader(process.getErrorStream()));
        String line;
        while ((line = reader.readLine()) != null) {
            System.err.println(line);
        }

        int exitCode = process.waitFor();
        if (exitCode == 0) {
            System.out.println("SchemaSpy report copied to: " + hostPath.toAbsolutePath());
        } else {
            System.err.println("Error copying SchemaSpy report. Exit code: " + exitCode);
        }

        container.stop();
    }
}
```

This example directly invokes the `docker cp` command.  Error handling is implemented by checking the exit code of the process. The crucial aspect is obtaining the container ID (`container.getContainerId()`) which is used to specify the source for the copy operation.

**Example 3: Volume Mounting (For Large Datasets)**

For very large SchemaSpy reports or if multiple files need to be copied, mounting a volume provides the most efficient solution.


```java
import org.testcontainers.containers.GenericContainer;
import org.testcontainers.containers.wait.strategy.Wait;
import org.testcontainers.utility.MountableFile;

import java.nio.file.Path;
import java.nio.file.Paths;

public class SchemaSpyExtractionVolume {

    public static void main(String[] args) throws InterruptedException {
        Path hostPath = Paths.get("target/schemaspy");
        MountableFile mountableFile = MountableFile.forHostPath(hostPath);


        GenericContainer<?> container = new GenericContainer<>("mysql:8")
                .withEnv("MYSQL_ROOT_PASSWORD", "password")
                .withExposedPorts(3306)
                .withCopyFileToContainer(mountableFile, "/tmp/schemaspy")
                .waitingFor(Wait.forLogMessage(".*mysqld: ready for connections.*\\s", 1));

        container.start();

        // ... (Your database setup code, including SchemaSpy execution within the container, writing to /tmp/schemaspy) ...

        container.stop();
    }
}
```

In this approach,  we use `withCopyFileToContainer` to mount a host directory to a location inside the container. Any file written to `/tmp/schemaspy` within the container will be directly accessible in the `target/schemaspy` directory on the host machine.  This approach avoids the copy operation entirely after the container has finished running.


**4. Resource Recommendations**

*   **Testcontainers documentation:**  Thoroughly reviewing the Testcontainers documentation is crucial for understanding advanced features and best practices.  Pay close attention to the sections on container management and file operations.
*   **Docker documentation:**  Familiarize yourself with the `docker cp` command and Docker's volume management capabilities. This understanding is vital for troubleshooting and implementing alternative solutions.
*   **SchemaSpy documentation:**  Understanding SchemaSpy's configuration options, including output directory specifications, is essential for accurate file path identification within the container.  Pay close attention to command-line options to configure output directory.


By carefully selecting and implementing one of these strategies, depending on your environment and project needs, you can effectively extract SchemaSpy results from Testcontainers without disrupting your testing workflow.  Remember to consider error handling and adjust file paths based on your specific configuration.  Thoroughly testing these solutions in a controlled environment before integrating into larger test suites is highly recommended.
