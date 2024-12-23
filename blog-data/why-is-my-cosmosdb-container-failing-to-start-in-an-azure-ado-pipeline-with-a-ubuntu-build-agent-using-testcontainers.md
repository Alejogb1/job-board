---
title: "Why is my CosmosDB container failing to start in an Azure ADO pipeline with a Ubuntu build agent using Testcontainers?"
date: "2024-12-23"
id: "why-is-my-cosmosdb-container-failing-to-start-in-an-azure-ado-pipeline-with-a-ubuntu-build-agent-using-testcontainers"
---

Alright,  Container initialization failures, especially in an Azure DevOps pipeline context using Testcontainers, can be a real head-scratcher at first glance. I've definitely seen my fair share of these, particularly when CosmosDB is in the mix. The fact that it's a Ubuntu build agent adds another layer we should carefully consider. Here’s a breakdown based on experiences I've had working on similar projects, which should provide some solid troubleshooting paths for your situation.

First off, let's dissect the core components. We have a CosmosDB container being orchestrated through Testcontainers within an Azure DevOps pipeline running on a Ubuntu build agent. The fundamental problem is not necessarily with the CosmosDB itself, but rather the environment in which it’s trying to run. The key things that I would investigate from a systems perspective are connectivity, resource constraints, and specific container-related settings.

One of the first issues I encountered in a similar project involved network configuration. Testcontainers creates a docker network to allow communication between the test container and the main test suite container. In my case, the default docker network driver wasn’t compatible with the virtualized environment used by the build agent. This manifested as the CosmosDB container timing out during startup, which often appeared as if it was never starting. The symptoms could be: the docker logs show a container starting but then not responding to readiness checks, or the tests would just fail with connection timeouts.

To address this, the solution was to switch the docker network driver to something more compatible, typically 'bridge'. You can achieve this in a few ways, but I’ve found the easiest in a pipeline environment is to set a docker configuration environment variable at the start of the pipeline job. This can be achieved in the pipeline yaml:

```yaml
jobs:
  - job: RunTests
    pool:
      vmImage: ubuntu-latest
    steps:
    - bash: |
        export DOCKER_DRIVER=bridge
      displayName: 'Set Docker Driver to bridge'
    - task: Maven@3
      inputs:
        mavenPomFile: 'pom.xml'
        mavenOptions: '-Xmx3072m'
        mavenGoals: 'test'
```

Notice that `export DOCKER_DRIVER=bridge` statement sets the required environment variable. This environment variable will then be picked up by testcontainers as it is configuring the Docker environment. This small change can resolve a large range of initialization problems. If the tests are written in a java environment, this can also be implemented via the Testcontainer configuration:

```java
import org.testcontainers.DockerClientFactory;
import org.testcontainers.containers.GenericContainer;
import org.testcontainers.containers.Network;
import org.testcontainers.utility.DockerImageName;

public class TestContainersConfig {

    static {
        System.setProperty("testcontainers.docker.network.driver", "bridge");
    }

    public static GenericContainer<?> cosmosDbContainer() {
        return new GenericContainer<>(DockerImageName.parse("mcr.microsoft.com/cosmosdb/linux/azure-cosmos-emulator:latest"))
            .withExposedPorts(8081, 10250, 10251, 10252, 10253, 10254, 10255)
            .withNetwork(Network.SHARED)
            .withEnv("AZURE_COSMOS_EMULATOR_PARTITION_COUNT", "1")
            .withStartupTimeout(java.time.Duration.ofSeconds(120));
    }
}
```

The key here is the `System.setProperty("testcontainers.docker.network.driver", "bridge");` line. This sets the system property, which affects how Testcontainers creates the Docker network.

A further issue that can contribute to the problem is resource limitations within the build agent environment. Docker containers, especially complex ones like CosmosDB, require sufficient memory and cpu. In some instances, the default resources allocated to a virtualized build agent might not be adequate. When I've run into this, the container would again, either fail silently, time out, or throw obscure memory related exceptions. I have found the following java code segment very useful in these cases:

```java
import org.testcontainers.DockerClientFactory;
import org.testcontainers.containers.GenericContainer;
import org.testcontainers.containers.Network;
import org.testcontainers.utility.DockerImageName;
import com.github.dockerjava.api.model.MemoryStats;

public class TestContainersConfig {

    static {
      System.setProperty("testcontainers.docker.network.driver", "bridge");

        com.github.dockerjava.api.DockerClient dockerClient = DockerClientFactory.instance().client();
        MemoryStats memoryStats = dockerClient.inspectContainerCmd(dockerClient.listContainersCmd().exec().get(0).getId()).exec().getContainer().getMemory();
        System.out.println("Available memory in agent container: " + memoryStats.getLimit());


        Runtime runtime = Runtime.getRuntime();
        System.out.println("Available JVM memory: " + runtime.maxMemory() / (1024*1024) + "MB");

    }

    public static GenericContainer<?> cosmosDbContainer() {
        return new GenericContainer<>(DockerImageName.parse("mcr.microsoft.com/cosmosdb/linux/azure-cosmos-emulator:latest"))
            .withExposedPorts(8081, 10250, 10251, 10252, 10253, 10254, 10255)
            .withNetwork(Network.SHARED)
            .withEnv("AZURE_COSMOS_EMULATOR_PARTITION_COUNT", "1")
            .withStartupTimeout(java.time.Duration.ofSeconds(120));
    }
}

```

This code retrieves the available memory stats from the docker container and the JVM. This can be very helpful in understanding if the issue is related to resource limitations.

To resolve this, you may need to explicitly define resource constraints in the Azure DevOps pipeline when you create the agents or ensure that you use a pool with sufficient resources. For example, you might choose to use a larger VM size if you're using Azure-hosted agents. Alternatively, you can consider adjusting the memory allocation to your docker container or java virtual machine with maven options such as `-Xmx` or configuring resources in the testcontainers API. In the snippet above, we allocate 3072Mb of memory with the maven command line argument `-Xmx3072m`.

Another factor, which, while less common, has still occurred when I was debugging container issues with CosmosDB, relates to container images. Using the latest version of a container image might be tempting, but they can sometimes introduce breaking changes. Pinning to a specific, known stable image tag of the CosmosDB emulator is crucial, especially when you are relying on a specific version of the CosmosDB API during development or testing. By specifying the image tag explicitly in your `DockerImageName`, you can ensure a deterministic test environment, avoiding inconsistencies from automatic image updates. If you have not specified the tag, docker will implicitly use the "latest" tag. This can result in intermittent errors depending on image registry state.

Furthermore, examine the Testcontainers logs very carefully. They can often provide detailed error messages that hint at the root cause. Look closely at any networking errors, port conflicts, or container timeouts during startup, especially in the startup logs of the cosmosdb container itself. If errors related to certificates are evident, then this is often related to incorrect docker network driver configurations. The logs are your best friend when it comes to troubleshooting such errors.

To recap, common issues with CosmosDB containers failing to start within a Testcontainers-based test suite in an Azure DevOps pipeline using a Ubuntu build agent include network driver incompatibilities, resource constraints and container image specific errors. By first ensuring that the docker driver is set correctly, allocating sufficient resources to the container, using a known version of the image and inspecting the container logs, the solution will become clear.

For further reading and better understanding of the specific technologies that I've touched on, I would highly recommend:

*   **"Docker in Action" by Jeff Nickoloff:** provides a solid foundation on how docker works and is great for getting a detailed overview of docker networking.
*   **The official Testcontainers documentation:** This is crucial for understanding the API and its different options for configuring containers. I use the testcontainer docs often.
*   **Microsoft's official documentation on CosmosDB:** Specifically the documentation for the emulator itself, as that will help clarify any errors that might be logged during startup.
*   **Azure DevOps documentation on build agents:** This will help in understand resource constraints and configuration options.

I've faced these problems more than once in my work, and by tackling each of the root cause points, you can make the container environment a much more stable and effective testing solution. Don’t hesitate to double check the basics, as they're often the key to solving these kinds of tricky scenarios.
