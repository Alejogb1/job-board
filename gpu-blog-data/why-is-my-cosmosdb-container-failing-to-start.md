---
title: "Why is my CosmosDB container failing to start in an Azure ADO pipeline with a Ubuntu build agent using Testcontainers?"
date: "2025-01-30"
id: "why-is-my-cosmosdb-container-failing-to-start"
---
Azure Cosmos DB containers failing to start within an Azure DevOps pipeline using a Ubuntu build agent and Testcontainers often stems from a subtle mismatch between the environment's capabilities and Testcontainers' demands. Specifically, the prevalent issue is the absence or misconfiguration of virtualization support within the build agent's environment. Testcontainers relies heavily on Docker, and Docker, in turn, frequently utilizes virtualization extensions (like those provided by Intel VT-x or AMD-V) for optimal performance and, in many cases, required functionality. This manifests as failures during the Testcontainers initialization phase, where the creation and management of the Docker containers, including the Cosmos DB emulator, goes awry.

Fundamentally, the Ubuntu-based build agents provided by Microsoft, especially those running in their hosted pools, frequently do not have nested virtualization enabled. This means that while the agent can run Docker, the underlying hypervisor preventing direct access to hardware virtualization resources can impede the smooth functioning of the Docker containers spawned by Testcontainers. Testcontainers often defaults to using a Docker bridge network configuration, which performs best with direct access to these virtualization extensions. Without this, the container startup can stall, fail due to network issues, or simply become non-responsive, causing test failures.

Let's examine scenarios, using relevant code examples:

**Example 1: Basic Cosmos DB Testcontainer Initialization**

Assume a standard JUnit 5 test setup in a Java project, relying on Testcontainers to manage a Cosmos DB emulator:

```java
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.testcontainers.containers.CosmosDBEmulatorContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;

import static org.junit.jupiter.api.Assertions.assertTrue;

@Testcontainers
public class CosmosDbTest {

    @Container
    private static final CosmosDBEmulatorContainer cosmosDbContainer = new CosmosDBEmulatorContainer();

    @BeforeAll
    static void setUp() {
        cosmosDbContainer.start();
    }


   @AfterAll
    static void tearDown() {
       cosmosDbContainer.stop();
   }


    @Test
    void containerIsRunning() {
        assertTrue(cosmosDbContainer.isRunning());
    }
}
```

This seemingly straightforward test can fail in the described pipeline scenario. When the `cosmosDbContainer.start()` method executes, it initiates the Docker container using configurations derived from the `CosmosDBEmulatorContainer` class. The Docker daemon running on the agent tries to manage this container without the benefit of nested virtualization. If it cannot successfully launch the container or establish network connectivity to it (which depends on virtual networking correctly being initialized by Docker), the startup will hang or throw exceptions. Typical errors observed in the pipeline logs would point to Docker commands timing out, unable to communicate with the container, or specific network interface creation failures within the Docker host environment.

**Example 2: Customized Container Configuration**

To provide some mitigation, one might try to customize the Cosmos DB emulator configuration to use a non-bridge network:

```java
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.testcontainers.containers.CosmosDBEmulatorContainer;
import org.testcontainers.containers.Network;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;

import static org.junit.jupiter.api.Assertions.assertTrue;

@Testcontainers
public class CosmosDbTestCustomNetwork {

    private static final Network customNetwork = Network.newNetwork();

    @Container
    private static final CosmosDBEmulatorContainer cosmosDbContainer = new CosmosDBEmulatorContainer()
                                                    .withNetwork(customNetwork)
                                                    .withNetworkAliases("cosmosdb");

   @BeforeAll
    static void setUp() {
        cosmosDbContainer.start();
    }

    @AfterAll
    static void tearDown() {
        cosmosDbContainer.stop();
        customNetwork.close();
    }

    @Test
    void containerIsRunning() {
        assertTrue(cosmosDbContainer.isRunning());
    }
}

```
In this case, a custom Docker network is created for the Cosmos DB container. This change *may* alleviate some issues in certain configurations by avoiding network conflicts, but the core problem, the absence of nested virtualization, is not fully resolved.  The error logs might exhibit different error messages related to network interface bindings or DNS resolution. It is essential to note that without virtualization support, network driver conflicts within the container or host system itself are probable. Thus, even with custom network configuration, the failure can occur during the container initialization sequence. The `start()` call will hang, and tests will timeout.

**Example 3: Specific Network Interface Binding**

Another approach to diagnose connectivity issues might be to explicitly bind the exposed port to a particular host interface within the Testcontainer configuration:

```java
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.testcontainers.containers.CosmosDBEmulatorContainer;
import org.testcontainers.containers.Network;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;

import static org.junit.jupiter.api.Assertions.assertTrue;

@Testcontainers
public class CosmosDbTestCustomPort {

   @Container
    private static final CosmosDBEmulatorContainer cosmosDbContainer = new CosmosDBEmulatorContainer()
                                                .withExposedPorts(8081);

   @BeforeAll
    static void setUp() {
        cosmosDbContainer.start();
    }

   @AfterAll
    static void tearDown() {
        cosmosDbContainer.stop();
    }

    @Test
    void containerIsRunning() {
        assertTrue(cosmosDbContainer.isRunning());
    }
}
```

This focuses on exposing a specific port, often for debugging purposes. It can, in some cases, highlight issues related to address conflicts. However, the root problem lies in the container’s ability to use Docker's networking mechanisms given the constraints of the Azure DevOps build agent environment. When virtualization is absent, container interfaces might not be properly set up on the host network.  While these techniques may sometimes improve specific networking scenarios, the underlying problem often remains. Thus, the `start()` method will still often timeout or raise exceptions.

**Recommendations for Resolution:**

To reliably use Testcontainers with Cosmos DB on Azure DevOps pipelines using Ubuntu agents, consider the following:

1. **Self-Hosted Agents:** The most robust solution is to use self-hosted agents where you have full control over the underlying hardware and the ability to enable nested virtualization. This allows Docker to manage containers more efficiently. Configure your self-hosted machine’s BIOS to enable VT-x/AMD-V (depending on processor manufacturer) and confirm virtualization is exposed to the host operating system. Also make sure that the machine you are using has adequate resources (CPU and RAM).

2. **Docker Configuration:** Investigate and adjust the Docker daemon configuration on the build agent. Explore alternative networking driver options and configurations available in Docker. However, be aware that altering the daemon configuration may have other downstream consequences in your deployment pipeline. The docker daemon might need additional configuration to work with the lack of virtualization.

3.  **Azure Container Instances:** For cases where self-hosted agents are not viable, explore using Azure Container Instances to run the test container workload. Then the pipeline can trigger the tests on these instances.  This decouples the Testcontainers implementation from the build agent environment.

4.  **Alternative Testing Strategies:** If running a true emulator instance is proving consistently problematic in the build pipeline, consider alternative strategies, such as using mock data or a lightweight in-memory database instead of relying on Testcontainers in the pipeline for integration testing. This approach should be limited to cases where a truly integrated test is not mandatory.

5.  **Agent Pool Configuration:** Examine agent pool configurations in Azure DevOps to ensure proper Docker versions are installed, and that no specific limitations or software conflicts exists in the default machine image. Evaluate the default available image and use a custom image where you have more control over configurations. This might require modifications in the Azure pipeline YAML.

6. **Testcontainers Debugging:** While Testcontainers offers decent debugging capabilities, make a detailed audit of all container initialization logs. Use the testcontainers logging capabilities to pin down specific points of failure. This may reveal more information about Docker configuration or resource contention issues.

By diligently investigating these potential causes and applying the suggested mitigations, you can typically diagnose and resolve the issues that manifest as Cosmos DB container startup failures within your Azure DevOps pipelines. The specific implementation will depend on your infrastructure, and will require iterative testing and monitoring.
