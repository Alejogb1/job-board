---
title: "Why are Spring Boot container tests failing with testcontainers/ryuk on Docker Engine 20.10?"
date: "2025-01-30"
id: "why-are-spring-boot-container-tests-failing-with"
---
My experience indicates that Spring Boot container tests utilizing Testcontainers alongside Ryuk often fail on Docker Engine 20.10 due to a subtle interaction between Ryuk's container cleanup mechanism and a change in Docker's container networking behavior introduced in that specific version. Ryuk, the resource reaping container, relies on the Docker API to identify and remove orphaned containers created during test execution. Prior to Docker 20.10, the network configurations and cleanup of bridge networks were typically consistent and predictable. However, Docker 20.10 implemented subtle changes in how bridge networks are managed, specifically around container cleanup lifecycle events that Ryuk implicitly depends upon. This often manifests as Ryuk failing to identify and consequently clean up the test containers, leading to resource exhaustion and test failures.

The core issue arises because of the way Docker handles network deletion in relation to associated containers. In earlier versions, when a Docker network was marked for deletion, the containers associated with it were generally immediately considered "dead" from Ryuk's perspective. Ryuk, upon observing this "death," would initiate the container removal. However, in Docker 20.10, there seems to be a slight delay or change in how the Docker API reports a container's state in relation to network cleanup, specifically involving custom bridge networks that Testcontainers often uses. This results in Ryuk occasionally checking for orphaned containers *before* Docker fully updates the container's network-related state, leading to missed cleanup cycles. Ryuk then considers those containers as still actively running, or at least not yet eligible for reaping. This can result in accumulating containers, which eventually consumes resources or conflicts with future test container creation, causing tests to either timeout, fail to start, or run into unexpected network port binding issues.

To clarify, the problem is not directly a bug within Testcontainers itself, but rather an unfortunate timing issue exacerbated by changes in the Docker API's behavior within Docker 20.10. Specifically, containers remain "alive" for longer in Ryuk’s view after the bridge network associated with the tests has been marked for removal by Docker itself. The impact is that Ryuk does not promptly clean them, which will affect both resource usage and test execution stability. This is particularly noticeable in integration test suites that instantiate numerous short-lived containers.

Here are a few code examples illustrating this issue, alongside explanations of the problem areas:

**Example 1: Basic Spring Boot Test with Testcontainers**

```java
import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;
import org.testcontainers.containers.PostgreSQLContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;

import static org.junit.jupiter.api.Assertions.assertTrue;

@SpringBootTest
@Testcontainers
public class SimplePostgresTest {

    @Container
    private static final PostgreSQLContainer<?> postgres = new PostgreSQLContainer<>("postgres:13.3");


    @Test
    void testContainerStarts() {
        assertTrue(postgres.isRunning());
    }

}
```

*   **Commentary:** This is a typical simple setup. Here, the `PostgreSQLContainer` is started using the `@Container` annotation, managed by Testcontainers. The issue with Docker 20.10 surfaces when numerous tests using this or similar setups are run in sequence. Ryuk struggles to remove containers after the test, leading to increased resource consumption and, eventually, failed test runs. There's nothing intrinsically wrong with this code; the problem is purely in how the underlying infrastructure of Docker and Ryuk interact. The individual test will likely pass as containers will be created and started as expected.

**Example 2: Spring Boot Application Context Test**

```java
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.jdbc.core.JdbcTemplate;
import org.testcontainers.containers.PostgreSQLContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;
import javax.sql.DataSource;
import static org.junit.jupiter.api.Assertions.assertNotNull;


@SpringBootTest
@Testcontainers
public class DataSourceIntegrationTest {

    @Container
    private static final PostgreSQLContainer<?> postgres = new PostgreSQLContainer<>("postgres:13.3");

    @Autowired
    private DataSource dataSource;


    @Test
    void testDataSourceInjection() {
        assertNotNull(dataSource);
    }


}
```

*   **Commentary:** This example is a slight variation, focusing on Spring’s `DataSource` injection based on an external PostgreSQL container. Again, the test itself works correctly. The resource buildup and eventual failure manifests during larger test suites, or under specific CI/CD pipeline load. The core issue with Ryuk misidentifying containers as active persists. The network associated with the `PostgreSQLContainer` is a temporary bridge network. If the cleanup is missed, future tests might attempt to bind to the same ports, resulting in port binding exceptions or timeouts.

**Example 3: Multiple Containers in a Test Suite**

```java
import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;
import org.testcontainers.containers.GenericContainer;
import org.testcontainers.containers.PostgreSQLContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;

import static org.junit.jupiter.api.Assertions.assertTrue;

@SpringBootTest
@Testcontainers
public class MultipleContainerTest {

    @Container
    private static final PostgreSQLContainer<?> postgres = new PostgreSQLContainer<>("postgres:13.3");
    @Container
    private static final GenericContainer<?> redis = new GenericContainer<>("redis:6.2").withExposedPorts(6379);

    @Test
    void testMultipleContainers() {
        assertTrue(postgres.isRunning());
        assertTrue(redis.isRunning());
    }

}
```

*   **Commentary:** This example shows two containers, which significantly exacerbates the Ryuk problem. The increased container density on the Docker host magnifies the likelihood of timing-related cleanup failures in Docker 20.10. With multiple containers starting and stopping per test, the window where Ryuk misses containers widens considerably. In practical scenarios, this leads to flaky builds, where tests will succeed sporadically if Docker and the host OS resource management are temporarily favorable, but will eventually consistently fail due to insufficient resources.

Mitigation for this specific problem involves careful management of the Testcontainers lifecycle and potentially using system properties that control container management settings. Here are some resource recommendations:

1.  **Testcontainers Documentation:** Thoroughly review the Testcontainers documentation (available on the official website), particularly the sections related to Ryuk and resource management. It contains crucial details about system properties that influence container cleanup.
2.  **Docker Release Notes:** Consult the Docker Engine release notes for Docker 20.10, looking specifically at the sections regarding container networking and network deletion behavior. This is necessary to understand the root cause of the issue.
3.  **GitHub Issues for Testcontainers:** Review the GitHub issues for the Testcontainers project. There may be existing discussions regarding this specific problem with potential workaround and mitigation strategies being explored by the maintainers and community.
4.  **General Docker Knowledge:** A deeper understanding of Docker's core concepts related to networks, container lifecycle and the API will help diagnose problems that may be specific to your environment. Reading Docker’s documentation can be useful.

My conclusion is that while the code examples provided function as expected under normal conditions, the underlying issue with Testcontainers, Ryuk, and Docker 20.10 interactions introduces a systemic problem that only becomes apparent during large or long running testing cycles or high usage situations. Careful review of the suggested resources is necessary to fully comprehend and ultimately mitigate the observed behavior. Simply relying on the functionality of the Testcontainers library will not suffice. The critical area that requires attention is the interplay between Ryuk, the Docker API’s responses regarding container and network states, and how that interaction has changed since earlier Docker versions. Careful handling of configuration options for Testcontainers, and ensuring the resources consumed by docker are limited can reduce the impact.
