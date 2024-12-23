---
title: "How can I run Testcontainers with a dynamic port for Spring Data Elasticsearch?"
date: "2024-12-23"
id: "how-can-i-run-testcontainers-with-a-dynamic-port-for-spring-data-elasticsearch"
---

Alright,  I've bumped into this exact scenario a few times, particularly when setting up integration tests for applications heavily reliant on Elasticsearch. Dynamically assigning ports for Testcontainers when integrating with Spring Data Elasticsearch can initially seem like a minor hurdle, but it’s crucial for avoiding port conflicts, especially in continuous integration environments. Let me walk you through a practical approach, focusing on how I’ve handled it in past projects.

The core problem revolves around Testcontainers' inherent randomness when allocating ports, and how Spring Data Elasticsearch typically expects a fixed, preconfigured host and port combination. We need to bridge this gap so Spring's auto-configuration can connect successfully to our containerized Elasticsearch instance without requiring manual intervention each test run. There are several ways, each with its merits and quirks, but I’ve found the following methodology to be the most consistent and robust.

The essence is to use the `GenericContainer` capability of Testcontainers combined with a mechanism to fetch the randomly allocated port and then propagate this information to Spring's application context before the test runs. We’re not reinventing the wheel, but rather leveraging Testcontainers' strengths with a bit of glue code.

Here's how I usually structure it:

**1. Defining the Testcontainer and Fetching the Dynamic Port**

We begin by declaring our Elasticsearch container. I often extend `GenericContainer` for finer control and clarity. Inside this container setup, we make sure to extract the dynamically allocated port *after* the container has started. This is critical as the port is only known after the container is running. We'll need this port later when configuring our Spring Data Elasticsearch client.

```java
import org.testcontainers.containers.GenericContainer;
import org.testcontainers.utility.DockerImageName;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;
import java.util.Map;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;

@Testcontainers
@SpringBootTest
public class ElasticsearchIntegrationTest {

    private static final String ELASTICSEARCH_DOCKER_IMAGE = "docker.elastic.co/elasticsearch/elasticsearch:7.17.9";

    @Container
    private static final GenericContainer<?> elasticsearchContainer =
            new GenericContainer<>(DockerImageName.parse(ELASTICSEARCH_DOCKER_IMAGE))
                    .withExposedPorts(9200)
                    .withEnv(Map.of("discovery.type", "single-node"))
                    ;

    private static int elasticsearchPort;

    @BeforeAll
    static void setUp() {
        elasticsearchPort = elasticsearchContainer.getMappedPort(9200);
        System.setProperty("spring.elasticsearch.rest.uris", "http://localhost:" + elasticsearchPort);
    }

    @Test
    void contextLoads() {
        // Test logic goes here
        System.out.println("Test is running with Elasticsearch on port: " + elasticsearchPort);
    }

}

```

In this snippet:

*   We're using `@Testcontainers` to manage the lifecycle of the Docker container automatically.
*   We define `elasticsearchContainer` as a `GenericContainer`, using a specific version of the Elasticsearch docker image. The `"single-node"` setting is crucial for tests; it avoids issues related to cluster discovery.
*   We expose port 9200 (the default Elasticsearch HTTP port).
*   The `@BeforeAll` method is where the magic happens. After the container starts, we call `getMappedPort(9200)` to fetch the randomly allocated host port. This dynamic port is then used to set the `spring.elasticsearch.rest.uris` system property before the application context is initialized. This dynamic configuration via system properties is essential.

**2. Configuring Spring Data Elasticsearch Using System Properties**

The key here is to not hardcode port settings into our application configuration files. Instead, we leverage Spring's ability to resolve properties through system variables. In our setup, we set `spring.elasticsearch.rest.uris` to `http://localhost:${elasticsearch.port}`. We don't actually define 'elasticsearch.port' in properties or configuration files. Instead we use the system property as set in the previous section.

By using System properties, we dynamically configure the elasticsearch client based on the runtime configuration of the container. Spring Data Elasticsearch will then use this property to form a connection to the container. No manual configuration of the test environment is required. This approach ensures our tests are both isolated and self-contained, making them easier to execute in varied CI environments.

**3. Testing the Setup**

Within our test class, we can now rely on Spring's auto-configured Elasticsearch client to communicate with the container. Here’s a concise example of how this might look:

```java
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.elasticsearch.core.ElasticsearchOperations;
import static org.assertj.core.api.Assertions.assertThat;


public class ElasticsearchIntegrationTest {
    //Previous Code included Here.
    @Autowired
    private ElasticsearchOperations elasticsearchOperations;

   @Test
    void canConnectToElasticsearch() {
        assertThat(elasticsearchOperations.indexExists("test-index")).isFalse(); //Asserting connectivity by checking for a test index
   }
}
```

In the test case, I autowire `ElasticsearchOperations`. If the setup is correct, I can now use it to interact with the running Elasticsearch instance. The assertion that the "test-index" does not exist serves as a simple verification that our Spring context can connect to our dynamic Testcontainer.

**Additional Considerations and Best Practices:**

*   **Image Versioning:** Always pin your Docker image versions (`docker.elastic.co/elasticsearch/elasticsearch:7.17.9` rather than simply `elasticsearch:latest`). This prevents unexpected behavior due to image updates.
*   **Startup Time:** Elasticsearch can sometimes take a little while to initialize inside a container. You might want to add health checks to the container definition or use Testcontainers' wait strategies to ensure the service is available before attempting to connect. Refer to the Testcontainers documentation for methods such as `Wait.forHttp`.
*   **Cleanup:** Testcontainers automatically cleans up containers after tests, but it's good practice to have explicit shutdown handling in integration tests with more complex setups.
*   **Logging:** Enable verbose logging for both Testcontainers and Spring Data Elasticsearch if you encounter issues. This can provide valuable insights into connection problems.
*   **Resource Management:** Be mindful of resource usage. For large-scale integration tests, consider adjusting Docker's memory and CPU allocations.

**Recommended Resources:**

For deeper understanding of the underlying concepts and related technologies, I would recommend these resources:

*   **Testcontainers Documentation:** The official Testcontainers documentation (available online) is the authoritative source for everything related to container lifecycle management within tests.
*   **"Spring Data Elasticsearch" Reference Documentation:** The official documentation for Spring Data Elasticsearch details all the configurations and customizations available for the Spring framework.
*   **"Docker in Action" by Jeff Nickoloff and Ian Miell:** A practical and in-depth guide on Docker, which is essential for understanding Testcontainers.
*   **"Effective Java" by Joshua Bloch:** While not specific to Testcontainers, this book helps refine your general coding practices, which is beneficial when structuring tests and ensuring maintainability.

In summary, the technique of dynamically assigning ports and injecting configuration via system properties, combined with Testcontainers, provides a reliable and reproducible testing environment. The focus should always be on automating as much as possible, removing manual setup, and ensuring your tests are as close as possible to your production configurations. I hope this approach will prove as useful to you as it has been for me in the past.
