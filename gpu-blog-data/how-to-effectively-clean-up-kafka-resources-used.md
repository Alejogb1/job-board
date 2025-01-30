---
title: "How to effectively clean up Kafka resources used by TestContainers?"
date: "2025-01-30"
id: "how-to-effectively-clean-up-kafka-resources-used"
---
Kafka, when deployed within Testcontainers for integration testing, commonly exhibits resource persistence issues if not handled correctly post-test execution. These issues typically manifest as lingering container instances, orphaned topics, and potentially clogged ZooKeeper nodes, ultimately impacting test repeatability and development environment hygiene. My experience, having orchestrated several Kafka-centric microservice test suites using Testcontainers, has underscored the necessity of employing explicit cleanup mechanisms. The default Testcontainers behavior, while convenient for development, often falls short of ensuring a consistently clean slate for subsequent tests.

The primary challenge stems from the fact that Testcontainers manages container lifecycles based on the scope of its declared containers. If a Kafka container, along with its associated ZooKeeper dependency, are declared as `@ClassRule` or `@BeforeClass`-scoped fields, they persist across all tests in the class. While resource pooling improves test performance, it also mandates careful disposal when the tests are completed. Failure to perform adequate cleanup can lead to port conflicts, incorrect state for subsequent tests, and overall confusion in the test environment. In my earlier projects, I often encountered situations where seemingly unrelated tests started to fail due to lingering state from prior test runs.

Effective cleanup centers around a two-pronged strategy: stopping and removing the Kafka and ZooKeeper containers, and deleting any topics created during the test. The former is straightforward using the `stop()` and `close()` methods provided by Testcontainers for each container instance. The latter involves leveraging the Kafka Admin Client to programmatically purge topics. We must perform the container cleanup as the last operation, typically in the `@AfterClass` annotated method or similar. Without the guarantee of a fresh environment, test result reliability and reproducibility severely degrade, making debugging significantly harder.

Here are three code examples illustrating different approaches to cleanup:

**Example 1: Basic Container Cleanup**

This example demonstrates the most fundamental aspect: ensuring the containers stop and are removed. The code uses JUnit 5 but the principle remains the same for other test frameworks.

```java
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.testcontainers.containers.KafkaContainer;
import org.testcontainers.utility.DockerImageName;

public class KafkaBasicCleanupTest {

    private static KafkaContainer kafka;

    @BeforeAll
    static void setup() {
        kafka = new KafkaContainer(DockerImageName.parse("confluentinc/cp-kafka:7.5.0"));
        kafka.start();
    }

    @Test
    void dummyTest() {
        // Test using the Kafka container
        assert(true); // some assert to keep the test running
    }

    @AfterAll
    static void tearDown() {
      if (kafka != null) {
        kafka.stop();
        kafka.close();
      }

    }
}

```

*Commentary:* This example creates a `KafkaContainer` instance, starts it before all tests, and then stops and closes it after all tests finish. While this effectively removes the container, it does not address the challenge of deleting topics. The `if` check ensures that even if setup fails the cleanup will not throw errors. Using `stop()` prevents the containers from using system resources, while using `close()` will ensure the container will not be used in future test executions. This is crucial for isolation. The use of a specific docker image version is also recommended for reproducibility.

**Example 2: Programmatic Topic Deletion**

This example extends the previous one by adding topic deletion using the Kafka Admin Client. This assumes the test creates topics that should be removed.

```java
import org.apache.kafka.clients.admin.AdminClient;
import org.apache.kafka.clients.admin.AdminClientConfig;
import org.apache.kafka.clients.admin.DeleteTopicsResult;
import org.apache.kafka.clients.admin.NewTopic;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.testcontainers.containers.KafkaContainer;
import org.testcontainers.utility.DockerImageName;

import java.time.Duration;
import java.util.Collections;
import java.util.Properties;
import java.util.concurrent.ExecutionException;

public class KafkaTopicCleanupTest {

    private static KafkaContainer kafka;
    private static AdminClient kafkaAdmin;
    private static final String TEST_TOPIC = "test-topic";

    @BeforeAll
    static void setup() throws ExecutionException, InterruptedException {
        kafka = new KafkaContainer(DockerImageName.parse("confluentinc/cp-kafka:7.5.0"));
        kafka.start();

        Properties properties = new Properties();
        properties.put(AdminClientConfig.BOOTSTRAP_SERVERS_CONFIG, kafka.getBootstrapServers());
        kafkaAdmin = AdminClient.create(properties);

        NewTopic newTopic = new NewTopic(TEST_TOPIC, 1, (short) 1);
        kafkaAdmin.createTopics(Collections.singletonList(newTopic)).all().get();
    }

    @Test
    void testWithTopic() {
        // Test logic that uses the created topic
        assert(true);
    }


    @AfterAll
    static void tearDown() throws ExecutionException, InterruptedException {
        if (kafkaAdmin != null) {
            DeleteTopicsResult deleteTopicsResult = kafkaAdmin.deleteTopics(Collections.singletonList(TEST_TOPIC));
           deleteTopicsResult.all().get();

            kafkaAdmin.close(Duration.ofSeconds(5));

        }

        if(kafka!= null) {
            kafka.stop();
            kafka.close();
        }
    }
}
```

*Commentary:*  Here, the `AdminClient` is used to create a test topic before the test and then to delete it afterward.  The `deleteTopics` method returns a `DeleteTopicsResult` object, on which we need to invoke `all().get()` to ensure the deletion completes synchronously before proceeding to close the client or container. The `kafkaAdmin.close` method accepts a timeout, which ensures that cleanup does not indefinitely wait. Properly cleaning up resources is crucial for avoiding conflicts during continuous test execution. Failure to close the admin client will create memory leaks.

**Example 3:  Cleanup with Multiple Topics and Exception Handling**

This example refines the previous approach by handling multiple topics and incorporates basic exception handling. It demonstrates better real-world cleanup practices.

```java
import org.apache.kafka.clients.admin.AdminClient;
import org.apache.kafka.clients.admin.AdminClientConfig;
import org.apache.kafka.clients.admin.DeleteTopicsResult;
import org.apache.kafka.clients.admin.NewTopic;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testcontainers.containers.KafkaContainer;
import org.testcontainers.utility.DockerImageName;

import java.time.Duration;
import java.util.Arrays;
import java.util.List;
import java.util.Properties;
import java.util.concurrent.ExecutionException;

public class KafkaMultipleTopicCleanupTest {

    private static final Logger log = LoggerFactory.getLogger(KafkaMultipleTopicCleanupTest.class);
    private static KafkaContainer kafka;
    private static AdminClient kafkaAdmin;
    private static final List<String> TEST_TOPICS = Arrays.asList("test-topic-1", "test-topic-2", "test-topic-3");


    @BeforeAll
    static void setup() throws ExecutionException, InterruptedException {
        kafka = new KafkaContainer(DockerImageName.parse("confluentinc/cp-kafka:7.5.0"));
        kafka.start();

        Properties properties = new Properties();
        properties.put(AdminClientConfig.BOOTSTRAP_SERVERS_CONFIG, kafka.getBootstrapServers());
        kafkaAdmin = AdminClient.create(properties);

        for (String topic : TEST_TOPICS) {
            NewTopic newTopic = new NewTopic(topic, 1, (short) 1);
            kafkaAdmin.createTopics(Collections.singletonList(newTopic)).all().get();
        }
    }

    @Test
    void testMultipleTopics() {
        // Test logic using multiple topics
        assert(true);
    }

    @AfterAll
    static void tearDown() {
        try {
            if (kafkaAdmin != null) {
                DeleteTopicsResult deleteTopicsResult = kafkaAdmin.deleteTopics(TEST_TOPICS);
                deleteTopicsResult.all().get();
                kafkaAdmin.close(Duration.ofSeconds(5));

                log.info("Topics deleted successfully: {}", TEST_TOPICS);
            }
        } catch(Exception e) {
            log.error("Error deleting topics", e);
        }
       finally {
        if(kafka!=null) {
            kafka.stop();
            kafka.close();
        }
    }

    }
}

```

*Commentary:* This example illustrates handling multiple topics created in the test suite. It uses `log` for error tracking during the cleanup phase. The `finally` block ensures that even if deleting the topics fails, the container is shut down which will prevent any lingering processes. This showcases more resilient cleanup practices and improves the debuggability of tests. The use of SLF4j for logging is essential in maintaining a comprehensive trace of the cleanup process.

**Resource Recommendations:**

For understanding Kafka client concepts, refer to the official Apache Kafka documentation for its Java client. For Testcontainers, the official website contains detailed API references and guides on container management. Additionally, reviewing community-driven resources focusing on integration testing practices is beneficial. While specific Kafka Admin Client usage is detailed in the Apache Kafka documentation itself, looking at examples within the framework documentation related to integration tests using Kafka can yield additional context and use cases. When considering persistent volumes with Testcontainers, reviewing related documentation on data management can help avoid scenarios where data may not be correctly discarded after tests.

Effective resource cleanup is not merely a best practice, but a fundamental requirement for constructing reliable and maintainable integration tests. Without employing a robust cleanup strategy, the benefits of Testcontainers are easily offset by the challenges associated with unpredictable test environments and resource leaks. Therefore, the recommended practices outlined, along with appropriate documentation research, are critical to consider when integrating Kafka within a test suite.
