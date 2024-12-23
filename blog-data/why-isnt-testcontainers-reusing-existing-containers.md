---
title: "Why isn't TestContainers reusing existing containers?"
date: "2024-12-23"
id: "why-isnt-testcontainers-reusing-existing-containers"
---

Okay, let's tackle this one. It’s a question that surfaces frequently enough, and frankly, it’s often more nuanced than it might initially seem. Over the years, I've bumped into this exact scenario multiple times – specifically during CI/CD pipeline optimizations, where every second shaved off builds was a big win. The question at hand is: why doesn't testcontainers, a brilliant library designed to spin up temporary containers for integration tests, always reuse existing containers? It's not due to some oversight or architectural flaw, but rather a careful balancing act between test integrity, isolation, and practical limitations.

The heart of the matter lies in the intended use case of testcontainers. It's engineered to create ephemeral environments. Each test execution, ideally, should start with a fresh, known state to guarantee the validity and reproducibility of results. Reusing containers would inherently introduce stateful behavior between tests, creating the very flakiness we strive to eliminate with automated testing. Imagine two tests that both rely on writing to the same database within a container. If the second test ran and was expecting a blank slate, the data left over from the first test would lead to spurious failures and a lot of debugging headaches.

There are several facets to this issue. Firstly, testcontainers is designed to be a black box for its container management. It abstracts away the underlying docker infrastructure to provide a consistent experience across various environments. Its default behavior is to treat each test run as independent and needing its own pristine container instance. This design simplifies the API and reduces the mental overhead for developers. Instead of worrying about container lifecycle management, you simply define what you need in your test setup, and testcontainers handles the rest – including container creation, startup, and teardown.

Second, container reuse introduces complexities around lifecycle management. If testcontainers were to reuse containers, it would need to maintain a persistent state of which containers were available, which were in use, and their configuration. This would involve implementing caching and cleanup mechanisms. While not inherently impossible, it adds significant complexity. The current approach avoids these issues and keeps the library streamlined. Remember, testcontainers is built on top of the docker api, so managing container states would imply maintaining yet another state machine which would likely fail to scale.

Third, practical considerations. Docker itself, while efficient, has overhead. Spinning up containers takes time, which is the primary performance concern when running integration tests. However, it's often quicker to simply create a new container from an existing image, than it is to thoroughly reset the container to a blank state, especially when more complex or customized image configurations are in play. It’s about trading off between potential reuse benefits against the complexity it creates and the risks of inconsistent state between tests. The time to tear down a container and create a new one from its image is often less than what it would take to programmatically reset an existing container, and this process is more predictable.

Now, having stated all of that, there are situations where some level of ‘reuse’ might seem feasible, or at least, a performance optimization could be explored. Instead of reusing the *container*, we could reuse the *image*. Testcontainers provides configuration options to help here. Docker images are immutable, and the layer caching mechanisms within docker make fetching existing layers of an image significantly faster if the image is already present on the host machine. Thus, testcontainers does take advantage of docker’s caching when creating new containers. You won't always get exactly identical performance when running a test the first time vs multiple times locally, but subsequent runs after a first one should be much faster. The creation of the container still occurs, however, reusing layers of a Docker image still provides considerable optimization.

To illustrate these points, let’s walk through some examples.

**Example 1: Standard Usage (No Reuse)**

Here's the most basic example of how testcontainers is used in java, which highlights why container reuse is not the default:

```java
import org.testcontainers.containers.PostgreSQLContainer;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class ExampleTest {

    @Test
    void testDatabaseConnection() {
        try (PostgreSQLContainer<?> postgres = new PostgreSQLContainer<>("postgres:15-alpine")) {
            postgres.start();
            // Perform test logic here using the started postgres container
            assertTrue(postgres.isRunning());
        }
    }

    @Test
    void anotherTestDatabaseConnection(){
        try (PostgreSQLContainer<?> postgres = new PostgreSQLContainer<>("postgres:15-alpine")){
          postgres.start();
           // perform some test logic.
          assertTrue(postgres.isRunning());
        }
    }
}
```

In this scenario, every time `testDatabaseConnection` or `anotherTestDatabaseConnection` is executed, a fresh postgres container is created and started. There is no attempt to reuse the previous instance, which guarantees isolation and ensures each test is performed against a clean slate.

**Example 2: Image Reuse (Implicit)**

Testcontainers will, however, take advantage of image caching when spinning up the container instances. Even though new containers are created every test run, underlying layers will be reused from previous runs. This helps speed up the creation process.

```java
import org.testcontainers.containers.PostgreSQLContainer;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class ExampleTestWithImplicitReuse {

    @Test
    void testDatabaseConnection() {
        try (PostgreSQLContainer<?> postgres = new PostgreSQLContainer<>("postgres:15-alpine")) {
            long start = System.currentTimeMillis();
            postgres.start();
            long end = System.currentTimeMillis();
            System.out.println("First run time: " + (end - start));
             assertTrue(postgres.isRunning());

        }
    }

    @Test
    void anotherTestDatabaseConnection() {
        try (PostgreSQLContainer<?> postgres = new PostgreSQLContainer<>("postgres:15-alpine")) {
            long start = System.currentTimeMillis();
            postgres.start();
            long end = System.currentTimeMillis();
            System.out.println("Second run time: " + (end-start));
            assertTrue(postgres.isRunning());
        }
    }
}
```
Here, when running each of these tests sequentially, the second one will generally spin up faster since docker is able to reuse image layers. Although each instance is fresh, the performance is improved using docker's caching mechanism, with testcontainers benefiting from it.

**Example 3: Explicit Network Reuse (Limited form of Reuse)**

While not typical container reuse, in certain cases, such as with complex network setups involving linked containers, testcontainers enables defining and reusing named networks. This is a form of ‘reuse’ that is carefully scoped to network contexts.

```java
import org.testcontainers.containers.Network;
import org.testcontainers.containers.GenericContainer;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class ExampleTestWithNetworkReuse {
        private static final Network network = Network.newNetwork();
    @Test
    void testContainersInNetwork() {
        try (
                GenericContainer<?> firstContainer = new GenericContainer<>("alpine")
                    .withNetwork(network)
                    .withNetworkAliases("first")
                    .withCommand("sleep infinity");
                GenericContainer<?> secondContainer = new GenericContainer<>("alpine")
                    .withNetwork(network)
                    .withNetworkAliases("second")
                    .withCommand("sleep infinity");

            )
        {
            firstContainer.start();
            secondContainer.start();
            assertTrue(firstContainer.isRunning());
            assertTrue(secondContainer.isRunning());
        }
    }

    @Test
    void anotherTestContainersInNetwork() {
        try (
                 GenericContainer<?> firstContainer = new GenericContainer<>("alpine")
                    .withNetwork(network)
                    .withNetworkAliases("first")
                    .withCommand("sleep infinity");
                GenericContainer<?> secondContainer = new GenericContainer<>("alpine")
                    .withNetwork(network)
                    .withNetworkAliases("second")
                    .withCommand("sleep infinity");
        )
        {
            firstContainer.start();
            secondContainer.start();
            assertTrue(firstContainer.isRunning());
            assertTrue(secondContainer.isRunning());
        }
    }
}
```

In this scenario, even if these tests execute one after another, the `Network` created once at class level will be reused, allowing both containers in each test to share the same network. It's critical to recognize that while the network is reused here, the containers still get created anew each test.

To dig deeper into docker and its inner workings I’d suggest diving into "Docker in Action" by Jeff Nickoloff, or "The Docker Book" by James Turnbull. These resources will provide a solid foundation for understanding containerization at a lower level.

In summary, the decision not to reuse containers in testcontainers is not an oversight but rather a deliberate design choice that prioritizes test reliability and consistency over potential performance gains from container reuse. While some form of reuse can be achieved through networking and cached image layers, the library fundamentally prefers creating ephemeral environments to avoid stateful interactions. This approach might seem slower initially, but it is essential for the reliability of automated testing.
