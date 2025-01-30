---
title: "How do TestContainers manage random ports for Scala Cassandra tests?"
date: "2025-01-30"
id: "how-do-testcontainers-manage-random-ports-for-scala"
---
Cassandra, in a typical testing scenario, requires a port to be accessible for connections. When running concurrent tests or multiple Cassandra instances, relying on a fixed port becomes problematic due to port conflicts. Testcontainers, through its abstraction layer, provides a robust mechanism for dynamically allocating random ports, eliminating these conflicts and improving the reliability of integration tests. I've encountered this challenge numerous times while building distributed data pipelines at my previous company, where concurrent testing with Cassandra was crucial.

Testcontainers’ core functionality, regardless of the underlying database technology, involves containerizing services. For Cassandra, it spins up a Docker container of a configured version. The crucial part for random port allocation lies within how Testcontainers manages the exposed ports specified in the Dockerfile of the target service. When we define ports to be exposed in a Dockerfile, such as port `9042` for native client connections, Testcontainers doesn’t bind it directly to a fixed host port. Instead, it leverages Docker’s inherent ability to publish container ports to random free ports on the host. This is done automatically through a sophisticated binding mechanism implemented within the Testcontainers library.

When initiating a container instance through Testcontainers in Scala, such as with `CassandraContainer`, we define the container image and the ports we want to expose. Testcontainers then:

1.  Pulls the specified Docker image (if it isn’t already present locally).
2.  Creates a Docker container instance based on that image.
3.  **Crucially, it maps the exposed container ports (e.g., 9042) to dynamically assigned free ports on the host operating system.**
4.  Provides a programmatic way to query the assigned port number(s) on the host.

This dynamic mapping is managed by Docker and facilitated by Testcontainers. We don't dictate the specific port number; rather, Testcontainers retrieves this allocated port through Docker API. After retrieving this dynamically allocated port, the Testcontainers library surfaces it for the tests to use when configuring their client connections. This process ensures no conflicts since each newly initiated container gets its unique set of available random ports.

Here is a basic code example utilizing the Java API (which is easily usable within Scala) to illustrate this:

```java
import org.testcontainers.containers.CassandraContainer;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class CassandraPortTest {

    @Test
    public void testCassandraContainerWithDynamicPort() {
        try (CassandraContainer<?> cassandra = new CassandraContainer<>("cassandra:latest")) {
            cassandra.start();

            Integer mappedPort = cassandra.getMappedPort(9042); // Native client port
            assertTrue(mappedPort > 0, "Dynamic port allocation failed.");
            System.out.println("Mapped port for Cassandra: " + mappedPort);
            
            // Tests can now use this mappedPort to establish client connections.
        }
    }
}
```
This Java code, readily adapted for a Scala project using the same test libraries, starts a Cassandra container. The line `cassandra.getMappedPort(9042)` retrieves the dynamically allocated port for Cassandra’s native client port. The assertion verifies that the port is indeed a positive number, indicating a successful allocation, demonstrating how Testcontainers grants direct access to dynamically mapped port. The test outputs the resolved port for verification. I've utilized a similar setup countless times, confirming the reliability of the port mapping.

The underlying mechanism uses Docker’s port mapping capabilities. Docker manages the binding of container ports to host ports, and Testcontainers' role is to retrieve the random port chosen by Docker. We avoid any manual port management, making the tests less fragile and more resilient to environmental variability. This process isn't limited to Cassandra. It is a general principle applicable to any containerized service.

Now, consider a slightly more involved example using Scala Test's framework and the Scala API for Testcontainers, highlighting typical use.

```scala
import org.testcontainers.containers.CassandraContainer
import org.testcontainers.utility.DockerImageName
import org.scalatest.BeforeAndAfterAll
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import com.datastax.oss.driver.api.core.{CqlSession, CqlSessionBuilder}

class ScalaCassandraTest extends AnyFlatSpec with Matchers with BeforeAndAfterAll {

  private var cassandraContainer: CassandraContainer[_] = _
  private var session: CqlSession = _

  override def beforeAll(): Unit = {
    cassandraContainer = new CassandraContainer(DockerImageName.parse("cassandra:latest"))
    cassandraContainer.start()
    val mappedPort = cassandraContainer.getMappedPort(9042)

    session = CqlSession.builder()
      .addContactPoint(new java.net.InetSocketAddress("localhost", mappedPort))
      .withLocalDatacenter("datacenter1")
      .build()

  }

  override def afterAll(): Unit = {
    if (session != null) session.close()
    if (cassandraContainer != null) cassandraContainer.stop()
  }

  "Cassandra" should "be reachable with allocated port" in {
    val result = session.execute("SELECT release_version FROM system.local").one()
    result.getString("release_version") should not be empty
  }
}
```

In this Scala example, we declare a Cassandra container, starting it with `cassandraContainer.start()`. Immediately afterward, `cassandraContainer.getMappedPort(9042)` is used to retrieve the dynamically generated port for the native CQL client. This retrieved port is then used in the `CqlSession` builder, enabling a successful connection to the Cassandra instance running inside the Docker container. The test executes a basic CQL query to verify connectivity, reinforcing the practical application of dynamic port allocation. This setup reflects a real-world scenario where I needed to integrate with Cassandra and required a reliable and consistent testing environment. The clean-up process in the `afterAll` function ensures the container and connection are properly closed, preventing resource leaks.

For more complex test scenarios involving multiple Cassandra containers or advanced configurations, I would recommend exploring the more advanced configuration options Testcontainers provides for custom network setups. Testcontainers provides mechanisms to create a Docker network and connect containers to this network. This simplifies testing interactions between containers in a more realistic manner, allowing, for instance, a microservice to communicate with a Cassandra cluster running as part of the integration test.

Let's consider a final code sample which illustrates creating a custom Docker network.

```scala
import org.testcontainers.containers.{CassandraContainer, Network}
import org.testcontainers.utility.DockerImageName
import org.scalatest.BeforeAndAfterAll
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import com.datastax.oss.driver.api.core.{CqlSession, CqlSessionBuilder}

class MultiCassandraTest extends AnyFlatSpec with Matchers with BeforeAndAfterAll {

    private val network: Network = Network.newNetwork()
    private var cassandraContainer1: CassandraContainer[_] = _
    private var cassandraContainer2: CassandraContainer[_] = _
    private var session1: CqlSession = _
    private var session2: CqlSession = _

    override def beforeAll(): Unit = {
        cassandraContainer1 = new CassandraContainer(DockerImageName.parse("cassandra:latest"))
          .withNetwork(network).withNetworkAliases("cassandra1")
        cassandraContainer2 = new CassandraContainer(DockerImageName.parse("cassandra:latest"))
            .withNetwork(network).withNetworkAliases("cassandra2")

        cassandraContainer1.start()
        cassandraContainer2.start()

        val mappedPort1 = cassandraContainer1.getMappedPort(9042)
        val mappedPort2 = cassandraContainer2.getMappedPort(9042)


        session1 = CqlSession.builder()
            .addContactPoint(new java.net.InetSocketAddress("localhost", mappedPort1))
            .withLocalDatacenter("datacenter1")
            .build()

        session2 = CqlSession.builder()
            .addContactPoint(new java.net.InetSocketAddress("localhost", mappedPort2))
            .withLocalDatacenter("datacenter1")
            .build()
    }

    override def afterAll(): Unit = {
      if (session1 != null) session1.close()
      if (session2 != null) session2.close()
        if (cassandraContainer1 != null) cassandraContainer1.stop()
        if (cassandraContainer2 != null) cassandraContainer2.stop()
        if (network != null) network.close()
    }

    "Both Cassandra instances" should "be reachable with allocated ports" in {
        val result1 = session1.execute("SELECT release_version FROM system.local").one()
        val result2 = session2.execute("SELECT release_version FROM system.local").one()

        result1.getString("release_version") should not be empty
        result2.getString("release_version") should not be empty

    }
}
```

This Scala example demonstrates setting up two Cassandra containers on a custom Docker network. `Network.newNetwork()` creates an isolated network, and each Cassandra container joins this network via `withNetwork(network)`. `withNetworkAliases` allows the containers to communicate using network names instead of IP addresses.  The key point here is that each container has its allocated port, accessible through `getMappedPort`. We configure `CqlSession` to connect to both instances on their respective dynamically allocated ports, further showcasing how Testcontainers streamlines managing multiple containers.

For individuals looking to further their expertise with Testcontainers and Cassandra, I would recommend the official Testcontainers documentation, which is meticulously detailed. In addition, reviewing blog posts detailing sophisticated testing strategies using containers provides a great understanding of best practices. Exploring the Cassandra project's documentation, especially its driver setup guidance, proves invaluable. Finally, examination of real-world integration tests within open-source projects provides pragmatic insights.
