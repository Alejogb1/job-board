---
title: "How to reuse containers using TestContainers in scala?"
date: "2024-12-15"
id: "how-to-reuse-containers-using-testcontainers-in-scala"
---

alright, so you're asking about reusing containers with testcontainers in scala, eh? been there, done that, got the t-shirt (and a few sleepless nights debugging integration tests, ha!). it's a common challenge, and frankly, it's one of the things that can make your test suites run super slow if you don’t handle it properly. let's break it down.

the core issue is that by default, testcontainers will spin up a new container instance every single time your test runs. while this is great for isolation, it's just not practical for larger projects or environments where you have several tests needing the same service—like a database, a message queue, or even a custom app. imagine how long your pipeline would take if it had to start a postgres instance for every single test case. it's painful.

my first real encounter with this was, i think, back in 2018. we were building a microservice architecture, and every service had its own integration test suite. the build time started to go from minutes to over an hour! i remember trying all sorts of funky ways to share containers, involving custom docker compose setups and manual port mappings. needless to say, it was a mess. that's when i really started diving deeper into how testcontainers lifecycle works.

the solution to container reuse lies mainly in using testcontainers' lifecycle features in smart ways, especially container caching. we can achieve reuse on a per-jvm basis, which is usually what you want. basically, we tell testcontainers to keep a container running if a previous one with the same configuration already exists within the jvm. it's much simpler than it sounds!

the first thing to nail down is how to configure your containers. the key is to make sure that the container definition, from testcontainers' perspective, is consistent across your test suites. this means the same image, ports, environment variables, and volumes. if any of these change, then testcontainers won't reuse the container because it sees it as a different container instance.

here's a simple example of how you can configure a postgres container in scala with testcontainers:

```scala
import org.testcontainers.containers.PostgreSQLContainer

object PostgresContainer {
  val container: PostgreSQLContainer[_] = new PostgreSQLContainer("postgres:13.3")
    .withDatabaseName("testdb")
    .withUsername("testuser")
    .withPassword("testpass")

  container.start()
  sys.addShutdownHook(container.stop())
}
```

in this snippet, we’ve created a singleton object that holds our postgres container. crucially, we've also included `container.start()` which fires once the jvm process is started and then a hook which takes care of tearing down when our program exists. every test can then reuse this single container instance. the `withDatabaseName`, `withUsername` and `withPassword` method calls ensure that the definition of the container is fixed. the `postgres:13.3` image tag fixes the image and version of this container.

now, let's see how to use it. here's a basic example of a test using scalatest:

```scala
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import java.sql.{DriverManager, ResultSet}
import PostgresContainer.container

class ExampleDatabaseTest extends AnyFlatSpec with Matchers {

  "The database" should "be up and running" in {
    val url = container.getJdbcUrl
    val username = container.getUsername
    val password = container.getPassword

    val connection = DriverManager.getConnection(url, username, password)
    val statement = connection.createStatement()
    val resultSet: ResultSet = statement.executeQuery("SELECT 1;")

    resultSet.next()
    val result = resultSet.getInt(1)
    result shouldBe 1
    connection.close()
  }
}
```

note that in this example, we are not starting or stopping the container as that logic is in the `PostgresContainer` singleton object. the key is we are accessing the same `container` instance for every test that uses the `PostgresContainer` object. the `ExampleDatabaseTest` can be repeated n number of times without restarting the container, and as long as it is run in the same jvm process, reusing the container. this approach will greatly improve your integration test performance.

now, there's a twist. what if you've got more complex container setups, like ones that involve several containers linked together, with dependencies between them? testcontainers has a feature called 'generic containers' which works like a charm for that.

you would usually use `DockerComposeContainer` and `network`, in practice, but for simplicity, i'll show how to set up two containers in a network using `GenericContainer` only. imagine a scenario where you have an application container that depends on a kafka broker. the setup would look like this:

```scala
import org.testcontainers.containers.{GenericContainer, Network}
import org.testcontainers.utility.DockerImageName

object KafkaWithAppContainers {
  val network = Network.newNetwork()

  val kafka = new GenericContainer(DockerImageName.parse("confluentinc/cp-kafka:latest"))
    .withNetwork(network)
    .withNetworkAliases("kafka")
    .withEnv("KAFKA_LISTENERS", "PLAINTEXT://0.0.0.0:9092")
    .withEnv("KAFKA_ADVERTISED_LISTENERS", "PLAINTEXT://kafka:9092")
    .withEnv("KAFKA_ZOOKEEPER_CONNECT", "zookeeper:2181")
    .withExposedPorts(9092)

  val zookeeper = new GenericContainer(DockerImageName.parse("confluentinc/cp-zookeeper:latest"))
    .withNetwork(network)
    .withNetworkAliases("zookeeper")
    .withEnv("ZOOKEEPER_CLIENT_PORT", "2181")
    .withExposedPorts(2181)


  val app = new GenericContainer(DockerImageName.parse("your-application-image:latest"))
    .withNetwork(network)
    .withEnv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
    .dependsOn(kafka) //dependsOn is key here!
    .withExposedPorts(8080)

  zookeeper.start()
  kafka.start()
  app.start()

  sys.addShutdownHook {
    app.stop()
    kafka.stop()
    zookeeper.stop()
  }

}
```

here we have three containers, a kafka broker, a zookeeper instance, and a sample application that depends on kafka. the important part is that each one of these containers has been added to a network via `withNetwork` that allows them to be discoverable by their alias specified in `withNetworkAliases`. the `dependsOn` clause tells testcontainers to start kafka before starting the app, ensuring that dependencies are set up correctly. again, if you reuse this `KafkaWithAppContainers` object in different test files in the same jvm, then the containers will be reused.

a quick note on container cleanup. testcontainers automatically handles stopping and removing containers when the jvm process exists, as seen in the `sys.addShutdownHook` call. this prevents dangling containers. you can also manually manage the container lifecycle using `.stop()` if you need that.

for resources, i strongly recommend diving into the testcontainers documentation itself. it's quite extensive. there are also some good articles and blog posts out there, but the official docs and the code is what really gave me a good grasp on all this. books like "test-driven development" by kent beck are very useful and will give you very clear guidance on how to approach testing, although not directly related to testcontainers itself. another resource worth checking is "effective java" by joshua bloch where principles of effective java programming are explained which is applicable when using external libraries like testcontainers.

to recap, container reuse in testcontainers using scala is generally achieved through a static singleton container instance. when you run all your integration tests they will reuse this static container object. avoid creating a new container instance for every test case, instead, centralize and reuse the container definition via a singleton object. and always define dependencies between containers where needed. that's the simple truth. it makes your tests faster, your dev cycle happier, and your computer fans slightly less loud, and i think that’s a win, win, win.
