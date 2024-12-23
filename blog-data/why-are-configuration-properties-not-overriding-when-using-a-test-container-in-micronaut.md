---
title: "Why are configuration properties not overriding when using a test container in Micronaut?"
date: "2024-12-23"
id: "why-are-configuration-properties-not-overriding-when-using-a-test-container-in-micronaut"
---

Alright, let's delve into this. I remember a rather frustrating debugging session a few years back when I first encountered this precise issue with Micronaut and testcontainers. It felt like my configuration was simply being ignored, leaving me scratching my head. The core problem, as I later discovered, stems from how Micronaut’s configuration system prioritizes sources, particularly within a test environment where the lifecycle of application contexts becomes quite dynamic.

The short answer is that default configuration loading has a specific order, and when you introduce testcontainers into the mix, this order can be subtly disrupted. Micronaut, by design, loads configurations from a multitude of sources: command-line arguments, environment variables, `application.yml` (or `.properties`) files, and more. This prioritization is generally beneficial, allowing for fine-grained control over settings in different environments. However, in testing, especially with container-backed databases or services, you often want test-specific configurations to override the main application's defaults. The problem occurs because these test-specific settings need to be loaded *after* the base configuration, which may not happen as intuitively as one might expect.

Typically, Micronaut leverages the `application.yml` (or `application.properties`) file located in your `src/main/resources` folder as the primary configuration source. Within your test suite (typically `src/test/resources`), you often have `application-test.yml` or similar files intended for overriding settings during tests. However, when testcontainers are involved, the container’s startup sequence and Micronaut's context initialization can create a timing issue. The testcontainer might be up and running with its specific port or connection details *after* Micronaut loads its initial configuration from the base application file. Consequently, the test-specific configuration, intended to override, doesn't take effect.

I’ve seen this manifest in scenarios like a test trying to access a database on a default port (e.g., 5432 for PostgreSQL), while the testcontainer actually exposes it on a different, dynamic port. The test fails because Micronaut's data source is configured with the original, hardcoded port. We might think that the `application-test.yml` which provides a different database URL to the container would be used by the tests, but that is often not the case. This is a common hurdle and requires a bit of understanding how Micronaut’s configuration is loaded, how the application context is created and the timings involved.

To address this, we usually need to implement a strategy to ensure test-specific configurations are loaded at the right moment. Here are three approaches I’ve found consistently effective, accompanied by short code snippets:

**1. Using `@MicronautTest` with Configuration Properties Overrides:**

   Micronaut's `@MicronautTest` annotation is a powerful tool for setting up test environments. One of its features allows specifying configuration properties directly, which are guaranteed to override defaults in the main config files. This works very well for simple overrides, specifically if you know beforehand what the testcontainers provide in terms of ports or connection strings.

   ```java
   import io.micronaut.test.extensions.junit5.annotation.MicronautTest;
   import org.junit.jupiter.api.Test;
   import jakarta.inject.Inject;
   import javax.sql.DataSource;

   import static org.junit.jupiter.api.Assertions.assertNotNull;

   @MicronautTest(propertySources = @PropertySource(value = "classpath:application-test.yml"))
   class MyServiceTest {

       @Inject
       DataSource dataSource;

       @Test
       void testDataSourceConnection() {
           assertNotNull(dataSource); // Will pass using test-specific properties
       }
   }
   ```
   In this example, the `@MicronautTest` annotation explicitly points to a `application-test.yml` which contains the test-specific settings. The `propertySources` attribute allows explicit injection of the test file.

**2. Programmatic Configuration Using `EmbeddedServer` and `ApplicationContextBuilder`:**

   A more granular approach involves programmatically building the application context within your test. This provides the most control over configuration loading and allows for dynamically injecting configurations based on the environment of the running testcontainer. This is useful when the ports or connection strings are only known at test runtime.

   ```java
   import io.micronaut.context.ApplicationContext;
   import io.micronaut.context.ApplicationContextBuilder;
   import io.micronaut.runtime.server.EmbeddedServer;
   import org.junit.jupiter.api.AfterAll;
   import org.junit.jupiter.api.BeforeAll;
   import org.junit.jupiter.api.Test;
   import jakarta.inject.Inject;
   import javax.sql.DataSource;

   import org.testcontainers.containers.PostgreSQLContainer;
   import static org.junit.jupiter.api.Assertions.assertNotNull;

   class MyServiceProgrammaticTest {

        private static PostgreSQLContainer<?> postgreSQLContainer;
        private static EmbeddedServer embeddedServer;

        @BeforeAll
       public static void setup() {
         postgreSQLContainer = new PostgreSQLContainer<>("postgres:15");
           postgreSQLContainer.start();
           ApplicationContext applicationContext = new ApplicationContextBuilder()
                   .properties(Map.of(
                           "datasources.default.url", postgreSQLContainer.getJdbcUrl(),
                           "datasources.default.username", postgreSQLContainer.getUsername(),
                           "datasources.default.password", postgreSQLContainer.getPassword()))
                  .build();
           embeddedServer = applicationContext.getBean(EmbeddedServer.class).start();
       }


       @AfterAll
       static void tearDown(){
          if (embeddedServer != null) {
            embeddedServer.stop();
          }
           if(postgreSQLContainer != null){
               postgreSQLContainer.stop();
           }
       }
       @Inject
       DataSource dataSource;

       @Test
       void testDataSourceConnection() {
           assertNotNull(dataSource); // Will pass with programmatically set properties
       }
   }

   ```
   Here, we start a PostgreSQL testcontainer programmatically. We then use `ApplicationContextBuilder` to configure the data source properties based on the testcontainer’s settings *before* the Micronaut application context starts. This ensures the correct settings are active from the get-go.

**3. Leveraging Testcontainers' Service Discovery (if applicable):**

   If you're working with service discovery, such as Consul or Eureka, testcontainers often provide built-in methods for handling their discovery. While this is not as frequently needed for smaller applications, if you use discovery for integration, use a similar approach to the above example to configure this programmatically by adding necessary configurations.

**Recommended Resources:**

For a comprehensive understanding of Micronaut’s configuration system, I'd highly recommend checking out the official Micronaut documentation, specifically the sections on configuration loading and environments. The section on testing provides useful insight into the various mechanisms available for testing, including configuration.

For deeper knowledge on the integration between testcontainers and Java, the official Testcontainers documentation is an invaluable resource. It’s always kept up-to-date and provides the most precise and correct information on setting up test environments using containers.

Additionally, *Effective Java* by Joshua Bloch includes valuable chapters on design patterns and handling complex configurations, which have often helped me in structuring applications with configurable properties. It isn't specific to Micronaut, but the design patterns apply generally.

To recap, the reason your configuration properties might not be overriding in your Micronaut test with testcontainers often comes down to the timing and ordering of configuration loading. Utilizing the `@MicronautTest` annotation with explicit `propertySources`, programmatically configuring the `ApplicationContextBuilder`, or leveraging testcontainers service discovery are methods that ensure your test-specific settings are loaded appropriately, allowing for more robust and reliable tests. Understanding the specific sequence of events during context creation is paramount to getting these configurations to work as expected.
