---
title: "How can I disable stack trace trimming in JaCoCo Failsafe tests?"
date: "2025-01-30"
id: "how-can-i-disable-stack-trace-trimming-in"
---
Disabling stack trace trimming in JaCoCo Failsafe tests is crucial for comprehensive debugging, particularly when dealing with complex integrations or asynchronous operations. By default, JaCoCo, when used in conjunction with the Maven Failsafe plugin, often trims stack traces to a concise version focusing on the direct exception location within the test method. This behavior, while suitable for basic failures, obscures the underlying call chain when a fault originates deep within the application. I've encountered this limitation numerous times during integration testing of multi-service architectures. I’ve found that a thorough investigation into the root cause of such failures requires the full stack trace.

JaCoCo itself doesn't handle stack trace manipulation. Instead, the trimming is a feature of the Failsafe plugin and how it processes exceptions occurring within integration tests. Failsafe captures the exceptions, and it’s this reporting mechanism that applies the trimming. To disable this behavior, we need to configure Failsafe to provide the full exception details. This configuration involves directly modifying the plugin’s settings within the Maven project's `pom.xml` file. The core mechanism is instructing the plugin to bypass its default exception formatting and instead report exceptions using standard Java reporting.

**Explanation:**

The Maven Failsafe plugin, by default, uses a specific `SurefireReporter` implementation for formatting and outputting test results. This reporter is optimized for succinct error reporting. When an exception is thrown in the test code, the default reporter will capture the exception and then typically only output the last few frames of the stack trace. This approach prioritizes test result readability but sacrifices detailed debugging information. In complex test scenarios where multiple services interact or asynchronous processes are involved, tracing the origin of the fault through a full stack trace becomes critical. By altering the Failsafe configuration, we can direct it to use a standard Java exception reporting mechanism, which effectively disables the trimming. This is achieved by overriding the default `reporter` configuration and specifying that all exceptions should be handled with the default system error stream output.

The necessary configuration is implemented within the `<plugin>` section of your `pom.xml` file. This involves adding or modifying the `<configuration>` element within the plugin definition. Specifically, a nested `<reports>` element needs to be updated with an explicit `<report>` that specifies an empty `reporter` configuration. By setting the `<reporter>` element to empty, the Failsafe plugin utilizes the default Java error reporting mechanism when encountering an exception.

It is important to emphasize that disabling stack trace trimming applies only to the Failsafe tests, which are integration tests performed after the unit test phase. Unit tests using the Surefire plugin typically report full stack traces; the modifications described here should not affect them.

**Code Examples with Commentary:**

The primary modification lies within the `<plugin>` configuration of the `maven-failsafe-plugin` in the `pom.xml` file. Below are illustrative examples showcasing how to achieve disabling of stack trace trimming.

**Example 1: Basic Configuration**

This example demonstrates the simplest configuration needed to disable stack trace trimming.

```xml
<plugin>
    <groupId>org.apache.maven.plugins</groupId>
    <artifactId>maven-failsafe-plugin</artifactId>
    <version>3.2.5</version>
    <executions>
        <execution>
            <goals>
                <goal>integration-test</goal>
                <goal>verify</goal>
            </goals>
        </execution>
    </executions>
    <configuration>
         <reports>
            <report>
               <reporter implementation=""/>
            </report>
         </reports>
    </configuration>
</plugin>

```

*   **`<plugin>`:** Declares the Failsafe plugin.
*   **`<version>`:**  Specifies the plugin version. Using a specific version ensures consistency.
*   **`<executions>`:** Defines when and how the plugin goals are run. The `integration-test` goal runs before the verification and `verify` ensures the test phase completes successfully.
*   **`<configuration>`:** Contains specific settings for the plugin.
*   **`<reports>`:**  Enables configuration of reporting.
*   **`<report>`:** Configures a report; here we define the `<reporter>`
*   **`<reporter implementation="" />`:** The most critical part. Setting the `implementation` attribute to an empty string disables the custom reporter and directs the plugin to use the default exception handling. This results in the full exception stack trace being printed.

**Example 2: Configuration with system properties**

In certain situations, passing system properties to the test JVM can be beneficial. This example adds support for such functionality alongside the core configuration.

```xml
<plugin>
  <groupId>org.apache.maven.plugins</groupId>
  <artifactId>maven-failsafe-plugin</artifactId>
  <version>3.2.5</version>
  <executions>
    <execution>
      <goals>
        <goal>integration-test</goal>
        <goal>verify</goal>
      </goals>
    </execution>
  </executions>
    <configuration>
        <reports>
            <report>
                <reporter implementation=""/>
            </report>
         </reports>
        <systemProperties>
            <property>
                <name>my.custom.property</name>
                <value>myValue</value>
            </property>
        </systemProperties>
    </configuration>
</plugin>
```

*   **`<systemProperties>`:** Allows us to define system properties that are passed to the integration tests.
*   **`<property>`:** Defines a key-value pair for a property.
*   **`<name>`:**  The name of the system property.
*   **`<value>`:** The corresponding value of the system property.
This is an example showing how we can still customize other parts of the Failsafe configuration while explicitly disabling stack trace trimming.

**Example 3:  Incorporating Fork Count**

Integration tests can be resource-intensive. This example demonstrates how to add a fork count to allow better testing in different environments while retaining the full stack trace feature.

```xml
<plugin>
  <groupId>org.apache.maven.plugins</groupId>
  <artifactId>maven-failsafe-plugin</artifactId>
  <version>3.2.5</version>
  <executions>
    <execution>
      <goals>
        <goal>integration-test</goal>
        <goal>verify</goal>
      </goals>
    </execution>
  </executions>
    <configuration>
        <reports>
            <report>
                <reporter implementation=""/>
            </report>
         </reports>
        <forkCount>1</forkCount>
        <reuseForks>false</reuseForks>
    </configuration>
</plugin>
```

*   **`<forkCount>`:**  Specifies the number of JVM instances to run test classes in parallel.
*   **`<reuseForks>`:** Determines if the forked JVM instances are reused. Setting to `false` forces a new JVM for each fork. This demonstrates further customization without impacting stack trace output.

By utilizing these techniques, we can gain full stack trace information during Failsafe testing, which dramatically simplifies the diagnosis of issues, especially when dealing with deep call stacks and asynchronous operations.

**Resource Recommendations:**

For a comprehensive understanding of the Maven Failsafe plugin, its official documentation offers in-depth information about the plugin’s features and configuration. The documentation provided by the Apache Maven project explains its functionality in detail. Additionally, reviewing the documentation for the Surefire plugin can offer helpful insights into the behavior of testing and reporting in the Maven ecosystem in general. Reading the relevant articles on the specifics of system properties and how they behave can give the developer a stronger understanding of their impact on debugging. The Maven project's official website also offers guidance on general best practices for plugin configuration.
