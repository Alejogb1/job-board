---
title: "Why does Eclipse execute integration tests but Maven does not?"
date: "2025-01-30"
id: "why-does-eclipse-execute-integration-tests-but-maven"
---
The discrepancy between Eclipse's execution of integration tests and Maven's failure to do so typically stems from a misconfiguration of the project's build lifecycle and the associated test execution plugins.  In my experience troubleshooting similar issues across numerous Java projects, I've observed that this often boils down to a lack of explicit definition of the integration test phase within the Maven build, or an incorrect mapping between the project's directory structure and the plugin configurations. Eclipse, depending on its configuration (specifically its build mechanism, usually through the m2e plugin), may implicitly execute tests found in designated directories, whereas Maven requires explicit instructions.


**1. Clear Explanation:**

Maven's lifecycle is structured around distinct phases, each responsible for a specific aspect of the build process.  Crucially, Maven does not automatically recognize and execute tests merely because they exist; it needs clear instructions to do so. The standard Maven lifecycle includes phases like `compile`, `test`, `package`, and `install`.  The `test` phase, by default, executes *unit* tests located in the `src/test/java` directory.  Integration tests, however, usually reside in a different directory, for example, `src/it/java` (integration-test). Because integration tests often involve external dependencies and require a different setup than unit tests (e.g., database connections, external services), they are not implicitly included in the standard `test` phase.

To execute integration tests with Maven, we must explicitly configure the build to recognize and execute them. This typically involves using the `failsafe-plugin`, distinct from the `surefire-plugin` used for unit tests. The `failsafe-plugin` is responsible for executing tests found in the designated integration-test directory.  If this plugin isn't configured or its configuration is incorrect (pointing to the wrong directory or using an incompatible version), Maven will not execute integration tests even if they exist.

Furthermore, the failure might also be rooted in the dependency management.  Integration tests often depend on specific libraries or modules not required for unit tests or the main application.  If these dependencies aren't properly declared in the project's `pom.xml`, the integration tests might fail to compile or run due to missing classes or resources.


**2. Code Examples with Commentary:**

**Example 1: Correct Configuration with `failsafe-plugin`**

This example demonstrates the proper configuration of the `failsafe-plugin` to execute integration tests from the `src/it/java` directory.


```xml
<project>
  ...
  <build>
    <plugins>
      <plugin>
        <groupId>org.apache.maven.plugins</groupId>
        <artifactId>maven-surefire-plugin</artifactId>
        <version>3.0.0-M7</version> <!-- Use appropriate version -->
        <configuration>
          <testSourceDirectory>src/test/java</testSourceDirectory>
        </configuration>
      </plugin>
      <plugin>
        <groupId>org.apache.maven.plugins</groupId>
        <artifactId>maven-failsafe-plugin</artifactId>
        <version>3.0.0-M7</version> <!-- Use appropriate version -->
        <executions>
          <execution>
            <goals>
              <goal>integration-test</goal>
              <goal>verify</goal>
            </goals>
          </execution>
        </executions>
        <configuration>
          <integrationTest>
              <skipTests>false</skipTests>
          </integrationTest>
          <systemPropertyVariables>
            <myProperty>myValue</myProperty> <!--Example System Property-->
          </systemPropertyVariables>
          <includes>
            <include>**/*IT.java</include> <!-- Matching pattern for integration test files -->
          </includes>
          <testSourceDirectory>src/it/java</testSourceDirectory>
        </configuration>
      </plugin>
    </plugins>
  </build>
  ...
</project>
```

This configuration explicitly defines the `integration-test` goal, which runs integration tests.  The `verify` goal ensures that the tests are run and any failures are properly reported.  Note the explicit specification of the `testSourceDirectory` for the `failsafe-plugin`, directing it to the correct location of integration tests.


**Example 2: Incorrect Directory Structure**

This exemplifies a common error:  the integration tests are in a different directory, and the plugin's configuration doesn't reflect this.


```xml
<project>
  ...
  <build>
    <plugins>
       ... <!-- failsafe-plugin is missing or incorrectly configured -->
    </plugins>
  </build>
  ...
</project>
```

If the `src/it/java` directory is used for integration tests, but the `maven-failsafe-plugin` is missing or incorrectly configured (e.g., points to `src/test/java`), Maven will not execute them.  Even if Eclipse, through its m2e plugin, might find and run the tests, Maven won't.



**Example 3: Missing Dependencies in Integration Tests**

This illustrates a scenario where dependencies required by integration tests are missing from the `pom.xml`, preventing execution.

```xml
<project>
  ...
  <dependencies>
    <!-- Dependencies for the main application -->
    <dependency>
      <groupId>com.example</groupId>
      <artifactId>my-library</artifactId>
      <version>1.0.0</version>
    </dependency>
  </dependencies>

  <build>
    <plugins>
      <plugin>
        <groupId>org.apache.maven.plugins</groupId>
        <artifactId>maven-failsafe-plugin</artifactId>
        <version>3.0.0-M7</version>
        <executions>
            <execution>
              <goals>
                <goal>integration-test</goal>
                <goal>verify</goal>
              </goals>
            </execution>
        </executions>
        <configuration>
          <testSourceDirectory>src/it/java</testSourceDirectory>
        </configuration>
      </plugin>
    </plugins>
  </build>
  ...
</project>
```

If the integration tests in `src/it/java` depend on an external library (`com.example:another-library`),  it must be added as a dependency within the `<dependencies>` section, potentially within a `<dependencyManagement>` section if the version is managed elsewhere.  Failure to include this dependency will result in compilation errors during the integration-test phase.  Note that this library needs to be added to the project's dependency scope such as 'test' or 'integration-test' based on its usage.


**3. Resource Recommendations:**

Maven's official documentation, particularly the sections on the `maven-surefire-plugin` and `maven-failsafe-plugin`, are invaluable.  Thorough understanding of the Maven lifecycle and its phases is essential.  Refer to established Java testing best practices guides which address the differences between unit and integration testing, and the appropriate setup for each. Finally, a good understanding of plugin management in Maven will help avoid issues related to plugin versions and configurations.
