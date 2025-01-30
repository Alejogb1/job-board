---
title: "Why is Thorntail Arquillian JaCoCo test coverage 0%?"
date: "2025-01-30"
id: "why-is-thorntail-arquillian-jacoco-test-coverage-0"
---
Zero percent test coverage reported by JaCoCo within a Thorntail (now Quarkus) Arquillian environment frequently stems from misconfiguration of the JaCoCo agent, specifically concerning its interaction with the application server's classloading mechanism.  My experience debugging similar issues across numerous enterprise-level deployments revealed this as the primary culprit.  The problem isn't necessarily a deficiency in JaCoCo or Arquillian, but a subtle incompatibility requiring precise configuration.

**1.  Explanation:**

Thorntail, a now-deprecated predecessor to Quarkus, utilized a modular classloading strategy.  This means that different parts of your application, including test classes and the application itself, might reside in distinct classloaders.  JaCoCo relies on instrumenting bytecode at runtime to track execution paths.  If the JaCoCo agent, responsible for this instrumentation, is not correctly configured to reach all relevant classloaders, it won't be able to instrument the classes loaded by the application server, leading to zero percent coverage despite seemingly correct test execution. This is exacerbated by the use of Arquillian, which further compartmentalizes the test environment.  The key is ensuring that the JaCoCo agent’s scope encompasses both the test classes and the classes loaded by the deployed application.  Failure to do so creates a scenario where tests run successfully, but JaCoCo reports zero coverage because it never instrumented the application's core code.

Further contributing factors can include improper exclusion filtering within the JaCoCo configuration. Overly restrictive filters can prevent JaCoCo from reaching and instrumenting the necessary code. This happens when developers unintentionally exclude parts of the application they intend to test, often through regular expression mismatches or overzealous patterns.  Finally, issues with the execution environment itself – such as insufficient permissions or conflicting libraries – may indirectly manifest as zero coverage reports, masking the root cause.


**2. Code Examples with Commentary:**

The following examples illustrate how to correctly configure JaCoCo within an Arquillian test environment targeting a Thorntail (or similar application server) deployment.  Note that these are simplified examples and might require adjustments based on your specific project structure and dependencies.  Assume a Maven project structure throughout.


**Example 1:  Correct Configuration (Maven Surefire Plugin)**

```xml
<plugin>
    <groupId>org.apache.maven.plugins</groupId>
    <artifactId>maven-surefire-plugin</artifactId>
    <version>3.0.0-M7</version>  <!-- Or a later suitable version -->
    <configuration>
        <argLine>
            -javaagent:${settings.localRepository}/org/jacoco/org.jacoco.agent/0.8.8/org.jacoco.agent-0.8.8-runtime.jar=destfile=${project.build.directory}/jacoco.exec
        </argLine>
        <!-- Other Surefire configuration -->
    </configuration>
</plugin>
```

**Commentary:** This configuration uses the Maven Surefire plugin to attach the JaCoCo agent as a Java agent (`-javaagent`).  Crucially, the path to the JaCoCo agent JAR is specified, ensuring the agent is loaded correctly.  The `destfile` parameter specifies the location where the execution data will be saved. The version number should match your project's JaCoCo dependency version.  This approach is generally preferred as it integrates the JaCoCo agent seamlessly into the testing lifecycle.


**Example 2:  Illustrating a Common Error (Incorrect Classloading)**

```xml
<!-- Incorrect Configuration: JaCoCo agent not properly integrated -->
<plugin>
    <groupId>org.apache.maven.plugins</groupId>
    <artifactId>maven-surefire-plugin</artifactId>
    <version>3.0.0-M7</version>
    <configuration>
       <!-- Missing -javaagent configuration -->
       <!-- Other Surefire configuration -->
    </configuration>
</plugin>
```

**Commentary:** This example omits the crucial `-javaagent` parameter.  Without this, JaCoCo won't be attached to the JVM, resulting in zero coverage, even if your tests run successfully. This demonstrates a very common source of the 0% coverage problem.


**Example 3:  Incorporating JaCoCo in Arquillian Test Deployment (Simplified)**

This example would need adaptation to your Arquillian setup but illustrates the essential principle:

```java
@RunWith(Arquillian.class)
public class MyArquillianTest {

    @Deployment
    public static Archive<?> createDeployment() {
        JavaArchive jar = ShrinkWrap.create(JavaArchive.class)
                .addClass(MyClassToTest.class)
                // ... other classes ...
                ;

        return jar;
    }

    // ... your test methods ...
}
```

**Commentary:**  This snippet highlights how to create a deployment archive using Arquillian's ShrinkWrap API.  Ensure that `MyClassToTest` and all relevant classes from your application are included in the archive.  The JaCoCo agent, correctly configured as in Example 1, will instrument this archive during deployment. The absence of classes from this deployment is another reason why 0% coverage might be reported.


**3. Resource Recommendations:**

For more in-depth understanding, I recommend consulting the official documentation for JaCoCo, Arquillian, and the specific version of Thorntail (or Quarkus) used in your project.  Thoroughly reviewing the plugin documentation for Maven Surefire and any other relevant build plugins is also crucial.  Understanding classloaders and their behavior within application servers is fundamentally important for resolving these types of integration problems.  Finally, examining the JaCoCo execution data file (typically `jacoco.exec`) for clues about which parts of the code were actually instrumented can offer valuable insights into the root cause.  A detailed understanding of the Maven lifecycle and its phases can significantly aid in troubleshooting.


In summary, the 0% JaCoCo coverage within your Thorntail Arquillian environment is highly likely due to misconfiguration of the JaCoCo agent's classloader integration.  Carefully review your Maven configuration, ensuring correct inclusion of the `-javaagent` parameter, verification of your JaCoCo and Arquillian dependencies, and careful examination of deployment archives for completeness.  Thorough understanding of the underlying mechanisms involved will greatly increase your ability to identify and rectify such issues in the future.  Remember to always consult the official documentation for the most accurate and up-to-date information.
