---
title: "How can Maven, Surefire, and Failsafe be configured correctly?"
date: "2024-12-23"
id: "how-can-maven-surefire-and-failsafe-be-configured-correctly"
---

Okay, let's talk about Maven, Surefire, and Failsafe configurations. This trio often presents challenges, and I’ve definitely spent my fair share of evenings debugging poorly configured test suites. I remember one particularly grueling project back in the early 2010s; a large microservices architecture where integration tests were consistently failing due to a mix-up in plugin executions. It was a painful lesson in the subtleties of lifecycle bindings.

The crux of the issue lies in understanding how these plugins interact within Maven’s lifecycle. Surefire, the workhorse for unit tests, is typically bound to the `test` phase. Failsafe, designed for integration tests, is generally bound to the `integration-test`, `verify`, and sometimes `post-integration-test` phases. The potential for conflicts arises when these phases overlap or when plugin configurations are not appropriately scoped. Let's break it down:

**Surefire Configuration**

Surefire’s primary job is to execute unit tests, those fast, targeted tests against individual components. The key to configuring Surefire effectively is usually about fine-tuning its behavior through configuration parameters within your project's `pom.xml` file. Here's an example snippet that shows some common settings:

```xml
    <plugin>
        <groupId>org.apache.maven.plugins</groupId>
        <artifactId>maven-surefire-plugin</artifactId>
        <version>3.2.5</version>
        <configuration>
            <includes>
                <include>**/*Test.java</include>
            </includes>
             <excludes>
               <exclude>**/*IT.java</exclude>
             </excludes>
            <parallel>methods</parallel>
            <threadCount>4</threadCount>
            <useUnlimitedThreads>false</useUnlimitedThreads>
            <forkCount>2</forkCount>
            <reuseForks>true</reuseForks>
        </configuration>
    </plugin>
```

*   **`includes` and `excludes`**:  These control which test classes are picked up by Surefire.  `**/*Test.java` is a common default, indicating all files ending in `Test.java`. However, you might want to explicitly exclude or include specific packages or naming conventions. In this example, we are explicitly excluding any classes ending in 'IT.java' to separate unit and integration tests.
*   **`parallel`**: This setting allows you to run tests in parallel. `methods` is a common choice, running test methods in parallel. This is essential for reducing test execution time on larger projects. Be mindful of thread safety in your tests when using this setting.
*   **`threadCount`**: Sets the maximum number of threads to use for parallel test execution.
*   **`useUnlimitedThreads`**: If set to true, it ignores 'threadCount' and uses all available processors, which can be risky.
*   **`forkCount`**: Specifies how many JVM instances should be used for running tests in parallel. Forking JVMs can be helpful to avoid resource contention and can also provide better isolation.
*   **`reuseForks`**: Controls whether to reuse JVMs or create a new one for each test run. It's often useful to reuse them to avoid the overhead of starting a JVM for every test execution.

The Maven Surefire Plugin documentation is your best friend here. I recommend spending time familiarizing yourself with the various configuration options beyond what I've covered.

**Failsafe Configuration**

Failsafe, on the other hand, manages integration tests. These are generally tests that operate on a deployed application or interact with external systems. Proper configuration here is crucial for avoiding conflicts with unit tests and managing the full lifecycle of integration tests. Here’s a typical configuration:

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
                <configuration>
                    <includes>
                        <include>**/*IT.java</include>
                    </includes>
                     <excludes>
                        <exclude>**/*Test.java</exclude>
                     </excludes>
                    <parallel>methods</parallel>
                    <threadCount>4</threadCount>
                    <forkCount>1</forkCount>
                    <reuseForks>false</reuseForks>
                </configuration>
            </execution>
        </executions>
    </plugin>
```

*   **`executions`**: Failsafe operates within specific executions. Here, we bind the `integration-test` and `verify` goals. This ensures the integration tests run after the application is deployed and then that their results are verified.
*   **`includes` and `excludes`**: Similar to Surefire, these settings control which classes are considered integration tests. Here, we've explicitly included `*IT.java` which is commonly used to differentiate them from unit tests. We've also excluded unit test classes here, as we don't want them executed twice.
*   **`parallel`, `threadCount`, `forkCount`, `reuseForks`**: As with Surefire, these options enable fine-tuning parallel execution. Note that in this example we set `reuseForks` to `false` to create a clean environment for each integration test execution. This is particularly important for integration tests which might have side effects. We reduced the forkCount to 1 as these tests are often more resource intensive and we don't want to overload the system.
* Note that without explicitly specifying the phases `integration-test` and `verify` using `executions`, the failsafe plugin will not automatically execute integration tests.

Again, I'd strongly encourage you to read through the official Maven Failsafe Plugin documentation. It provides details on various options and specific scenarios that you may encounter.

**Common Pitfalls and Solutions**

Here are some common issues that I’ve experienced, and how to address them:

1.  **Test Classes Not Being Recognized:** Ensure your `includes` and `excludes` settings are correct in both plugins. This is often due to inconsistent naming conventions or incorrect patterns in your configurations. Always double-check your patterns to match the file names in your project directory.
2.  **Overlapping Test Executions**: If both Surefire and Failsafe are picking up the same test classes, you may have some classes being executed twice. Ensure separation of naming conventions; using `*Test.java` for unit tests and `*IT.java` for integration tests is a good practice. Then configure each plugin’s `include` and `exclude` appropriately.
3.  **Inconsistent Lifecycle Bindings**: Problems arise when you don't have `integration-test` bound to the correct phases in Failsafe or you haven't created custom phases, such as 'pre-integration-test' if you need to set up your environment, using the `maven-antrun-plugin`. The integration test phase typically runs after the `package` phase and before the `verify` phase. Ensure your plugin is executed during these phases to prevent deployment and testing order issues.
4.  **Resource Exhaustion**:  Parallel test execution can improve speed but can also lead to resource contention if `threadCount` and `forkCount` are set too high. Monitor your system resources, and adjust these parameters accordingly. Additionally, set `reuseForks` to `false` if you find that your tests are not isolated enough between runs.

**A More Advanced Example – Setup and Teardown**

Sometimes, you need to perform some setup before integration tests and cleanup afterward. The `maven-antrun-plugin` in conjunction with Failsafe is handy for this:

```xml
<plugin>
    <artifactId>maven-antrun-plugin</artifactId>
    <version>3.1.0</version>
    <executions>
        <execution>
            <id>integration-test-setup</id>
            <phase>pre-integration-test</phase>
            <goals>
                <goal>run</goal>
            </goals>
            <configuration>
                <target>
                   <echo message="Setting up integration test environment..." />
                     <!-- Example: Start your app server here using Ant tasks -->
                </target>
            </configuration>
        </execution>
        <execution>
            <id>integration-test-teardown</id>
            <phase>post-integration-test</phase>
            <goals>
                <goal>run</goal>
            </goals>
            <configuration>
                <target>
                     <echo message="Tearing down integration test environment..." />
                    <!-- Example: Stop your app server here using Ant tasks -->
                </target>
            </configuration>
        </execution>
    </executions>
</plugin>
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
             <configuration>
                    <includes>
                        <include>**/*IT.java</include>
                    </includes>
                      <excludes>
                        <exclude>**/*Test.java</exclude>
                    </excludes>
                 </configuration>
        </execution>
    </executions>
</plugin>
```

This example utilizes `maven-antrun-plugin` to perform specific setup tasks using Ant within the `pre-integration-test` phase and to tear down resources within the `post-integration-test` phase. It ensures that setup and teardown occur surrounding Failsafe's `integration-test` and `verify` phases.

**Recommended Reading**

For a deeper understanding, I highly recommend:

*   **“Maven: The Complete Reference”** by Sonatype. This is an excellent resource covering all things Maven, including plugins and lifecycle management.
*   The **official Apache Maven documentation** for the Surefire and Failsafe plugins. These are the source of truth and should be your primary reference.
*    **"Effective Java"** by Joshua Bloch. While not Maven-specific, this book contains crucial guidance on writing effective tests, which is valuable when working with test runners like Surefire and Failsafe.

In conclusion, effectively configuring Maven, Surefire, and Failsafe involves a clear understanding of the Maven lifecycle, the specific roles of each plugin, and careful planning of test execution sequences. Start with clear naming conventions, fine-tune your include/exclude filters, and don't be afraid to explore the wealth of configuration options available in these plugins.
