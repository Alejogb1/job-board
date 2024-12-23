---
title: "How can parallel testing be achieved using the Maven Failsafe plugin?"
date: "2024-12-23"
id: "how-can-parallel-testing-be-achieved-using-the-maven-failsafe-plugin"
---

Alright, let's unpack parallel testing with Maven Failsafe. I've spent a fair chunk of time optimizing test suites, and believe me, getting parallel execution working correctly with Failsafe can significantly reduce your build times, particularly on larger projects. It's not just about slapping a switch; there are nuances to consider for a stable and efficient parallel setup.

First off, Failsafe, unlike the Surefire plugin that's primarily used for unit tests, is geared towards integration tests. These are often more time-consuming, making parallel execution even more crucial. By default, Failsafe runs tests sequentially, one after another. To enable parallelism, we need to configure the plugin using the `configuration` tag within the `<plugin>` block of our `pom.xml`.

The core setting that controls parallelism is the `parallel` parameter. You can set it to several values, most commonly: `classes`, `methods`, or `suites`. This dictates how the parallel execution occurs – either by class, by individual test method, or by test suites (in the case of TestNG). I've found that `methods` usually offers the best granularity for performance gains, as it maximizes the utilization of available resources, particularly if your methods are relatively short.

A common mistake I've seen is not pairing the `parallel` parameter with the `threadCount` parameter. Specifying only `parallel` won't magically optimize your tests; it simply enables the parallel mode. `threadCount` actually controls the number of threads Failsafe will use. I usually determine this based on the number of cores available on the build machine, leaving one or two threads free to avoid resource contention. I’ve found that the optimal thread count rarely matches the total number of CPU cores because overhead can actually degrade performance. Experimentation is key, but a sensible starting point is usually around `(number of cores - 2)` to `number of cores`.

Let’s break down how this works with some code examples.

```xml
<!-- Example 1: Basic parallel testing by class, common for initial setups. -->
<plugin>
  <groupId>org.apache.maven.plugins</groupId>
  <artifactId>maven-failsafe-plugin</artifactId>
  <version>3.2.3</version>
  <configuration>
    <parallel>classes</parallel>
    <threadCount>4</threadCount>
    <includes>
        <include>**/*IT.java</include>
    </includes>
  </configuration>
  <executions>
    <execution>
      <goals>
        <goal>integration-test</goal>
        <goal>verify</goal>
      </goals>
    </execution>
  </executions>
</plugin>
```

In this first example, we've configured Failsafe to execute tests in parallel at the *class* level using four threads. The includes tag specifies the naming convention for tests that Failsafe will execute, commonly with `IT` suffix. This configuration is simple, and it works, but it may not fully utilize resources if your classes have few or small tests.

Now, let's look at parallel execution by method:

```xml
<!-- Example 2: Parallel testing by method, offering better granularity for CPU usage. -->
<plugin>
  <groupId>org.apache.maven.plugins</groupId>
  <artifactId>maven-failsafe-plugin</artifactId>
  <version>3.2.3</version>
  <configuration>
    <parallel>methods</parallel>
    <threadCount>8</threadCount>
    <perCoreThreadCount>true</perCoreThreadCount>
       <includes>
        <include>**/*IT.java</include>
       </includes>
     </configuration>
  <executions>
    <execution>
      <goals>
        <goal>integration-test</goal>
        <goal>verify</goal>
      </goals>
    </execution>
  </executions>
</plugin>
```

Here, we’ve changed the `parallel` parameter to `methods`, allowing Failsafe to run each test method in parallel, up to a maximum of eight threads. Additionally, I've included `<perCoreThreadCount>true</perCoreThreadCount>`. This crucial addition instructs Failsafe to use the system's core count when determining the thread count. Without it, the explicit `threadCount` overrides any dynamic configuration based on the actual hardware. This addition can help ensure the thread count is more optimized for the executing system. I remember a time where not having this caused a massive slowdown because I had manually over-configured the threads on a system with limited resources.

Now, let's consider a slightly more advanced scenario involving thread-safety and resource management. You often see issues with shared resources when running tests in parallel. For example, if your integration tests all interact with the same database or file system location, parallel execution might cause race conditions and unexpected test failures. There are a few strategies, but sometimes you can't get away from needing to isolate resources by test. In that situation, TestNG’s ability to work with test suites can be used to our advantage, even when we aren't using the full suite functionality:

```xml
<!-- Example 3: Suite level parallelism in combination with TestNG -->
<plugin>
    <groupId>org.apache.maven.plugins</groupId>
    <artifactId>maven-failsafe-plugin</artifactId>
    <version>3.2.3</version>
    <configuration>
      <parallel>suites</parallel>
      <threadCount>3</threadCount>
      <suiteXmlFiles>
            <suiteXmlFile>src/test/resources/testng.xml</suiteXmlFile>
        </suiteXmlFiles>
      <includes>
         <include>**/*IT.java</include>
      </includes>
    </configuration>
    <executions>
      <execution>
        <goals>
          <goal>integration-test</goal>
          <goal>verify</goal>
        </goals>
      </execution>
    </executions>
  </plugin>
```

And your `testng.xml` might contain something like:

```xml
<!DOCTYPE suite SYSTEM "https://testng.org/testng-1.0.dtd">
<suite name="Integration Test Suite" parallel="false">
    <test name="Isolated Resource Test 1">
        <classes>
            <class name="com.example.IsolatedTest1IT"/>
        </classes>
    </test>

    <test name="Isolated Resource Test 2">
        <classes>
             <class name="com.example.IsolatedTest2IT"/>
        </classes>
    </test>

    <test name="Isolated Resource Test 3">
      <classes>
            <class name="com.example.IsolatedTest3IT"/>
      </classes>
    </test>
</suite>

```

In this setup, Failsafe is configured to use the TestNG test suite mechanism. While the `<parallel>` setting specifies `suites`, we keep parallel set to `false` inside the testng configuration file, so test methods within a suite will still be run serially (which is desirable when the tests need isolated environments, and each test needs to be in an isolated environment of its own). However, by setting the `threadCount` to three, three test suites (defined by the `<test>` tags in the xml) will be run in parallel. The advantage of this setup is that we can configure suites in a way that they use different resources (e.g. databases), so we can have multiple tests using the same database, but in isolation from other parallel executions. This method is particularly useful when integration tests have high setup/teardown costs and can't run truly in parallel using the `methods` option. I had one project that involved a rather large database migration for every test, and using suites with isolated databases was the only way we could speed things up.

A few words of advice I've picked up over the years: Always ensure your tests are thread-safe, or they have resource requirements that allow them to be run in parallel. Sometimes, the problem is not the testing configuration, but the design of the test itself. Second, when making these changes to the build configuration, do it in small increments and measure performance improvements after each step. Jumping straight to the highest thread count could cause problems. You should see performance gains gradually, not suddenly. Lastly, monitor your resources during parallel test runs, specifically CPU usage, memory, and disk i/o. It's important to confirm that the performance benefits of parallelism are not negated by resource bottlenecks.

For further reading, I'd recommend checking out "Effective Java" by Joshua Bloch for general best practices on concurrency, and the official Maven Failsafe documentation for the most detailed understanding of its configuration options. If you’re using TestNG as I have in the last example, you should also go through its official documentation for its rich set of parallel testing capabilities. Pay special attention to the details of concurrent execution with TestNG. Also, the book "Java Concurrency in Practice" by Brian Goetz is an extremely detailed overview of java concurrency concepts that can help you understand the implications of parallel testing and how to avoid common concurrency related bugs.

Parallelizing your test suite with Failsafe is a valuable tool. It's not just about speed but about making your development cycles more efficient. Just remember that it’s a process of careful configuration and monitoring rather than a “set and forget” operation.
