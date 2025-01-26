---
title: "Why can't AsyncProfiler load in a JMH benchmark?"
date: "2025-01-26"
id: "why-cant-asyncprofiler-load-in-a-jmh-benchmark"
---

Async Profiler, a powerful sampling profiler for Java and other languages, often encounters difficulties when directly integrated within a Java Microbenchmark Harness (JMH) execution environment. The fundamental challenge stems from the way JMH manipulates the target process and the timing sensitivities of Async Profiler. JMH benchmarks, by design, require precise and controlled measurements of short-duration code segments. The typical JVM instrumentation mechanisms employed by profilers, including Async Profiler, can introduce overhead that invalidates these measurements. This interference isn't necessarily a bug in either tool but rather a conflict in their respective operational paradigms.

The crux of the problem lies in the fact that JMH typically forks a new JVM process to execute each benchmark run. This is done to isolate the benchmarking code from the JMH harness itself and to mitigate the influence of garbage collection and other JVM activities on benchmark results. The isolation is vital for generating statistically valid measurements, but this behavior poses a problem for profilers like Async Profiler. Async Profiler operates by attaching itself to an already running JVM using Java Agent mechanisms. When a new JVM process is forked by JMH, any existing Async Profiler instance is not automatically inherited or re-attached to the forked JVM process.

Furthermore, direct attachment using the `attach` functionality within Async Profiler might not be feasible or reliable within JMH's lifecycle. Attempting to dynamically attach to a JVM instance after it has been forked and is executing its benchmark can be problematic due to timing issues. The benchmark's code runs within a constrained timeframe, and any delay caused by attaching to the process might distort the measurement and create an inconsistent environment. Furthermore, if the forked process completes before the attach operation is fully finished, the profiling data will be incomplete or non-existent.

Finally, even if attachment was technically achievable within the time constraints, the profiling data generated might be corrupted by the initial overhead of loading and initializing Async Profiler within a benchmark, potentially skewing results. Benchmarks generally target the steady-state performance of a method or a code snippet, not the startup costs associated with a tool. Therefore, any profiler action that introduces a substantial initial overhead would defeat the purpose of the benchmark.

To address this incompatibility, profiling a JMH benchmark is generally done by profiling the parent JVM instance running the JMH harness, not the forked processes running the actual benchmarks. However, the data obtained this way describes the JMH framework, not the benchmark itself. This is largely because the benchmarked code is running within the forked JVM, and it's not under the direct scrutiny of the parent JVM's profiler. Another potential solution involves setting up the Async Profiler to run via JVMTI agent when a specific JVM starts, essentially forcing the agent to load during the benchmarked JVM initialization phase. This requires careful configuration, though, and may still lead to less than optimal results depending on the benchmark itself and the JVM's behavior. Below, I will illustrate code examples of failed attempts and successful alternative approaches to achieve profiling, with the understanding that directly attaching a running profiler to the benchmark JVM at runtime is not ideal.

**Code Example 1: Attempting Direct Attachment (Typically Fails)**

```java
// Intentionally simplified and representative code
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.concurrent.TimeUnit;
import java.io.File;


@State(Scope.Benchmark)
public class MyBenchmark {

    @Benchmark
    @Fork(value = 1, jvmArgsAppend = {
        "-XX:+UnlockDiagnosticVMOptions",
        "-XX:+DebugNonSafepoints",
        "-XX:+DisableAttachMechanism" // For test
    })
    @BenchmarkMode(Mode.AverageTime)
    @OutputTimeUnit(TimeUnit.NANOSECONDS)
    public long testMethod() {
        long sum = 0;
        for (int i=0; i<100; ++i){
            sum += i * i;
        }
       return sum;
    }


    public static void main(String[] args) throws RunnerException {
        Options options = new OptionsBuilder()
                .include(MyBenchmark.class.getSimpleName())
                .forks(1)
                .warmupIterations(0) // For brevity
                .measurementIterations(1)
                .build();

        // Below Code is incorrect and will not work inside forked JVM.
        // AsyncProfiler profiler = new AsyncProfiler("/path/to/async-profiler/libasyncProfiler.so");
        // profiler.start("testMethod", 1000); // Attempted at runtime, WRONG!

        new Runner(options).run();

       // profiler.stop(); // Attempted at runtime
        // profiler.dumpCollapsed("/tmp/test.collapsed");
    }
}
```

**Commentary:** This first example demonstrates what would commonly be attempted – instantiating `AsyncProfiler` directly within the main method of the benchmark. This, however, is flawed. The profiler instantiation and attempts to start/stop and dump are performed in the *parent* JVM, not the forked JVM in which the benchmark executes. This code does not profile the benchmarked method, it profiles parts of the JMH framework itself, and even then, the profiler would likely fail to operate properly without a proper attachment process. We have also deliberately disabled the attach mechanism in the forked JVM to explicitly demonstrate why an attempted attachment at runtime, will fail.

**Code Example 2: Profiling the Parent JMH Process (Valid, but Not the Benchmark)**

```java
// Intentionally simplified and representative code
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;


import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.concurrent.TimeUnit;
import java.io.File;


@State(Scope.Benchmark)
public class MyBenchmark {

    @Benchmark
    @Fork(1)
    @BenchmarkMode(Mode.AverageTime)
    @OutputTimeUnit(TimeUnit.NANOSECONDS)
    public long testMethod() {
        long sum = 0;
        for (int i=0; i<100; ++i){
            sum += i * i;
        }
       return sum;
    }

    public static void main(String[] args) throws RunnerException {
        Options options = new OptionsBuilder()
                .include(MyBenchmark.class.getSimpleName())
                .forks(1)
                .warmupIterations(0)
                .measurementIterations(1)
                .build();


        // The JVM that starts this program is the one running the JMH
        // and can be profiled using the async-profiler's command-line interface.

        // This example shows how *outside* the scope of the program, using async-profiler on the
        // JVM pid can be used to get data on the JMH execution. This is
        // NOT the same as profiling testMethod()

        // For example, using Linux:
        //   $ ./async-profiler.sh -e cpu -f /tmp/jmh.collapsed -d 60  <PID_of_this_JVM>


        new Runner(options).run();
    }
}

```

**Commentary:** This second example demonstrates a correct way to leverage Async Profiler in conjunction with JMH, but it highlights a crucial point:  The profiler does *not* directly profile `testMethod()`. Instead, the profiler, started via command line *outside* the JMH process, will collect performance data from the JVM process running the JMH harness. This is helpful to debug performance of the JMH framework itself, not the benchmarked code. The crucial aspect here is that the profiling is performed on the process that launches the benchmark JVM (the parent), which is completely distinct from the JVM executing the benchmarked method (`testMethod()`). The profiling command is provided as a comment and runs outside of the Java process.

**Code Example 3: Pre-configured JVMTI Agent (Potential Solution, Complex Setup)**

This example cannot show a functional code example because a JVM agent requires a native library and configuration, but I will outline conceptually how this can be configured.

To profile the forked JMH JVM processes, you could use the `-agentpath` option to load the Async Profiler agent at startup. This requires setting a JVM argument in the benchmark configuration (e.g., using `@Fork(jvmArgsAppend = { ... })`). The `-agentpath` would be used to point to the location of the Async Profiler native library (`libasyncProfiler.so` on Linux/macOS or `asyncProfiler.dll` on Windows). The agent would then be configured via an additional property to start a profiling session on the benchmark method. Here’s a representation of the relevant JMH config:

```java
@Fork(value = 1, jvmArgsAppend = {
    "-agentpath:/path/to/async-profiler/libasyncProfiler.so=start,event=cpu,file=/tmp/benchmark.collapsed,interval=100000,jfr=false,framebuf=1048576"
})
```

**Commentary:**  This approach attempts to leverage the fact that we can provide command line arguments to the forked JVM. This allows the agent to be started before the benchmark begins, potentially capturing a more accurate representation of the performance data. However, this method introduces complexities in the setup, and the agent starts with the forked JVM, potentially including startup costs that might influence the benchmark results if not carefully managed. Moreover, this approach needs a more careful configuration to ensure that the profiler is stopped at the end of the benchmark. This method also requires the JVM to be started with specific options, which can be complicated for end users. The profiler should be configured to generate data in a specific location, and the data should be retrieved after the JMH benchmark is finished, requiring extra care and setup.

**Resource Recommendations**

To gain deeper understanding of JVM performance analysis and profiling techniques, several resources prove valuable. Exploring the official documentation of the JVM itself offers insights into instrumentation mechanisms and how profilers typically interact with the runtime environment. Detailed information about the Java Virtual Machine Tool Interface (JVMTI) is crucial for advanced techniques such as agent-based profiling. Moreover, resources dedicated to performance analysis methodology, such as texts and papers on microbenchmarking techniques, can highlight potential pitfalls and provide a sound foundation for interpreting results. Understanding JMH and its forking behavior is also essential for working within its context, and the project's documentation is a great place to start. Finally, studying documentation related to other profiling tools like JProfiler or YourKit can help understand the commonalities and differences between different approaches and their applicability in different scenarios. While specific vendor recommendations may change over time, these generic recommendations provide a solid foundation in the areas of performance and JVM specifics.
