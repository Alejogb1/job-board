---
title: "How to exclude @Setup methods from JMH profiling?"
date: "2025-01-30"
id: "how-to-exclude-setup-methods-from-jmh-profiling"
---
JMH, the Java Microbenchmark Harness, by default executes methods annotated with `@Setup` for each benchmark invocation, contributing to execution overhead that isn't representative of the code you're actually profiling.  This setup phase can skew results significantly, particularly when setup logic is complex or time-consuming relative to the benchmarked code. Excluding `@Setup` methods from the profiling output, therefore, is critical for obtaining accurate performance measurements of the target code snippet.

The core challenge arises from JMH's lifecycle management, which automatically includes setup and teardown phases. The solution isn't to *prevent* setup execution entirely; that would render many benchmarks invalid. Instead, we aim to isolate the performance impact of the benchmarked method itself by focusing JMH's profilers specifically on the `@Benchmark` annotated methods, effectively excluding the setup from the profiler's analysis. I encountered this issue frequently during my work optimizing a large data processing pipeline, where initial data loading in `@Setup` methods masked the performance characteristics of the core calculation methods.

The mechanism for achieving this isolation lies within JMH's profiler configuration. The configuration isn't exposed through annotation properties directly, necessitating the use of a separate JMH API, typically done when configuring the benchmark's `Runner`. JMH’s command-line arguments also allow for similar configurability.

To understand this practically, let's examine three code examples, each demonstrating different aspects of the problem and the solution.

**Example 1: Naive Benchmark (Problem)**

```java
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.runner.*;
import org.openjdk.jmh.runner.options.*;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;


@State(Scope.Thread)
public class BenchmarkSetup {

    private List<Integer> data;

    @Setup
    public void setup() {
        data = new ArrayList<>();
        for (int i = 0; i < 1000; i++) {
            data.add(i);
        }
    }

    @Benchmark
    @BenchmarkMode(Mode.AverageTime)
    @OutputTimeUnit(TimeUnit.NANOSECONDS)
    public int calculateSum() {
        int sum = 0;
        for(int value: data){
            sum += value;
        }
        return sum;
    }

    public static void main(String[] args) throws RunnerException {
        Options opt = new OptionsBuilder()
                .include(BenchmarkSetup.class.getSimpleName())
                .forks(1)
                .warmupIterations(5)
                .measurementIterations(5)
                .build();
        new Runner(opt).run();
    }

}
```

This first example illustrates a typical benchmark setup. The `setup()` method, annotated with `@Setup`, initializes a list of integers. The `calculateSum()` method, annotated with `@Benchmark`, then calculates their sum. When executed, profilers will analyze *both* methods. This is problematic because the setup method's execution contributes to the overall reported time, skewing results particularly when data generation becomes more complex. Here, the relatively simple `ArrayList` population makes this effect subtle, but in real-world scenarios with complex object construction, the impact can be substantial.

**Example 2: Profiler Configuration (Solution - Direct API)**

```java
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.profile.GCProfiler;
import org.openjdk.jmh.runner.*;
import org.openjdk.jmh.runner.options.*;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

@State(Scope.Thread)
public class BenchmarkSetupProfile {

    private List<Integer> data;

    @Setup
    public void setup() {
        data = new ArrayList<>();
        for (int i = 0; i < 1000; i++) {
            data.add(i);
        }
    }

    @Benchmark
    @BenchmarkMode(Mode.AverageTime)
    @OutputTimeUnit(TimeUnit.NANOSECONDS)
    public int calculateSum() {
        int sum = 0;
        for(int value: data){
            sum += value;
        }
        return sum;
    }

    public static void main(String[] args) throws RunnerException {
        Options opt = new OptionsBuilder()
                .include(BenchmarkSetupProfile.class.getSimpleName())
                .forks(1)
                .warmupIterations(5)
                .measurementIterations(5)
                .addProfiler(GCProfiler.class) // Enable GC profiler
                .build();

        new Runner(opt).run();
    }
}
```

This second example introduces a critical modification. The `addProfiler(GCProfiler.class)` line activates the GC profiler directly. Other profilers work similarly. Importantly, by default, JMH profilers only analyze `@Benchmark` annotated methods; `@Setup` is implicitly *excluded* from the profiling output. While this example showcases only the GC profiler, the principle is universal across various JMH profilers.  The reported time measurements will now be primarily influenced by the `calculateSum()` method.

The use of a particular profiler, even if not directly analyzing time but resources (like `GCProfiler`), forces JMH to focus profiling efforts exclusively on the benchmarked methods. This mechanism isn’t explicitly documented but is observable through the profile reports generated. In practice, this is the most common and effective way to focus profiling.

**Example 3: Profiler Configuration (Solution - CommandLine)**

```java
import org.openjdk.jmh.annotations.*;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

@State(Scope.Thread)
public class BenchmarkSetupProfileCommandLine {

    private List<Integer> data;

    @Setup
    public void setup() {
        data = new ArrayList<>();
        for (int i = 0; i < 1000; i++) {
            data.add(i);
        }
    }

    @Benchmark
    @BenchmarkMode(Mode.AverageTime)
    @OutputTimeUnit(TimeUnit.NANOSECONDS)
    public int calculateSum() {
        int sum = 0;
        for(int value: data){
            sum += value;
        }
        return sum;
    }

     public static void main(String[] args) {

     } // main method is intentionally empty. Run this class from command line
}
```

This third example demonstrates how to exclude the setup phase from profiling when invoking JMH from the command line. The `main` method is intentionally empty here because the JMH configuration occurs externally via the command-line.

To run this benchmark and exclude `@Setup` from profiling, use a command similar to the following:

```bash
java -jar target/benchmarks.jar  BenchmarkSetupProfileCommandLine -prof gc -f 1 -wi 5 -i 5
```

The `-prof gc` option activates the GC profiler, effectively excluding setup methods from timing. The `-f`, `-wi` and `-i` flags control the number of forks, warmup iterations and iterations, respectively. Other JMH profilers can be activated similarly via `-prof <profiler_name>`, including `perfasm`, `perfnorm`, `xperf`, and `flightrecorder`. I often used the `-prof perfasm` to analyze low-level assembly in critical sections of code. This example showcases how the same profiling configurations are available via the command-line options. This approach enables automation of benchmark execution and integrates well with continuous integration environments. The ability to configure JMH both programmatically, via its API, and via command-line options provides immense flexibility for benchmark development and execution.

In summary, while JMH's default behaviour includes setup phases in profiling, focusing on accurate profiling of benchmarked methods is easily achieved by configuring profilers via JMH's `Runner` API or via command-line arguments.  The key is to understand that JMH profilers inherently target benchmarked methods when explicitly invoked, thus implicitly excluding `@Setup` methods. I found the command-line flexibility particularly valuable for repeatable testing environments.

To deepen understanding of JMH and its profiling mechanisms, I would recommend reviewing the JMH documentation available on the OpenJDK project, the official JMH samples (often available through GitHub repositories) and articles available on performance tuning using JMH. The knowledge of various JMH profilers like `perfasm` and `FlightRecorder` is also beneficial for detailed performance analysis.
