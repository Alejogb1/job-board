---
title: "How can a running Java application be profiled from the command line?"
date: "2025-01-30"
id: "how-can-a-running-java-application-be-profiled"
---
Java application profiling from the command line provides crucial insights into runtime performance without requiring IDE integration. This capability is invaluable for production environments and continuous integration pipelines where interactive debugging is often infeasible. Profiling in this manner typically involves attaching a Java Virtual Machine (JVM) agent to a running process, capturing performance data, and then analyzing that data using external tools. I have employed this technique extensively during my time supporting backend systems for a high-throughput transactional platform, where transient performance degradations could have significant business impact.

The primary mechanism for command-line profiling involves the Java Virtual Machine Tools Interface (JVMTI). JVMTI is a native programming interface that allows monitoring and control of the JVM. Profilers utilize JVMTI to instrument the running application, gathering metrics like CPU usage, memory allocation, and thread activity. These profilers are implemented as Java agents, which are dynamically loaded into the JVM when it starts or is attached to a running process. The process of command-line profiling therefore usually entails three stages: agent selection, agent attachment, and data analysis.

**Agent Selection:** Numerous profiling agents are available, each tailored to specific needs. Some agents, like those used with the `jvisualvm` tool, provide a comprehensive view of application behavior but are resource-intensive and generally unsuitable for production. More lightweight agents focus on specific aspects like CPU time sampling or memory allocation tracing. Selecting the appropriate agent for your application requires understanding what information you seek. For example, if the goal is to identify CPU hotspots, a sampling profiler will be most efficient. Conversely, pinpointing memory leaks or excessive garbage collection might require a memory allocation tracker. My preference for production profiling leans towards sampling profilers initially to identify problem areas before investigating them with deeper, more focused probes.

**Agent Attachment:** Java provides the `jattach` utility for attaching agents to running JVM processes. This utility requires the process ID (PID) of the target Java application and the path to the agent's JAR file. The basic syntax is `jattach <PID> load <agent_path>=<agent_options>`. The PID can be obtained using operating system utilities like `jps` (Java Process Status) on Unix-like systems or through the task manager on Windows. Agent options allow you to configure the agent's behavior, such as the output file or sampling frequency. After attachment, the agent runs within the target JVM, accumulating performance data based on its configured parameters. This process avoids restarting the target application, a crucial requirement for most production environments. The agent can often be detached after data capture, minimizing any continued overhead on the application.

**Data Analysis:** The agent typically generates a data file in a specific format. This file needs to be analyzed using a tool designed to interpret that format. The analysis is usually done offline. For instance, if a sampling profiler generates a `.jfr` file (Java Flight Recorder file), you need a tool like Java Mission Control or `jfrprint` to examine the profiled data. These tools present the data visually, allowing you to navigate call stacks, observe thread behavior, and quantify resource usage. I’ve found that understanding the data file format, and selecting appropriate tools, are essential for drawing meaningful conclusions from profiled data.

Let’s examine this process with three examples:

**Example 1: Using Async Profiler for CPU Time Sampling**

The Async Profiler is a low-overhead CPU profiler that is excellent for production environments. It employs native code for minimal performance impact. I've used it frequently for performance-sensitive backend services.

```bash
# 1. Identify the Java Process ID
jps -v | grep MyApplication

# Output (example):
# 12345 MyApplication -Dmy.config.prop=value

# 2. Load the Async Profiler agent, specifying output file
jattach 12345 load /path/to/async-profiler.so=output=my_profile.jfr,duration=60s

# 3. After 60 seconds, detach the agent
jattach 12345 stop

# 4. Analyze the resulting jfr file
jfrprint my_profile.jfr #Or import into Java Mission Control
```

This example demonstrates attaching the `async-profiler.so` library to the Java process with PID 12345. It specifies that the output should be a Java Flight Recorder file named `my_profile.jfr` and that profiling should last for 60 seconds.  The `jfrprint` tool or Java Mission Control are used to analyze the resulting `.jfr` file. The `stop` command detaches the agent.

**Example 2: Profiling with the Built-in Java Flight Recorder (JFR) via jcmd**

JFR is part of the JDK itself. While typically configured at JVM startup, it can also be enabled and controlled at runtime using `jcmd`. I have found `jcmd` useful for targeted ad-hoc data collection.

```bash
# 1. Identify the Java Process ID
jps -v | grep MyApplication

# Output (example):
# 67890 MyApplication -Dmy.config.prop=value

# 2. Start recording using jcmd
jcmd 67890 JFR.start name=MyRecording duration=60s filename=jfr_recording.jfr

# 3. Stop recording using jcmd
jcmd 67890 JFR.dump name=MyRecording filename=jfr_recording_dump.jfr

# 4. Analyze the jfr file
jfrprint jfr_recording_dump.jfr #Or import into Java Mission Control

```

This example shows how to use `jcmd` to activate a JFR recording on a process with the PID 67890. A named recording `MyRecording` is started with a duration of 60 seconds and its output is saved as `jfr_recording.jfr`. A dump is then created. The final `.jfr` file is ready for analysis. The advantage of this approach is that JFR is directly integrated into the JDK, and thus is readily available.

**Example 3: Using Java Agents for Memory Allocation Tracking (Hypothetical Agent)**

While not a standard tool, this example illustrates how a custom agent, or a third-party agent, might function. Suppose there exists a hypothetical agent `memory_tracker.jar` that collects memory allocation information.

```bash
# 1. Identify the Java Process ID
jps -v | grep MyApplication

# Output (example):
# 98765 MyApplication -Dmy.config.prop=value

# 2. Attach the memory tracking agent and configure output path
jattach 98765 load /path/to/memory_tracker.jar=output=memory_allocation.log,interval=100ms

# 3. Detach the agent after a desired interval
jattach 98765 stop

# 4. Analyze the resulting log file (using a custom script, for example)
# cat memory_allocation.log | python analyze_memory.py
```

This demonstrates how you would attach a hypothetical `memory_tracker.jar` to the JVM. It takes parameters such as an output file location `memory_allocation.log` and a sampling `interval` to govern how often memory allocation details are captured. After gathering data, the agent is detached and the resulting output, in this case a text log file, would need further analysis through custom scripting or appropriate tooling specific to that agent.

In summary, command-line profiling offers a robust method for understanding Java application performance without requiring complex IDE setups or application restarts. Employing tools like Async Profiler, `jcmd` for JFR, and potentially other custom or third-party agents, provides flexibility and control over the profiling process. Crucially, careful selection of profiling agents and analysis tools is necessary to make accurate judgements about application behavior. Understanding the output format and selecting the right tool are key elements for extracting value from the profiled data. For learning more, I recommend investigating documentation for `jattach`, `jps`, `jcmd`, and any agent-specific documentation, as well as textbooks or courses covering Java performance monitoring. The Oracle documentation for the Java Virtual Machine Tool Interface and Java Flight Recorder is also very informative.
