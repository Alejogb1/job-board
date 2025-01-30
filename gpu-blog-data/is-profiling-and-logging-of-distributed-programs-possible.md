---
title: "Is profiling and logging of distributed programs possible in Elixir?"
date: "2025-01-30"
id: "is-profiling-and-logging-of-distributed-programs-possible"
---
Elixir, leveraging the Erlang VM (BEAM), inherently presents unique opportunities and challenges for profiling and logging distributed applications compared to more traditional single-server environments. The concurrency model, based on lightweight processes communicating via message passing, necessitates tooling and techniques that understand and accommodate this distributed nature. From my experience scaling Elixir applications within a microservices architecture at TechCorp Solutions, I’ve found that effective profiling and logging of such programs is not only possible but crucial for maintaining performance and diagnosing complex issues.

Profiling in Elixir, especially when dealing with distributed processes, involves several complementary approaches. One key distinction is the granularity at which profiling data is collected. For instance, traditional CPU profiling tools focused on hot code paths within a single process are useful, but only represent a fraction of the overall system behavior in a distributed context. I’ve encountered scenarios where one process was efficiently executing, yet the system was slow due to network latency or message queue backlogs. Therefore, we must consider profiling process interactions, message passing overhead, and network performance alongside individual process execution time.

A common technique I've used is leveraging the `:observer` application included with Erlang/OTP. While primarily designed as a real-time observability tool, it offers valuable insights into individual node performance. The “Processes” tab, for example, displays process activity including message queue length, which is an excellent indicator of potential bottlenecks within a specific node. Furthermore, the “Applications” tab provides data about resources being consumed by each active application which is particularly important when understanding if a given application is behaving as expected. While the observer is helpful, it doesn't directly profile across nodes. That's where custom instrumentation and logging comes in handy.

For a more granular, distributed profile, I’ve found it effective to introduce custom performance monitoring within specific modules. This means adding timing calculations around critical code blocks and tracking the resulting data, ideally with unique identifiers for individual requests that span multiple processes and nodes. This approach offers a fine-grained view of function execution and message passing durations across the distributed system. Below is an example showing this method:

```elixir
defmodule MyModule do
  def my_function(arg) do
    start_time = System.monotonic_time(:nanosecond)
    result =
      :timer.tc(fn ->
        # do some work that could be costly
        do_computation(arg)
      end)
    end_time = System.monotonic_time(:nanosecond)
    duration = (end_time - start_time)
    log_performance("my_function", duration, arg, result)
    result
  end
  defp do_computation(arg) do
    # some calculation
  end
  defp log_performance(fun_name, duration, arg, result) do
    Logger.info(
      "Performance: #{fun_name}, Duration: #{duration}ns, Arg: #{inspect(arg)}, Result: #{inspect(result)}"
    )
  end
end
```

In this snippet, we’re explicitly measuring the execution time using `:timer.tc` and `System.monotonic_time` in nanoseconds to get precise timings. We log function name, execution duration, arguments passed, and returned result via `Logger.info`, which we can later aggregate and analyze to identify bottlenecks. The crucial part of this example is the custom logging function. I would typically pass more metadata through this log statement like a trace ID that would identify this specific action across multiple applications in the distributed system. This example emphasizes that we aren't reliant on any external profiling tool but instead instrument the code ourselves for specific needs. This flexibility is a powerful asset when debugging in a distributed system.

When instrumenting, we also need to pay particular attention to messages sent between processes. Message queue sizes can become problematic as they grow, indicating a process unable to keep up with its workload. In the distributed context, the same applies to network traffic. To get more granular information about messages, I usually make use of custom wrapper functions that add logging around any inter process or inter node communication.

```elixir
defmodule MessageHelper do
  def send(pid, message) do
    start_time = System.monotonic_time(:nanosecond)
    send(pid, message)
    end_time = System.monotonic_time(:nanosecond)
    duration = end_time - start_time
    log_message_send(pid, message, duration)
  end
  defp log_message_send(pid, message, duration) do
    Logger.info(
      "Message Sent: To PID: #{inspect(pid)}, Message: #{inspect(message)}, Duration: #{duration}ns"
    )
  end
  def receive(timeout \\ :infinity) do
    receive do
      message ->
      start_time = System.monotonic_time(:nanosecond)
      {_, result} =
        :timer.tc(fn ->
          message
        end)
      end_time = System.monotonic_time(:nanosecond)
      duration = end_time - start_time
      log_message_received(message, duration)
      result
    after
      timeout ->
        nil
    end
  end
  defp log_message_received(message, duration) do
    Logger.info("Message Received: Message: #{inspect(message)}, Duration: #{duration}ns")
  end
end

```

Here we wrap the built-in `send` and `receive` functions with our own helpers that log the message, the duration and the PID on the other end of the `send`. This pattern has been useful to identify where bottlenecks occur within message passing and it can highlight issues with serialization/deserialization, especially with complex data structures. Additionally, tracking the time it takes to receive a message or process a message is also helpful for performance analysis. The `receive` function is wrapped in a custom function that logs the message and time taken to receive that message. The `after` clause is important to add here, because without it, the process would be blocked indefinitely waiting for a message that might not come. In this case, the process can move on and other activities might occur instead. This illustrates how adding a little instrumentation can lead to valuable insight without relying on third-party tools.

Beyond timing operations, capturing system metrics is critical in a distributed application. This includes resource utilization (CPU, memory, network), as well as application-specific metrics such as queue lengths, number of active processes, or database connection pool stats. Tools that are capable of pulling these metrics together from all participating nodes are essential for holistic system monitoring. I've utilized libraries that expose metrics through a standardized interface, which simplifies integration with various monitoring platforms and allows for centralized dashboarding and alerting.

Effective logging of distributed programs requires careful planning and a consistent approach. I employ structured logging with context-rich data, including trace IDs that allow me to follow a single request's journey across different nodes and processes. This means incorporating not just the log message itself but also the process ID, node name, timestamp, and any relevant metadata. Centralized log aggregation and analysis are crucial, allowing you to search, filter, and correlate log entries from various sources to reconstruct the entire system's activity at any point in time.

```elixir
defmodule Logging do
  @doc """
  Logs message with trace id, application name, node name, and process id.
  """
  def log(level, message, trace_id \\ nil) do
    context = %{
      trace_id: trace_id,
      app: :my_app,
      node: Node.self(),
      pid: self()
    }
    Logger.log(level, "#{message} - #{inspect(context)}")
  end
end
```

This example demonstrates the use of a logging helper that adds additional context for each log statement. This technique is quite useful in the process of debugging in distributed systems because of the fact that each log statement has a trace ID as well as the node and process from which it originated. With the ability to search for all log statements with a given trace ID, debugging becomes substantially more tractable. This structured approach to logging is critical for effective debugging in distributed systems.

In conclusion, profiling and logging distributed Elixir programs is absolutely possible and necessary to maintaining health and performance of the system. The BEAM's architecture necessitates a multi-faceted approach, combining traditional per-process profiling with message flow analysis, custom instrumentation, and centralized logging. Through consistent application of these techniques, alongside a deeper understanding of the Erlang ecosystem, one can build robust and observable distributed Elixir systems. For further study, explore resources on Erlang’s `:observer` application, metrics collection libraries like Prometheus client libraries for Erlang, and advanced logging strategies using structured logging. Also, delving into the tracing capabilities of distributed systems can be very helpful.
