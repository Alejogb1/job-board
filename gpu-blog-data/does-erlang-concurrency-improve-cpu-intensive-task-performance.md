---
title: "Does Erlang concurrency improve CPU-intensive task performance?"
date: "2025-01-30"
id: "does-erlang-concurrency-improve-cpu-intensive-task-performance"
---
Erlang’s concurrency model, predicated on lightweight processes and message passing, does not inherently accelerate the execution of CPU-intensive tasks. Instead, it excels at managing concurrent operations, particularly those involving I/O or blocking activities, by keeping processes isolated and preventing any single process from stalling the entire system. The key distinction lies in how Erlang’s concurrency interacts with the underlying operating system scheduler.

When a task is CPU-bound—meaning its performance is primarily limited by the processor’s capacity to perform calculations—Erlang’s lightweight processes offer little direct benefit. Spawning numerous Erlang processes, each attempting a computationally demanding calculation, will not magically enhance the single-core processing throughput. The operating system will still schedule these processes to run on a finite number of cores, and the context-switching overhead between Erlang processes might even introduce a slight performance penalty compared to a single-threaded approach. Erlang's true strength is in improving *responsiveness* and *fault tolerance*, not raw calculation speed.

The illusion of parallelism can arise because Erlang manages to keep many operations progressing seemingly concurrently. However, the actual CPU time allocated to each CPU-intensive process will remain constrained by available physical resources. While many processes can be in progress at once, their individual execution rates will be limited. This is an important point to grasp: the parallelism Erlang enables is more about the ability to manage multiple things *at the same time* rather than *doing them faster*.

To understand this in practice, consider the following scenario. I once worked on an image processing service where the core functionality was a highly CPU-intensive filter application to images coming in. Initially, we mistakenly believed that spawning an Erlang process per image request would improve processing speed. The reality was stark. The image processing itself, which required extensive pixel-by-pixel manipulation, was consistently the bottleneck, irrespective of the number of Erlang processes attempting it. The improvement came when we recognized this and utilized techniques that offloaded computation and improved algorithm implementation – not by just increasing the amount of processes.

Here are a few illustrative code examples, with explanations of their implications in the context of CPU-intensive tasks:

**Example 1: Naive Concurrent Computation (Demonstrates the lack of direct improvement)**

```erlang
-module(cpu_intensive_naive).
-export([start/1]).

calculate(N) ->
    lists:sum([math:sqrt(I) || I <- lists:seq(1, N)]).

process_worker(N) ->
    Result = calculate(N),
    io:format("Result from worker ~p: ~p~n", [self(), Result]).

start(NumWorkers) ->
    lists:foreach(fun(_) -> spawn(fun() -> process_worker(10000000) end) end, lists:seq(1, NumWorkers)).
```

In this example, the `calculate/1` function performs a computationally intensive task – calculating the sum of the square roots of a large sequence of numbers. The `process_worker/1` function executes this and reports the result. The `start/1` function spawns `NumWorkers` concurrent processes, each performing the same calculation. If you were to execute this with varying numbers of workers – say 1, 10, or even 100 – you would find that the total time taken to complete all calculations does *not* decrease significantly. The limiting factor is the CPU’s capacity to execute `calculate/1` efficiently. Adding more processes only contributes to context-switching overhead, and doesn’t inherently speed up the computation.

**Example 2: Introducing a Native Driver (Demonstrates CPU-bound optimization via a different path)**

```erlang
-module(cpu_intensive_nif).
-export([start/1]).

-on_load(init_nif).

init_nif() ->
    erlang:load_nif("./cpu_intensive_nif", 0).

calculate_native(N) ->
    error("NIF not loaded"). % Placeholder, implemented in C

process_worker(N) ->
    Result = calculate_native(N),
    io:format("Result from native worker ~p: ~p~n", [self(), Result]).

start(NumWorkers) ->
    lists:foreach(fun(_) -> spawn(fun() -> process_worker(10000000) end) end, lists:seq(1, NumWorkers)).
```
This example illustrates a strategy we found effective, specifically leveraging a NIF (Native Implemented Function). We create a C function to execute a highly optimized version of the calculation. The corresponding NIF in our Erlang code interfaces with the native code, delegating the CPU-intensive part. While Erlang manages concurrency around this call, the core computation occurs outside the BEAM VM, potentially harnessing lower-level CPU features more efficiently. Even in this improved situation, there is still no benefit from extra processes doing the work. However, it is now done faster and Erlang still benefits from its ability to handle concurrent operations *around* that. Here’s the hypothetical C code being compiled to `cpu_intensive_nif.so`:

```c
#include <math.h>
#include <erl_nif.h>

static ERL_NIF_TERM calculate_native_nif(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    int n;
    if (!enif_get_int(env, argv[0], &n)) {
        return enif_make_badarg(env);
    }
    double result = 0.0;
    for (int i = 1; i <= n; ++i) {
        result += sqrt((double)i);
    }
    return enif_make_double(env, result);
}

static ErlNifFunc nif_funcs[] = {
    {"calculate_native", 1, calculate_native_nif}
};

ERL_NIF_INIT(cpu_intensive_nif, nif_funcs, NULL, NULL, NULL, NULL)
```

**Example 3: Distributing CPU-bound computation with a Remote Node ( Demonstrates scaling to multiple CPU resources).**

```erlang
-module(cpu_intensive_distributed).
-export([start/2]).

calculate(N) ->
    lists:sum([math:sqrt(I) || I <- lists:seq(1, N)]).

process_worker(Node, N) ->
    {ok, Result} = rpc:call(Node, cpu_intensive_distributed, calculate, [N]),
    io:format("Result from worker on node ~p: ~p~n", [Node, Result]).

start(NumWorkers, Nodes) ->
    lists:foreach(fun(Node) ->
        lists:foreach(fun(_)-> spawn(fun() -> process_worker(Node, 10000000) end) end, lists:seq(1, NumWorkers))
      end, Nodes).
```
This example demonstrates a more suitable use-case for Erlang concurrency with a CPU-intensive task. The core calculation remains within `calculate/1`, which is still inherently CPU-bound. However, now we have distributed this across multiple remote Erlang nodes (provided by the `Nodes` input variable), via the `rpc:call` function. The `start/2` function iterates through the list of remote nodes, spawning workers on each node, delegating calculations there. If each node has access to its own processor resources, we can now achieve true parallel processing of CPU-bound tasks because they are actually being executed on *different machines*. This approach significantly enhances throughput in a multi-node environment, and leverages the concurrency mechanisms of Erlang in a way that is suited for CPU intensive tasks.

For further study of these concepts, several resources are invaluable. Explore "Programming Erlang" by Joe Armstrong for a deep understanding of Erlang’s concurrency model and design principles. “Erlang in Anger” by Fred Herbert is an excellent guide for handling and debugging production Erlang systems, highlighting potential pitfalls and best practices. Finally, delving into the BEAM (Bogdan/Björn’s Erlang Abstract Machine) documentation will provide deeper insights into how Erlang code is executed and scheduled. The key is to recognize that Erlang doesn't inherently make CPU-bound tasks faster but instead provides mechanisms that allow for more efficient resource utilisation when such tasks have a context of concurrency, especially on I/O or distributed systems. Focus on offloading or distributing the computations to benefit from the concurrency.
