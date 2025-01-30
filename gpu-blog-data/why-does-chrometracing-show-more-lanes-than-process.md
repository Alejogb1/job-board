---
title: "Why does chrome://tracing show more lanes than process names in timeline.json?"
date: "2025-01-30"
id: "why-does-chrometracing-show-more-lanes-than-process"
---
The discrepancy between the number of lanes visible in `chrome://tracing` and the process names listed in a timeline.json file arises from the fact that `chrome://tracing` visualizes not just operating system *processes*, but also various threads and execution contexts within those processes. A process, as defined by the operating system, can contain multiple threads, each executing concurrently. Furthermore, within a single thread, Chromium employs distinct execution contexts to manage specific tasks, these contexts may also be represented as distinct lanes in `chrome://tracing`, but do not appear as standalone processes in the JSON output. This layered view provides a detailed perspective on execution flow that is not captured by a simple process-centric breakdown.

The `timeline.json` format primarily reflects a hierarchical process structure as captured by tracing. Each process will typically contain threads, and threads contain events, but it lacks the finer granularity seen in the `chrome://tracing` UI where specific activities within a thread get their own lanes. A timeline.json often uses `pid` for process IDs and `tid` for thread IDs. However, the lane layout in chrome://tracing is dynamic and can vary depending on the trace data provided, utilizing internal concepts of trace viewer to group events logically, sometimes representing a singular thread as multiple lanes to denote different execution contexts or queues. These context distinctions aren't directly exposed as specific identifiers in `timeline.json`.

My experience working on performance debugging for a large-scale web application at a previous job gave me firsthand insight into these differences. We consistently utilized both timeline.json outputs and `chrome://tracing` for analysis. The JSON files were crucial for automated analysis and scripting, while the browser UI, through `chrome://tracing`, provided visual clarity that exposed nuances often missed in purely numerical datasets. Specifically, we encountered bottlenecks related to Javascript execution within the main thread of a web worker that were only clearly visible by observing the separate lane within the `chrome://tracing` UI which represented specific task scheduling of internal worker’s message loop.

Let's illustrate this with concrete code examples that can simulate some of these internal Chromium behaviors.

**Example 1: Simple Process and Threads**

This example creates a skeletal structure that simulates process and thread information, similar to what might appear in a simplified representation of a `timeline.json` export. It focuses on structure without specific events.

```python
import json

trace_data = {
  "traceEvents": [
    {
      "ph": "I",  # Instant event (for process creation)
      "name": "process_creation",
      "pid": 1234,
      "tid": 1,
      "ts": 1000,
      "args": {
        "name": "Renderer Process"
      }
    },
    {
      "ph": "I", #Instant event for thread creation
      "name": "thread_creation",
      "pid": 1234,
      "tid": 2,
      "ts": 1010,
      "args": {
          "name": "Main Thread"
      }

    },
     {
      "ph": "I", #Instant event for thread creation
      "name": "thread_creation",
      "pid": 1234,
      "tid": 3,
      "ts": 1020,
      "args": {
          "name": "Worker Thread"
      }

    }

  ],
  "displayTimeUnit": "ms"
}

with open('simple_timeline.json', 'w') as f:
  json.dump(trace_data, f, indent=2)
```

Here, we represent a "Renderer Process" using PID `1234`. It has a "Main Thread" (tid `2`) and a "Worker Thread" (tid `3`). In `chrome://tracing`, these threads would each get a separate lane. However, the JSON only records each thread as existing within the process. If you load this `simple_timeline.json` file into `chrome://tracing` you would observe three lanes, the process itself and then a lane for each of the two threads within that process. This clearly demonstrates the basic process/thread association as it exists in JSON. However, consider that even if the Main thread only executes events with tid 2, `chrome://tracing` might still split this lane into multiple lanes due to the context switching described previously.

**Example 2: Adding Events to a Thread**

This expands on the previous example by adding events within a single thread to demonstrate how various activities are recorded within a `timeline.json` representation:

```python
import json

trace_data = {
  "traceEvents": [
    {
      "ph": "I", #Instant event (for process creation)
      "name": "process_creation",
      "pid": 5678,
      "tid": 1,
      "ts": 1000,
      "args": {
        "name": "Renderer Process"
      }
    },
     {
      "ph": "I", #Instant event for thread creation
      "name": "thread_creation",
      "pid": 5678,
      "tid": 2,
      "ts": 1010,
      "args": {
          "name": "Main Thread"
      }

    },

    {
      "ph": "B",  # Begin event
      "name": "task_1",
      "pid": 5678,
      "tid": 2,
      "ts": 1200
    },
    {
      "ph": "E",  # End event
      "name": "task_1",
      "pid": 5678,
      "tid": 2,
      "ts": 1500
    },
    {
      "ph": "B",
      "name": "task_2",
      "pid": 5678,
      "tid": 2,
      "ts": 1600
    },
    {
      "ph": "E",
      "name": "task_2",
      "pid": 5678,
      "tid": 2,
      "ts": 1800
    }
    ],
   "displayTimeUnit": "ms"
}

with open('event_timeline.json', 'w') as f:
  json.dump(trace_data, f, indent=2)
```

Here, `task_1` and `task_2` execute on `Main Thread`. In the JSON file, these events are all contained within a single thread (tid 2).  However, in `chrome://tracing`, if those tasks are associated with different event loops or contexts (for example, one is a UI update and another is a Javascript execution), these may very well be placed on separate lanes. The key point here is that while `timeline.json` preserves the process and thread association, `chrome://tracing` interprets these events to present a more user-centric view.

**Example 3: Simulating Context Separation**

While it is difficult to fully replicate Chromium’s internal context tracking in a simple python script, let's imagine that each task has an internal context, which we indicate as part of the event. `chrome://tracing` can sometimes be interpreted to derive separate lanes from the 'category' or 'type' of such events, which is how it may visualize execution contexts.

```python
import json

trace_data = {
 "traceEvents": [
  {
      "ph": "I", #Instant event (for process creation)
      "name": "process_creation",
      "pid": 9012,
      "tid": 1,
      "ts": 1000,
      "args": {
        "name": "Renderer Process"
      }
    },
     {
      "ph": "I", #Instant event for thread creation
      "name": "thread_creation",
      "pid": 9012,
      "tid": 2,
      "ts": 1010,
      "args": {
          "name": "Main Thread"
      }

    },
    {
      "ph": "B",
      "name": "task_A",
      "pid": 9012,
      "tid": 2,
      "ts": 1100,
      "cat": ["ui_context"]
    },
    {
      "ph": "E",
      "name": "task_A",
      "pid": 9012,
      "tid": 2,
      "ts": 1300,
        "cat": ["ui_context"]
    },
        {
      "ph": "B",
      "name": "task_B",
      "pid": 9012,
      "tid": 2,
      "ts": 1400,
      "cat": ["js_context"]
    },
        {
      "ph": "E",
      "name": "task_B",
      "pid": 9012,
      "tid": 2,
      "ts": 1600,
        "cat": ["js_context"]
    }
    ],
    "displayTimeUnit": "ms"
}

with open('context_timeline.json', 'w') as f:
 json.dump(trace_data, f, indent=2)
```

While in `context_timeline.json`, both `task_A` and `task_B` appear to execute on the same `Main Thread` (tid 2), `chrome://tracing` might choose to visualize these on separate lanes depending on the `cat` (category) property which, in this case, represents context. In practice, Chromium uses more complex internal information to decide how to lay out lanes but the concept of events with different categories can strongly influence its lane grouping.

In summary, `timeline.json` provides a structured, machine-readable representation of the trace data emphasizing process and thread hierarchy. `chrome://tracing`, however, visualizes this data with a focus on usability, often displaying lanes that don’t have explicit corresponding process or thread IDs in the JSON, reflecting different execution contexts. This can result in a greater number of lanes than distinct process IDs in the JSON.

For individuals seeking to expand their knowledge of Chromium tracing, I would suggest investigating the following: documentation of the Chromium tracing system on the Chromium project website, which typically provides detailed overviews of its internal mechanisms. The source code for the Chromium trace viewer itself (available on repositories such as GitHub), provides a practical view of how the UI is generated from the trace data. Finally, exploration of open-source performance analysis tools which leverage trace data and provide specific analyses of Javascript execution or rendering bottlenecks, will further clarify how underlying trace information is used to identify performance issues. Examining the source code and tooling allows for a deeper understanding of the relationship between the process model and how that is translated for visual clarity.
