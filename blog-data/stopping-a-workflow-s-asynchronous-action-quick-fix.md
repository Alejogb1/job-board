---
title: "Stopping a Workflow's Asynchronous Action: Quick Fix?"
date: '2024-11-08'
id: 'stopping-a-workflow-s-asynchronous-action-quick-fix'
---

```java
CancellationScope longRunningCancellationScope =
          Workflow.newCancellationScope(
                  () -> Async.procedure(activities::longRunningActivity));
  longRunningCancellationScope.run();
  // Execute some synchronous activities
  Workflow.await(() -> !messageQueue.isEmpty());
  if (messageQueue.remove(0) == "something") {
      longRunningCancellationScope.cancel();
  }
```
