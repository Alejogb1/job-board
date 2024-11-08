---
title: "Kickstart Temporal Workflows with Signals: How to Trigger Child Processes"
date: '2024-11-08'
id: 'kickstart-temporal-workflows-with-signals-how-to-trigger-child-processes'
---

```java
import io.temporal.activity.ActivityOptions;
import io.temporal.common.cancellation.CancellationScope;
import io.temporal.workflow.Workflow;

public class ChildWorkflowReset {

    public void resetChildWorkflow() {
        try (CancellationScope scope = Workflow.newCancellationScope()) {
            // Start the child workflow within the cancellation scope
            ChildWorkflow childWorkflow = Workflow.newChildWorkflowStub(ChildWorkflow.class, ActivityOptions.newBuilder()
                    .setCancellationType(CancellationScope.CancellationType.CHILD_WORKFLOW)
                    .build());
            scope.registerChild(childWorkflow); // Register the child workflow for cancellation

            // Perform some work that might trigger the reset event
            // ...

            // If reset event occurs, cancel the cancellation scope
            if (resetEventOccurred()) {
                scope.cancel(); 
            }

            // If the scope wasn't canceled, the child workflow will finish naturally
            // If the scope was canceled, the child workflow will be canceled and cleaned up
        }

        // After the scope is closed (either naturally or through cancellation),
        // you can optionally start a new child workflow if needed.
        // This ensures that the child workflow is always in a consistent state.
        if (shouldStartNewChild()) {
            // Start a new child workflow
            Workflow.newChildWorkflowStub(ChildWorkflow.class);
        }
    }

    // Methods for your specific logic
    private boolean resetEventOccurred() {
        // Replace with your logic to detect the reset event
        return false;
    }

    private boolean shouldStartNewChild() {
        // Replace with your logic to determine if a new child workflow should be started
        return false;
    }

    // ... other workflow methods ...
}

// Child workflow implementation
interface ChildWorkflow {
    // Child workflow methods
} 
```
