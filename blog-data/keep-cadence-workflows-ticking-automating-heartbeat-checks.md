---
title: "Keep Cadence Workflows Ticking: Automating Heartbeat Checks"
date: '2024-11-08'
id: 'keep-cadence-workflows-ticking-automating-heartbeat-checks'
---

```go
package main

import (
	"context"
	"time"

	"go.uber.org/cadence/activity"
	"go.uber.org/cadence/worker"
)

// MyActivity is a sample activity implementation.
type MyActivity struct{}

// Execute is the main function of the activity.
func (a *MyActivity) Execute(ctx context.Context, input string) (string, error) {
	// Simulate long-running operation by sleeping for 2 hours.
	time.Sleep(2 * time.Hour)
	return "Activity completed!", nil
}

func main() {
	// Register the activity with auto heartbeat enabled and a heartbeat timeout of 10 seconds.
	// This will automatically send heartbeats every 8 seconds (80% of the heartbeat timeout).
	// It is recommended to use a smaller heartbeat timeout (10-20 seconds) for faster retry detection.
	//
	// The activity start to close timeout should be set to 2 hours in this case to accommodate the long-running operation.
	//
	// The heartbeat timeout is used to detect if the activity worker is still alive.
	// If the activity worker crashes or is killed, Cadence will retry the activity after the heartbeat timeout.
	// The activity start to close timeout is used to detect if the activity is taking too long to complete.
	// If the activity takes longer than the start to close timeout, Cadence will fail the activity.
	//
	// In this example, the activity will be retried after 10 seconds if the activity worker crashes or is killed.
	// If the activity takes longer than 2 hours, Cadence will fail the activity.
	activity.Register(worker.New(
		worker.Options{
			ActivityOptions: []activity.Options{
				{
					EnableAutoHeartbeat: true,
					HeartbeatTimeout:    10 * time.Second,
				},
			},
		},
	), &MyActivity{})

	// Start the worker.
	err := worker.Start(context.Background())
	if err != nil {
		panic(err)
	}
}
```
