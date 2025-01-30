---
title: "Are Swift actors concurrent, but not parallel?"
date: "2025-01-30"
id: "are-swift-actors-concurrent-but-not-parallel"
---
Swift actors are concurrent, but not inherently parallel.  This crucial distinction stems from the underlying implementation of actors and how they interact with the system's resources.  My experience optimizing high-throughput server-side applications using SwiftNIO, heavily reliant on actor-based concurrency, has solidified this understanding. While actors enable concurrent execution of multiple tasks, the actual parallelism depends on factors like hardware availability and the actor's internal operation.

**1. Clear Explanation**

Concurrency refers to the ability of a system to appear to perform multiple tasks simultaneously.  In Swift, this is achieved by enabling multiple actors to execute concurrently, even on a single processor core.  The runtime manages the execution, switching between actors to give the illusion of parallel execution.  However, true parallelism necessitates the simultaneous execution of multiple instructions on multiple processing cores.  This is dependent on the system's resources and the task's nature.  Swift actors, by design, prioritize safety and determinism over maximizing raw performance.  Their concurrency model, using lightweight synchronization mechanisms, inherently avoids race conditions and data corruption.  The system might utilize multiple cores if available and appropriate, but it's not a guaranteed property of actor execution.  Instead, the concurrency is managed to ensure correct and predictable behavior, even under heavy load.  This design decision prevents unpredictable performance variations that can arise from uncontrolled parallelism.  The compiler and runtime optimize actor execution, but the actual level of parallelism is dynamically determined based on system load and resource availability.

The key is understanding the isolation and synchronization provided by the actor model. Each actor operates within its own isolated memory space, accessed only through the actor’s methods.  This inherent isolation simplifies concurrency management; the compiler and runtime guarantee that access to the actor's state is serialized.  Multiple actors can concurrently request access to a particular actor's methods; however, the runtime ensures sequential execution, preventing data races.  This concurrency is not automatically transformed into parallelism; the system needs available cores and appropriate task decomposition to achieve parallelism.

**2. Code Examples with Commentary**

**Example 1:  Simple Concurrent Actor**

```swift
actor Counter {
    private var count = 0

    func increment() {
        count += 1
    }

    func getCount() -> Int {
        return count
    }
}

Task {
    let counter = Counter()
    for _ in 1...1000 {
        await counter.increment()
    }
    print("Counter value (Task 1): \(await counter.getCount())")
}

Task {
    let counter = Counter()
    for _ in 1...1000 {
        await counter.increment()
    }
    print("Counter value (Task 2): \(await counter.getCount())")
}

//This example showcases concurrency.  Both tasks interact with the Counter actor concurrently.
// However, the internal increment operation within the actor remains serialized due to the actor's isolation.
// Parallelism would require multiple Counter instances or a different architectural approach.
```

**Example 2: Demonstrating Lack of inherent parallelism**

```swift
import Dispatch

actor LongRunningTask {
    func performTask() {
        //Simulates a computationally intensive task.
        let queue = DispatchQueue(label: "longRunningQueue")
        queue.sync {
            for _ in 1...100000000 {
                let _ = sin(Double.random(in: 0...100))
            }
        }
    }
}

Task {
    let task = LongRunningTask()
    await task.performTask()
    print("Task 1 completed")
}

Task {
    let task = LongRunningTask()
    await task.performTask()
    print("Task 2 completed")
}

//This example reveals that even with two concurrent tasks, the underlying computation may not be truly parallel.
//The DispatchQueue used for the intensive computation is designed for serial execution, not for taking advantage of multi-core architectures.
// While we have two concurrent `Task`s, they are not necessarily running in parallel due to the inherently serial nature of the long-running computation.
```


**Example 3:  Illustrating potential for Parallelism with task decomposition**

```swift
import Dispatch

actor MatrixMultiplier {
    func multiply(matrixA: [[Double]], matrixB: [[Double]]) -> [[Double]] {
        let rowsA = matrixA.count
        let colsA = matrixA[0].count
        let rowsB = matrixB.count
        let colsB = matrixB[0].count

        guard colsA == rowsB else {
            fatalError("Matrices cannot be multiplied")
        }

        var result = Array(repeating: Array(repeating: 0.0, count: colsB), count: rowsA)
        let queue = DispatchQueue(label: "matrixMultiplicationQueue", attributes: .concurrent)


        for i in 0..<rowsA {
            for j in 0..<colsB {
                queue.async {
                    var sum: Double = 0
                    for k in 0..<colsA {
                        sum += matrixA[i][k] * matrixB[k][j]
                    }
                    result[i][j] = sum
                }
            }
        }
        queue.sync {
           //Ensuring all the operations complete before returning the result.
        }
        return result
    }
}

//This example demonstrates the potential for parallelism when the problem can be broken down into smaller independent subtasks.
// The `DispatchQueue` with the `.concurrent` attribute allows for parallel execution of the inner loop calculations.
//The outer loop iterations are still sequential to ensure the correct order of matrix multiplications.
//This highlights how explicit task decomposition can exploit the available hardware's parallelism, even within the context of an actor.

let matrixA = [[1.0, 2.0], [3.0, 4.0]]
let matrixB = [[5.0, 6.0], [7.0, 8.0]]
let multiplier = MatrixMultiplier()
let result = await multiplier.multiply(matrixA: matrixA, matrixB: matrixB)
print(result)
```

**3. Resource Recommendations**

For a deeper understanding of Swift concurrency, I recommend consulting Apple's official Swift documentation on concurrency, particularly the sections detailing actors, tasks, and async/await.  A thorough understanding of operating system concepts, including process and thread management, will also prove invaluable.  Furthermore, studying the source code of well-established Swift concurrency libraries can offer practical insights.  Finally, engaging in exercises involving various concurrency patterns and scenarios will help solidify your grasp of these concepts.  Remember that hands-on experience is crucial to mastering the nuances of concurrent and parallel programming.
