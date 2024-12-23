---
title: "How can Metal command buffers use async/await for completion handling?"
date: "2024-12-23"
id: "how-can-metal-command-buffers-use-asyncawait-for-completion-handling"
---

Let's dive straight in; it’s a topic that certainly prompted some late nights during my time optimizing a real-time rendering pipeline a few years back. The quest to integrate asynchronous operations with Metal command buffers, specifically with `async`/`await`, isn’t as straightforward as with some other asynchronous frameworks, but it's entirely achievable, and frankly, it's a game-changer for performance. The core issue stems from Metal’s design: command buffers are submitted to the gpu for execution, and their completion is signaled through a mechanism distinct from the event loops used by typical async/await implementations. The default is for a completion handler to fire *later*, via the cpu.

Fundamentally, `async`/`await` is syntactic sugar for managing asynchronous operations, but it relies on the underlying support of an event loop or a similar mechanism to manage continuations. Metal, on the other hand, reports completion using a `MTLCommandBuffer` completion handler. The challenge, then, is to bridge the gap between Metal’s asynchronous signaling and the cooperative multitasking provided by `async`/`await`. We need to construct a bridge that turns completion callbacks into awaitable tasks.

The first approach that comes to mind, and one I've used frequently, involves wrapping the command buffer execution within a `Task` and its associated `withUnsafeContinuation` function. This allows us to suspend execution until the completion handler is invoked and subsequently resume the task with the result (or error) of the operation. This pattern is essential when you want to integrate Metal’s execution flow directly into your async code, rather than just kicking off gpu tasks and immediately moving on.

Here’s an example of that:

```swift
import Metal
import Foundation

func executeCommandBufferAsync(device: MTLDevice, commandQueue: MTLCommandQueue, renderPassDescriptor: MTLRenderPassDescriptor, renderBlock: (_ commandBuffer: MTLCommandBuffer, _ renderEncoder: MTLRenderCommandEncoder) -> Void) async throws {

    return try await withUnsafeThrowingContinuation { continuation in
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            continuation.resume(throwing: NSError(domain: "MetalError", code: -1, userInfo: [NSLocalizedDescriptionKey: "Failed to create command buffer"]))
            return
        }

        guard let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else {
            continuation.resume(throwing: NSError(domain: "MetalError", code: -2, userInfo: [NSLocalizedDescriptionKey: "Failed to create render command encoder"]))
            return
        }

        renderBlock(commandBuffer, renderEncoder)
        renderEncoder.endEncoding()

        commandBuffer.addCompletedHandler { (buffer) in
            if let error = buffer.error {
                continuation.resume(throwing: error)
            } else {
                 continuation.resume(returning: ()) // Successfully completed the command buffer
            }
        }
        
        commandBuffer.commit()
    }
}


// Example Usage:

//Assume a device, queue and descriptor are initialized
func renderAsync(device: MTLDevice, commandQueue: MTLCommandQueue, renderPassDescriptor: MTLRenderPassDescriptor) async throws {

   try await executeCommandBufferAsync(device: device, commandQueue: commandQueue, renderPassDescriptor: renderPassDescriptor){ (commandBuffer, renderEncoder) in

       // Add your rendering commands here
      renderEncoder.clearColor = MTLClearColor(red: 0.0, green: 1.0, blue: 0.0, alpha: 1.0) // Set a simple clear color as an example
    }
   print("Render Command buffer completed asynchronously")

}
```

In this example, the `executeCommandBufferAsync` function neatly encapsulates the execution of a single command buffer into an `async` function. The `withUnsafeThrowingContinuation` handles the suspension until the completion handler is invoked, allowing for graceful error handling and reporting of successful completion to the caller.

Now, one might have a chain of command buffers, all of which need to execute sequentially and asynchronously. Instead of repeatedly wrapping each individual command buffer, we can create an asynchronous function that manages the execution of multiple command buffers sequentially using the same `withUnsafeThrowingContinuation` approach, but within a loop and leveraging the await keyword with previous async operations.

Here’s how you might implement that:

```swift
import Metal
import Foundation

func executeCommandBufferChainAsync(device: MTLDevice, commandQueue: MTLCommandQueue, renderPassDescriptors: [MTLRenderPassDescriptor], renderBlocks: [(_ commandBuffer: MTLCommandBuffer, _ renderEncoder: MTLRenderCommandEncoder) -> Void]) async throws {

    guard renderPassDescriptors.count == renderBlocks.count else {
        throw NSError(domain: "MetalError", code: -3, userInfo: [NSLocalizedDescriptionKey: "Number of render pass descriptors must match number of blocks"])
    }

    for (index, renderPassDescriptor) in renderPassDescriptors.enumerated(){

          try await withUnsafeThrowingContinuation { continuation in
              guard let commandBuffer = commandQueue.makeCommandBuffer() else {
                  continuation.resume(throwing: NSError(domain: "MetalError", code: -4, userInfo: [NSLocalizedDescriptionKey: "Failed to create command buffer"]))
                  return
              }

              guard let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else {
                  continuation.resume(throwing: NSError(domain: "MetalError", code: -5, userInfo: [NSLocalizedDescriptionKey: "Failed to create render command encoder"]))
                  return
              }

            renderBlocks[index](commandBuffer, renderEncoder)

            renderEncoder.endEncoding()


             commandBuffer.addCompletedHandler { (buffer) in
                  if let error = buffer.error {
                      continuation.resume(throwing: error)
                  } else {
                       continuation.resume(returning: ()) // Successfully completed the command buffer
                  }
             }

            commandBuffer.commit()

        }
    }
}

//Example Usage

func renderMultipleAsync(device: MTLDevice, commandQueue: MTLCommandQueue, renderPassDescriptors: [MTLRenderPassDescriptor]) async throws{
  
    //Assume you have an array of render pass descriptors
    let renderBlocks: [(_ commandBuffer: MTLCommandBuffer, _ renderEncoder: MTLRenderCommandEncoder) -> Void] = [
        {(commandBuffer, renderEncoder) in
            renderEncoder.clearColor = MTLClearColor(red: 1.0, green: 0.0, blue: 0.0, alpha: 1.0) //First pass, red
        },
        {(commandBuffer, renderEncoder) in
            renderEncoder.clearColor = MTLClearColor(red: 0.0, green: 0.0, blue: 1.0, alpha: 1.0) //Second pass, blue
        }
    ]

    try await executeCommandBufferChainAsync(device: device, commandQueue: commandQueue, renderPassDescriptors: renderPassDescriptors, renderBlocks: renderBlocks)

    print("Render chain completed asynchronously")


}
```

This `executeCommandBufferChainAsync` iterates over an array of render descriptors and blocks to be executed, awaiting the completion of each individually before moving on to the next, thus chaining the rendering passes and maintaining execution order with async/await. This is extremely useful for complex rendering tasks where the outputs of one step need to be available for the next.

Finally, there's a further optimization related to *resource* usage in Metal, that's critical to avoid stalls between command buffer submissions. Metal allows you to provide a list of *shared* resources that are used by *multiple* command buffers, via a *shared event*. So, the previous pattern is good for things that *depend* on each other, like one rendering pass depending on the output of another. But if you've got *independent* workloads that *share resources*, then that second pattern is going to serialize those executions when they could be running in parallel. Here is an example of that pattern:

```swift
import Metal
import Foundation


func executeCommandBufferConcurrentAsync(device: MTLDevice, commandQueue: MTLCommandQueue, renderPassDescriptors: [MTLRenderPassDescriptor], renderBlocks: [(_ commandBuffer: MTLCommandBuffer, _ renderEncoder: MTLRenderCommandEncoder) -> Void], sharedEvent: MTLEvent) async throws {

    guard renderPassDescriptors.count == renderBlocks.count else {
        throw NSError(domain: "MetalError", code: -6, userInfo: [NSLocalizedDescriptionKey: "Number of render pass descriptors must match number of blocks"])
    }

    await withThrowingTaskGroup(of: Void.self) { group in
            for (index, renderPassDescriptor) in renderPassDescriptors.enumerated(){

              group.addTask {
                  try await withUnsafeThrowingContinuation { continuation in
                    guard let commandBuffer = commandQueue.makeCommandBuffer() else {
                        continuation.resume(throwing: NSError(domain: "MetalError", code: -7, userInfo: [NSLocalizedDescriptionKey: "Failed to create command buffer"]))
                        return
                    }
                    
                    guard let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else {
                        continuation.resume(throwing: NSError(domain: "MetalError", code: -8, userInfo: [NSLocalizedDescriptionKey: "Failed to create render command encoder"]))
                        return
                    }
                    
                    renderBlocks[index](commandBuffer, renderEncoder)
                    renderEncoder.endEncoding()


                   commandBuffer.addCompletedHandler { (buffer) in
                        if let error = buffer.error {
                            continuation.resume(throwing: error)
                        } else {
                            continuation.resume(returning: ()) // Successfully completed the command buffer
                        }
                    }

                    commandBuffer.encodeWaitForEvent(sharedEvent, value: 0) // Encode a wait before our work begins
                    commandBuffer.commit()
                    commandBuffer.encodeSignalEvent(sharedEvent, value: 1) // Encode signal after we're finished
                }
              }
            }
        
            try await group.waitForAll()
        }
}


func renderConcurrentAsync(device: MTLDevice, commandQueue: MTLCommandQueue, renderPassDescriptors: [MTLRenderPassDescriptor], sharedEvent: MTLEvent) async throws {

    let renderBlocks: [(_ commandBuffer: MTLCommandBuffer, _ renderEncoder: MTLRenderCommandEncoder) -> Void] = [
        {(commandBuffer, renderEncoder) in
            renderEncoder.clearColor = MTLClearColor(red: 1.0, green: 0.0, blue: 0.0, alpha: 1.0) //First pass, red
        },
        {(commandBuffer, renderEncoder) in
            renderEncoder.clearColor = MTLClearColor(red: 0.0, green: 0.0, blue: 1.0, alpha: 1.0) //Second pass, blue
        }
    ]

     try await executeCommandBufferConcurrentAsync(device: device, commandQueue: commandQueue, renderPassDescriptors: renderPassDescriptors, renderBlocks: renderBlocks, sharedEvent: sharedEvent)
    print("Concurrent Render buffers completed")
}
```

In this concurrent example, we are using `withThrowingTaskGroup` to launch several rendering passes in parallel. The key here is the shared event, using `encodeWaitForEvent` and `encodeSignalEvent`, ensuring the gpu is not stalled waiting on resources.

For a more in-depth understanding of Metal, I highly recommend diving into Apple’s official Metal documentation, it's quite comprehensive. Furthermore, “Metal Programming Guide” from Apple’s Developer Library is a must-read. Additionally, the book "Metal by Tutorials" from raywenderlich.com is an excellent practical resource that blends theory with hands-on projects. For a deeper dive into concurrent programming, consider studying the material from Doug Lea's "Concurrent Programming in Java: Design Principles and Patterns", while the focus is Java, the core principles of concurrency are applicable across many platforms. Finally, the Swift Concurrency documentation should also be a reference, especially for things like task groups, async/await, and actors.

These approaches, particularly the third pattern using shared events, were crucial in the rendering pipelines I worked on, and they allowed us to seamlessly integrate the asynchronous nature of Metal with the modern asynchronous paradigms. It required a bit of extra plumbing, but the performance gains were well worth it. Remember that choosing between these solutions will depend entirely on your specific use case and the interplay of resources between your rendering passes.
