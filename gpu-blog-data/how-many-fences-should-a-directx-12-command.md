---
title: "How many fences should a DirectX 12 command allocator use?"
date: "2025-01-30"
id: "how-many-fences-should-a-directx-12-command"
---
The optimal number of fences for a DirectX 12 command allocator is typically one, and rarely more than two, per render thread, if utilizing multi-threading for rendering. I've found, through practical experience developing a custom game engine, that overusing fences, particularly when not coupled with careful command list management, can negate the performance gains multi-threading should provide, primarily due to unnecessary pipeline stalls. Let's delve into why.

Firstly, a command allocator in DirectX 12 is a memory pool used to allocate memory for command lists. These command lists encapsulate GPU instructions, and their execution is orchestrated by the command queue. Fences are signaling mechanisms used to synchronize operations between the CPU and GPU. They essentially allow the CPU to wait for the GPU to reach a specific point in its processing, preventing race conditions and ensuring data dependencies are handled correctly. The need for fences arises from the asynchronous nature of GPU execution. The CPU may submit command lists to the queue and proceed without waiting for their completion. We therefore need to know when specific resources are available for reuse, and that's where fences become important.

The key concept is command list lifetime. A command list allocated from a given allocator cannot be reset and reused until all operations submitted using it (and therefore all associated resources used within that command list) have completed on the GPU. If you attempt to reset a command list while it's still being processed, undefined behavior and likely crashes will occur. Fences ensure that reset operations are only executed after the appropriate work has completed.

A single fence per thread, particularly in a scenario where rendering operations are partitioned into separate threads, is often sufficient. In this case, each thread would obtain an allocator, record the command list, submit the command list to a command queue, and signal the fence. The next time this thread needs to record a command list (next frame), it waits for the fence to be signaled, verifying the GPU has finished its previous batch of work before resetting the command list.

The primary pitfall I’ve seen with using multiple fences per command allocator comes from overly aggressive pipelining and poor management of resources tied to those command lists. For instance, consider the following scenario: you create multiple fences and several command lists associated with the same allocator. You may be tempted to submit several command lists one after another, signaling a different fence each time, thinking this creates a deeper pipeline. While this can seem beneficial at first glance, it frequently results in the CPU being blocked waiting for all fences to be signaled before resubmitting further work, reducing multi-threading effectiveness. It also adds bookkeeping complexity. Furthermore, if the GPU does not complete the command lists in the order they are submitted, there is no simple way to know which specific fence to wait for next. This can actually decrease overall performance.

Here's an illustrative code example using one fence per command allocator within a single threaded operation, which represents an ideal starting point:

```cpp
// Assume device, commandQueue, and allocator are already created.
ID3D12Fence* frameFence = nullptr;
UINT64 fenceValue = 0;
HANDLE eventHandle = CreateEvent(nullptr, FALSE, FALSE, nullptr);
device->CreateFence(fenceValue, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&frameFence));

// Command list recording.
ID3D12GraphicsCommandList* cmdList = nullptr;
allocator->Reset();
device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator, nullptr, IID_PPV_ARGS(&cmdList));
// Record graphics commands into cmdList ...
cmdList->Close();

// Submit command list to command queue.
commandQueue->ExecuteCommandLists(1, reinterpret_cast<ID3D12CommandList**>(&cmdList));
commandQueue->Signal(frameFence, ++fenceValue);

// Wait for GPU completion.
if (frameFence->GetCompletedValue() < fenceValue) {
    frameFence->SetEventOnCompletion(fenceValue, eventHandle);
    WaitForSingleObject(eventHandle, INFINITE);
}

// Reset for next frame
cmdList->Reset(allocator, nullptr);
// ... Continue the cycle
```

In this example, `frameFence` and `fenceValue` track when the command list has finished executing on the GPU. `SetEventOnCompletion` and `WaitForSingleObject` are used to block the CPU thread until the GPU signals the fence, preventing the CPU from resetting the command list before it is ready. This provides synchronization between the CPU and GPU and utilizes a single fence for the allocator and command list.

A scenario where two fences might prove useful, and I have employed in some specific scenarios, involves double buffering. Here's a possible approach:

```cpp
// Assume device, commandQueue, and allocator are already created.
ID3D12Fence* frameFences[2];
UINT64 fenceValues[2] = {0, 0};
HANDLE eventHandles[2];

for(int i = 0; i < 2; i++){
    eventHandles[i] = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    device->CreateFence(fenceValues[i], D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&frameFences[i]));
}

int bufferIndex = 0;

// Command list recording.
ID3D12GraphicsCommandList* cmdList = nullptr;
allocator->Reset();
device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator, nullptr, IID_PPV_ARGS(&cmdList));

// Record commands...
cmdList->Close();

// Submit command list and signal the corresponding fence
commandQueue->ExecuteCommandLists(1, reinterpret_cast<ID3D12CommandList**>(&cmdList));
commandQueue->Signal(frameFences[bufferIndex], ++fenceValues[bufferIndex]);

// Wait for the previous frame's work to complete
int previousBufferIndex = (bufferIndex + 1) % 2;

if (frameFences[previousBufferIndex]->GetCompletedValue() < fenceValues[previousBufferIndex]) {
    frameFences[previousBufferIndex]->SetEventOnCompletion(fenceValues[previousBufferIndex], eventHandles[previousBufferIndex]);
    WaitForSingleObject(eventHandles[previousBufferIndex], INFINITE);
}

// Reset for next frame
cmdList->Reset(allocator, nullptr);
bufferIndex = (bufferIndex + 1) % 2;

// ... Continue the cycle
```

Here, we use two fences and two event handles. The `bufferIndex` cycles between 0 and 1 allowing the CPU to work on preparing the next frame whilst the previous frame's commands are executing on the GPU. While this can offer marginal benefits, it should be noted that this is more an example of *double buffering* rather than an increase in fences per command allocator, and is often more complex to manage. It only really yields benefit in situations with long render times and minimal CPU workload. Even then, it is often better to rely on multi-threading rather than double buffering.

Finally, consider a scenario with multiple render threads, a common scenario for complex rendering. Here, we’d use a single fence per thread's allocator:

```cpp
// Assume device, commandQueue, and allocator are already created for each thread.
// This is thread code! Assume thread-specific commandQueue and allocator.
ID3D12Fence* threadFence;
UINT64 threadFenceValue = 0;
HANDLE threadEventHandle = CreateEvent(nullptr, FALSE, FALSE, nullptr);
device->CreateFence(threadFenceValue, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&threadFence));

// Command list recording
ID3D12GraphicsCommandList* threadCmdList = nullptr;
allocator->Reset();
device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator, nullptr, IID_PPV_ARGS(&threadCmdList));

// Record commands into threadCmdList....
threadCmdList->Close();

// Submit command list and signal fence
commandQueue->ExecuteCommandLists(1, reinterpret_cast<ID3D12CommandList**>(&threadCmdList));
commandQueue->Signal(threadFence, ++threadFenceValue);

// Wait for previous frame on this thread
if (threadFence->GetCompletedValue() < threadFenceValue) {
    threadFence->SetEventOnCompletion(threadFenceValue, threadEventHandle);
    WaitForSingleObject(threadEventHandle, INFINITE);
}

// Reset for next frame
threadCmdList->Reset(allocator, nullptr);

// ... Continue the cycle
```

Here each thread possesses its own allocator, fence, and command queue. This is the typical structure of multi-threaded rendering and allows the GPU to process the work submitted by each thread in parallel. The key factor is that command allocators and command lists are never used between threads.

In summary, for most rendering scenarios, using one fence per command allocator per render thread is the most efficient approach. Overly complex fence setups generally yield minimal performance gain and increase code complexity. A single fence is sufficient to ensure correct synchronization of resources within a given render thread. Avoid multiple fences on the same allocator, as this typically introduces excessive complexity without improving performance. I would recommend studying the DirectX documentation relating to command queues, command allocators, and fences. The samples provided with the SDK are also a great resource. Furthermore, consider exploring advanced techniques like residency management and implicit synchronization, which can help optimize further. A deep understanding of GPU command submission is crucial.
