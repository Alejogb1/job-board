---
title: "How can I measure compute shader execution time in Unity?"
date: "2025-01-30"
id: "how-can-i-measure-compute-shader-execution-time"
---
Profiling compute shader execution accurately in Unity demands a nuanced approach, primarily because GPU operations are inherently asynchronous. Simply measuring time around a `Dispatch` call yields an incomplete and often misleading picture. The CPU initiates the work, but the GPU processes it independently, leading to discrepancies between CPU-side timing and actual GPU execution duration. My experiences developing real-time visual effects in Unity have underscored the importance of understanding this asynchronicity.

The key to accurate profiling lies in leveraging Unity’s built-in profiling tools, specifically the Frame Debugger and the Profiler window, in conjunction with GPU timing queries. Relying solely on `System.Diagnostics.Stopwatch` or similar CPU-based timers will measure the time it takes for the CPU to dispatch the compute shader, not the shader's execution duration on the GPU. This CPU time is typically dominated by the overhead of command buffer creation and submission to the graphics API rather than the compute kernel’s actual processing.

A proper measurement strategy involves creating a GPU query, executing the compute shader dispatch, and then retrieving the elapsed time associated with that query after the GPU completes the work. This approach ensures that only the actual GPU processing time is captured, excluding CPU overhead and any implicit synchronization points. Failing to account for the asynchronous nature of GPU operations may lead to inaccurate performance analysis and flawed optimization strategies.

First, we must initialize a GPU query using Unity's `Graphics.CreateQuery`. This query will act as a time measurement tool that resides on the GPU. We will then instruct the GPU to begin timing just before dispatching the compute shader, and then stop timing just after the shader has finished its work. To retrieve these timings, we will use `Graphics.GetQueryResults` once the results are ready. The query system, however, operates in a deferred manner so the results are not available immediately after submitting. The retrieval logic needs to be placed in a location in your code where all previous GPU work associated with the query has been processed. This is typically done toward the end of the frame, or just before rendering.

Now, let’s illustrate this with a simple example. Consider a compute shader that performs a basic vector addition. The following code snippet demonstrates how to set up and time this execution.

```csharp
using UnityEngine;

public class ComputeShaderTimer : MonoBehaviour
{
    public ComputeShader computeShader;
    public int dataSize = 1024;
    private int kernelId;
    private ComputeBuffer bufferA;
    private ComputeBuffer bufferB;
    private ComputeBuffer bufferResult;
    private int[] dataA;
    private int[] dataB;
    private uint[] queryResult = new uint[2];
    private AsyncGPUReadbackRequest readbackRequest;
    private string shaderTimerName = "Compute Shader";
    private UnityEngine.Profiling.CustomSampler customSampler;
    private UnityEngine.Profiling.CustomSampler gpuSampler;
    private GraphicsQuery timerQuery;
    private bool timerPending = false;

    void Start()
    {
        kernelId = computeShader.FindKernel("CSMain");
        bufferA = new ComputeBuffer(dataSize, sizeof(int));
        bufferB = new ComputeBuffer(dataSize, sizeof(int));
        bufferResult = new ComputeBuffer(dataSize, sizeof(int));
        dataA = new int[dataSize];
        dataB = new int[dataSize];
        for (int i = 0; i < dataSize; i++)
        {
            dataA[i] = i;
            dataB[i] = dataSize - i;
        }
        bufferA.SetData(dataA);
        bufferB.SetData(dataB);
        customSampler = UnityEngine.Profiling.CustomSampler.Create(shaderTimerName);
        timerQuery = Graphics.CreateQuery(GraphicsQueryType.TimeElapsed);
         gpuSampler = UnityEngine.Profiling.CustomSampler.Create("GPU Time");
    }

    void Update()
    {
       // Check to make sure previous frame is complete.
        if (timerPending && readbackRequest.done){
            // Retrieve the data.
            readbackRequest.GetData(queryResult);
            float gpuTime = (queryResult[1] - queryResult[0]) / 1000000.0f; // Convert to milliseconds.
            gpuSampler.End();
            timerPending = false;
            Debug.Log("GPU Time: " + gpuTime.ToString() + "ms");
        }
        // If we aren't pending, then start the timer
        if(!timerPending){
            customSampler.Begin();
             gpuSampler.Begin();
            timerPending = true;
           DispatchComputeShader();
           customSampler.End();
           readbackRequest = timerQuery.RequestAsyncReadback(queryResult.Length * sizeof(uint));
        }
    }

    void DispatchComputeShader()
    {
        computeShader.SetBuffer(kernelId, "bufferA", bufferA);
        computeShader.SetBuffer(kernelId, "bufferB", bufferB);
        computeShader.SetBuffer(kernelId, "bufferResult", bufferResult);
        Graphics.BeginSample("Compute Shader Dispatch");
        timerQuery.Begin();
        computeShader.Dispatch(kernelId, dataSize / 64, 1, 1);
        timerQuery.End();
        Graphics.EndSample();
    }

    void OnDestroy()
    {
        bufferA.Dispose();
        bufferB.Dispose();
        bufferResult.Dispose();
        timerQuery.Dispose();
    }
}

```

In this example, a `GraphicsQuery` is initialized with `GraphicsQueryType.TimeElapsed`. The `Begin` and `End` methods mark the start and end points for the timing on the GPU. To properly retrieve the timing values, we need to wait for the readback request to be complete. We then use the returned array of integers to calculate the GPU execution time in milliseconds, which is logged to the console. Additionally, we use the `CustomSampler` class to show how to view the timing results in the Unity Profiler.

The compute shader, for completeness, is as follows:

```shader
#pragma kernel CSMain

RWStructuredBuffer<int> bufferA;
RWStructuredBuffer<int> bufferB;
RWStructuredBuffer<int> bufferResult;

[numthreads(64,1,1)]
void CSMain (uint3 id : SV_DispatchThreadID)
{
    bufferResult[id.x] = bufferA[id.x] + bufferB[id.x];
}
```

This is a simple kernel that adds the elements from two input buffers and places the results in a third. The dispatch size is chosen so the thread count is equal to the size of the buffers.

Finally, a more complex example involves multiple dispatch calls and synchronizing the readback of query results. In this instance, we measure multiple dispatches using a single `GraphicsQuery`, illustrating how to measure the performance of several kernel calls at once. This approach can prove beneficial when assessing the impact of algorithmic changes across various parts of the same compute shader.

```csharp
using UnityEngine;
using System.Collections.Generic;

public class MultiDispatchTimer : MonoBehaviour
{
    public ComputeShader computeShader;
    public int dataSize = 1024;
    private int kernelId;
    private ComputeBuffer bufferA;
    private ComputeBuffer bufferB;
    private ComputeBuffer bufferResult;
    private uint[] queryResult = new uint[2];
    private int numberOfDispatches = 3; // Multiple dispatches
    private AsyncGPUReadbackRequest readbackRequest;
     private string shaderTimerName = "Compute Shader";
     private UnityEngine.Profiling.CustomSampler customSampler;
        private UnityEngine.Profiling.CustomSampler gpuSampler;
    private GraphicsQuery timerQuery;
    private bool timerPending = false;

    void Start()
    {
        kernelId = computeShader.FindKernel("CSMain");
        bufferA = new ComputeBuffer(dataSize, sizeof(int));
        bufferB = new ComputeBuffer(dataSize, sizeof(int));
        bufferResult = new ComputeBuffer(dataSize, sizeof(int));
        int[] data = new int[dataSize];
        for (int i = 0; i < dataSize; i++)
        {
            data[i] = i;
        }
        bufferA.SetData(data);
        bufferB.SetData(data);
           customSampler = UnityEngine.Profiling.CustomSampler.Create(shaderTimerName);
        timerQuery = Graphics.CreateQuery(GraphicsQueryType.TimeElapsed);
            gpuSampler = UnityEngine.Profiling.CustomSampler.Create("GPU Time");
    }

    void Update()
    {
     if (timerPending && readbackRequest.done){
            readbackRequest.GetData(queryResult);
             float gpuTime = (queryResult[1] - queryResult[0]) / 1000000.0f; // Convert to milliseconds.
            gpuSampler.End();
           Debug.Log("GPU Time: " + gpuTime.ToString() + "ms");
              timerPending = false;
        }
        if (!timerPending){
              customSampler.Begin();
             gpuSampler.Begin();
             timerPending = true;
             DispatchComputeShaders();
              customSampler.End();
             readbackRequest = timerQuery.RequestAsyncReadback(queryResult.Length * sizeof(uint));
         }
    }

    void DispatchComputeShaders()
    {
        Graphics.BeginSample("Multi Compute Shader Dispatch");
        timerQuery.Begin();
       for (int i = 0; i < numberOfDispatches; i++)
        {
           computeShader.SetBuffer(kernelId, "bufferA", bufferA);
            computeShader.SetBuffer(kernelId, "bufferB", bufferB);
            computeShader.SetBuffer(kernelId, "bufferResult", bufferResult);
            computeShader.Dispatch(kernelId, dataSize / 64, 1, 1);
        }
        timerQuery.End();
        Graphics.EndSample();
    }

      void OnDestroy()
    {
        bufferA.Dispose();
        bufferB.Dispose();
        bufferResult.Dispose();
        timerQuery.Dispose();
    }
}
```

Here, the `DispatchComputeShaders` method issues three dispatch calls inside a loop, all under the same timing query. This allows for measuring the total execution time of multiple consecutive compute dispatches. The results, similar to the previous example, are retrieved asynchronously after the GPU processing is complete.

For further exploration and deeper understanding of Unity's profiling capabilities, I recommend consulting Unity's official documentation on the Profiler and Frame Debugger tools. Additional resources that can prove beneficial are the Unity manual sections covering Graphics APIs, compute shaders, and performance optimization techniques. These resources can help you understand how the render pipeline works and where your shaders sit in that pipeline. The Unity API documentation is particularly important for staying current with the most up-to-date methods for timing GPU operations.
