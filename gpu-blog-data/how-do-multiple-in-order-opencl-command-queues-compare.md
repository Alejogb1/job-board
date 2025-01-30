---
title: "How do multiple in-order OpenCL command queues compare to a single out-of-order queue for performance?"
date: "2025-01-30"
id: "how-do-multiple-in-order-opencl-command-queues-compare"
---
OpenCL’s asynchronous execution model presents a critical performance trade-off between in-order and out-of-order command queues, influencing how kernels and memory transfers are processed on the underlying compute devices. My experience developing high-performance image processing pipelines using OpenCL highlights that while multiple in-order command queues might appear intuitively advantageous for task parallelism, a carefully managed single out-of-order queue often yields superior throughput, particularly when considering complex dependency graphs.

Fundamentally, an in-order command queue guarantees that commands are executed in the exact sequence they are enqueued. This strict ordering provides simplicity in reasoning about execution flow. If kernel A is enqueued before kernel B, we are assured that kernel A completes before kernel B begins execution on the compute device. In the context of multiple in-order queues, one might use different queues to logically separate work streams, implying a form of implicit parallelization at the command queue level. For example, a first queue might handle pre-processing steps, while another processes image convolutions. However, this method is fundamentally limited by the hardware’s ability to manage multiple active command queues, and also suffers the overhead of managing multiple queues, often involving explicit synchronization between them to maintain correct sequencing of operations across logical process boundaries.

In contrast, an out-of-order command queue allows the OpenCL runtime to schedule enqueued commands based on their data dependencies, rather than their enqueuing order. If kernel B requires the output of kernel A, the runtime will delay B's execution until A completes, regardless of whether B was enqueued before A. This allows for significant performance improvements through effective scheduling of computation and memory transfers, overlapping them to maximize device utilization. The challenge lies in explicitly managing the data dependencies such that the runtime has the necessary information to schedule efficiently.

The primary benefit of an out-of-order queue is the ability to overlap execution and memory transfers. With in-order queues, any transfer that is started blocks subsequent kernel execution until it's complete, even if the kernel could be launched on a different part of the device. Similarly, kernel execution blocks transfers, wasting potential parallelism. An out-of-order queue allows the scheduler to look ahead and overlap these operations. Consider a processing pipeline involving numerous kernels operating on multiple buffers. In an out-of-order queue, the runtime could begin a transfer operation for one buffer while the GPU is still working on other buffer-related kernels, maximizing the use of different parts of the device.

Let's consider three examples to illuminate these differences:

**Example 1: Simple Image Processing Pipeline using Multiple In-Order Queues**

```c
// Setup two in-order command queues
cl_command_queue queue_preprocess = clCreateCommandQueue(context, device, 0, &err);
cl_command_queue queue_process = clCreateCommandQueue(context, device, 0, &err);

// Kernel declarations
cl_kernel kernel_grayscale = clCreateKernel(program, "grayscale", &err);
cl_kernel kernel_blur = clCreateKernel(program, "blur", &err);

// Image buffer creation (assume buffers input_image and output_image are created)

// Pre-processing: convert to grayscale
clEnqueueNDRangeKernel(queue_preprocess, kernel_grayscale, 2, NULL, global_size, local_size, 0, NULL, NULL);
clEnqueueCopyBuffer(queue_preprocess, input_image, buffer_intermediate, 0, 0, input_image_size, 0, NULL, NULL); // copy to another buffer

clFinish(queue_preprocess); // Barrier for synchronization

// Processing: blur the grayscale image
clEnqueueNDRangeKernel(queue_process, kernel_blur, 2, NULL, global_size, local_size, 0, NULL, NULL);
clEnqueueCopyBuffer(queue_process, buffer_intermediate, output_image, 0, 0, output_image_size, 0, NULL, NULL);

clFinish(queue_process); // Barrier for synchronization
```

In this example, two command queues, `queue_preprocess` and `queue_process` are used. One to convert image to grayscale, and one to blur the grayscale image. The call to `clFinish` after processing each queue introduces a barrier for synchronization. Even though the hardware *could* potentially start a memory transfer associated with the second queue while the first queue is executing kernels, it won't do so here because each queue has to complete in sequence. This approach makes reasoning easier to reason about at a program level, however, it's performance is severely limited.

**Example 2: Equivalent Image Processing Pipeline using a Single Out-of-Order Queue**

```c
// Setup a single out-of-order command queue
cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);

// Kernel declarations (same as example 1)
cl_kernel kernel_grayscale = clCreateKernel(program, "grayscale", &err);
cl_kernel kernel_blur = clCreateKernel(program, "blur", &err);


// Image buffer creation (same as example 1)

// Pre-processing: convert to grayscale
cl_event event_grayscale;
clEnqueueNDRangeKernel(queue, kernel_grayscale, 2, NULL, global_size, local_size, 0, NULL, &event_grayscale);
cl_event event_copy_to_intermediate;
clEnqueueCopyBuffer(queue, input_image, buffer_intermediate, 0, 0, input_image_size, 0, NULL, &event_copy_to_intermediate); // copy to another buffer

// Processing: blur the grayscale image
cl_event event_blur;
clEnqueueNDRangeKernel(queue, kernel_blur, 2, NULL, global_size, local_size, 1, &event_copy_to_intermediate, &event_blur); // wait for copy before blur
clEnqueueCopyBuffer(queue, buffer_intermediate, output_image, 0, 0, output_image_size, 1, &event_blur, NULL);

clFinish(queue); // Barrier for synchronization
```

Here, a single out-of-order command queue, `queue`, is employed. Key to this method is the usage of OpenCL events, such as `event_copy_to_intermediate` and `event_blur`, and using them to specify data dependency. The `clEnqueueNDRangeKernel` of `kernel_blur` depends on the buffer copy, because the kernel uses the intermediate buffer. The runtime can now execute kernels concurrently where they don't depend on each other, and overlap memory transfers and compute kernel operations to maximize utilization of the hardware. Although more complexity is involved with the explicit event management, there is significant benefit in performance.

**Example 3: Handling Complex Dependencies in a Single Out-of-Order Queue**

```c
// Setup a single out-of-order command queue (same as example 2)
cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);

// Kernel declarations
cl_kernel kernel_process_a = clCreateKernel(program, "process_a", &err);
cl_kernel kernel_process_b = clCreateKernel(program, "process_b", &err);
cl_kernel kernel_process_c = clCreateKernel(program, "process_c", &err);
cl_kernel kernel_process_d = clCreateKernel(program, "process_d", &err);


// Buffer creation (assume buffers buffer_A, buffer_B, buffer_C, buffer_D, result_buffer are created)

// Enqueue kernels with complex dependencies
cl_event event_A;
clEnqueueNDRangeKernel(queue, kernel_process_a, 1, NULL, global_size, local_size, 0, NULL, &event_A);

cl_event event_B;
clEnqueueNDRangeKernel(queue, kernel_process_b, 1, NULL, global_size, local_size, 0, NULL, &event_B);

cl_event event_C;
clEnqueueNDRangeKernel(queue, kernel_process_c, 1, NULL, global_size, local_size, 1, &event_A, &event_C); // C depends on A

cl_event event_D;
clEnqueueNDRangeKernel(queue, kernel_process_d, 1, NULL, global_size, local_size, 2, (cl_event[]){event_B, event_C}, &event_D); // D depends on B and C

clEnqueueCopyBuffer(queue, buffer_D, result_buffer, 0, 0, buffer_size, 1, &event_D, NULL);
clFinish(queue);
```

In this more complicated example, four kernels, `kernel_process_a`, `kernel_process_b`, `kernel_process_c`, `kernel_process_d`, are executed in a pipeline with complex dependency, showing the importance of `event` usage. Kernel C depends on A, and kernel D depends on both B and C. The out-of-order queue correctly resolves these dependencies based on the events and their ordering. The runtime can therefore execute A, B and later C and D in the order necessary, or potentially concurrently if hardware resources allow, while maintaining correct execution. This would be extremely difficult to orchestrate using multiple in-order queues without introducing large and unnecessary synchronization barriers.

From these examples, I have observed that while multiple in-order queues offer simplicity, the single out-of-order queue, with careful dependency management, generally offers better performance potential due to the increased flexibility in scheduling operations. The primary source of this performance gain stems from overlapping of memory transfers and compute kernels where dependencies allow. It is also important to note that the complexity of the dependency graph within the workload has a large impact on the benefits of using out-of-order queues. For simple workloads with few dependencies, the difference may be negligible. However, for more complex workloads, the differences can be significant.

To fully understand these trade-offs, I recommend exploring resources that detail OpenCL's asynchronous command execution model, particularly: the official OpenCL specification documents published by Khronos, which contain all the necessary details; books that provide practical examples on the optimization of OpenCL kernels and memory transfers; vendor-specific documentation of OpenCL implementations that can offer detailed information on how specific hardware handles command scheduling and dependency resolution. I also found research papers focusing on task scheduling algorithms in GPU programming to be very useful. Analyzing benchmark results of complex kernels with different queue models can also offer useful insights.
