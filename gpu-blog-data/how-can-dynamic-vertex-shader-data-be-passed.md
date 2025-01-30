---
title: "How can dynamic vertex shader data be passed to a WGPU shader?"
date: "2025-01-30"
id: "how-can-dynamic-vertex-shader-data-be-passed"
---
Within the realm of real-time graphics rendering, directly modifying vertex attributes on a per-frame basis via the CPU is a common bottleneck. Modern APIs like WebGPU provide mechanisms to alleviate this via buffer objects, allowing dynamic data to be accessed within shaders. The key lies in creating appropriate buffer layouts, staging data correctly, and ensuring data synchronization between the CPU and the GPU.

The fundamental problem stems from the static nature of vertex buffers when initially defined. These buffers are typically designed to hold unchanging vertex positions, normals, and other attributes which determine a mesh's shape. However, there are many scenarios where we need per-instance or per-frame changes, such as for animation, particle systems, or complex procedural effects. To accomplish this efficiently, we must create a buffer that we can periodically update, and then configure the shader to read from this dynamic source.

My experience with WebGPU has involved moving computationally expensive effects from CPU-bound calculations to the GPU. A typical scenario involves simulating a large particle system where each particle's position, velocity, and lifetime evolve each frame. Initially, I updated each particle's data on the CPU, and re-uploaded all the data to a static buffer every single frame, leading to severe performance issues. By creating a dynamic buffer and utilizing the GPU for physics calculation, I was able to offload a significant portion of the calculation, and the performance improved drastically. The process involved binding the dynamic data buffer to the rendering pipeline and accessing it within the vertex shader.

The following code examples demonstrate the process with a progressive approach.

**Example 1: Basic dynamic buffer setup.**

This example creates a basic dynamic buffer that could be used to store per-instance data. It illustrates how to define the buffer and how to update its contents. We are using a float32 array as our data, which we will reinterpret in the shader code,

```typescript
// Assuming adapter and device are acquired in the setup process
// and are within scope.

const instanceCount = 100;
const dynamicBufferData = new Float32Array(instanceCount * 4); // 4 floats per instance

const dynamicBuffer = device.createBuffer({
    size: dynamicBufferData.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    mappedAtCreation: false
});

function updateDynamicBuffer(device:GPUDevice, buffer:GPUBuffer) {
  // Simulate update per instance data
  for (let i=0; i<instanceCount; i++)
  {
    dynamicBufferData[i*4+0] = Math.random(); // offset x
    dynamicBufferData[i*4+1] = Math.random(); // offset y
    dynamicBufferData[i*4+2] = Math.random(); // scale
    dynamicBufferData[i*4+3] = Math.random(); // time offset
  }


  device.queue.writeBuffer(
      buffer,
      0,
      dynamicBufferData,
      0,
      dynamicBufferData.byteLength
  );

}

```

Here, `instanceCount` represents the number of instances that will be affected by our dynamic data. The buffer `dynamicBufferData` holds the actual data that we’ll write to the GPU. The `createBuffer` call specifies the size, usage flags, and ensures that the buffer is not initially mapped (meaning its initial data is garbage on the GPU). The  `updateDynamicBuffer` shows how the data can be updated and written to the buffer using a device queue. Notice the `GPUBufferUsage.COPY_DST` flag is used, as it allows the device queue to write to this buffer, and `GPUBufferUsage.VERTEX` flag is used as the data will be used in vertex stage. The `writeBuffer` copies data from the CPU to the GPU, preparing it for use in the vertex shader. This process is executed each frame, or whenever the dynamic data changes.

**Example 2: Integrating the dynamic data into a vertex shader.**

This example demonstrates how to access the data provided by the dynamic buffer using shader code,

```wgsl
struct VertexInput {
    @location(0) position: vec4<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) @builtin(instance_index) instanceIndex: u32,
}

struct InstanceData {
  offset_x : f32,
  offset_y : f32,
  scale : f32,
  time_offset : f32,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@group(0) @binding(0) var<storage, read> instance_data: array<InstanceData>; // Storage buffer

@vertex
fn vs(input: VertexInput) -> VertexOutput {

    let instance_info : InstanceData = instance_data[input.instanceIndex];

    var output : VertexOutput;
    output.uv = input.uv;

    let offset : vec2<f32> = vec2(instance_info.offset_x,instance_info.offset_y);
    let scaledPos: vec4<f32> = input.position * instance_info.scale;
    output.position = scaledPos + vec4(offset.x,offset.y,0,0);

    return output;
}
```

Here, the shader defines `VertexInput` including the `instanceIndex`, and `InstanceData` which represents how we are interpreting each set of four floats from our dynamic buffer. The `instance_data` is a `storage` buffer which is indexed via the `instanceIndex` which is part of the Vertex Input. The shader accesses the appropriate data by using the built-in `instance_index` to perform a lookup within the `storage` buffer. For each vertex, it computes a position based on the original vertex position and applies offset based on the data, which demonstrates one possible usage of the per instance data.  This shows how the dynamic data is retrieved from the buffer and applied to each instance of geometry.

**Example 3: Setting up the render pipeline.**

This example completes the process by showcasing how to bind a buffer within the context of a render pipeline.

```typescript
// Assuming adapter, device and shaderModule are within scope.

//Create buffer layouts and bind groups
const instanceDataBufferLayout: GPUBufferBindingLayout = {
    type : "read-only-storage" // read only storage as we don't write back to the buffer.
};


const bindGroupLayout = device.createBindGroupLayout({
    entries: [
        {
            binding: 0,
            visibility: GPUShaderStage.VERTEX,
            buffer: instanceDataBufferLayout,
        },
    ],
});

const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries:[{
        binding: 0,
        resource:{
           buffer: dynamicBuffer
        }
    }]
});


//assuming vertex buffer and render target are previously defined.

const pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [bindGroupLayout],
});

const renderPipeline = device.createRenderPipeline({
    layout: pipelineLayout,
    vertex: {
        module: shaderModule, //Shader module from previous examples.
        entryPoint: "vs",
        buffers: [
            {
                arrayStride: 20, // 3 floats for position + 2 for uv = 5 floats * 4 bytes per float
                attributes: [
                    {
                        shaderLocation: 0,
                        offset: 0,
                        format: "float32x3",
                    },
                   {
                       shaderLocation :1,
                       offset : 12,
                       format : "float32x2"
                   }
                ],
            },
        ],
    },
    fragment: {
        module: shaderModule,
        entryPoint: "fs",
        targets: [{ format: "bgra8unorm" }], // assuming bgra8unorm format.
    },
    primitive:{
        topology:"triangle-list"
    }
});


function render(device:GPUDevice, encoder:GPUCommandEncoder, textureView:GPUTextureView, dynamicBuffer:GPUBuffer, vertexBuffer:GPUBuffer) {

     updateDynamicBuffer(device,dynamicBuffer);

     const renderPass = encoder.beginRenderPass({
          colorAttachments:[{
             view : textureView,
             clearValue : [0,0,0,1],
             loadOp: "clear",
             storeOp: "store"
          }]
     });

     renderPass.setPipeline(renderPipeline);
     renderPass.setBindGroup(0, bindGroup);
     renderPass.setVertexBuffer(0, vertexBuffer); //assuming vertex buffer defined elsewhere
     renderPass.draw(6,instanceCount); //draw 6 vertices for a quad, and use instanceCount.
     renderPass.end();
}
```

In this snippet, `device.createBindGroupLayout` creates the description for how a buffer will be used in the shader, in this case, as a read-only storage buffer visible in the vertex stage. `device.createBindGroup` ties the buffer and the bind group layout, and is passed into the `renderPass` during render call. We have to provide the appropriate layout during the pipeline creation to allow the device to know the structure of our buffers.  The render function updates the buffer each frame, and then during the `renderPass`, sets the buffer before rendering the geometry. Note that the `draw` call uses the `instanceCount` to draw all our instances using the per-instance information. This is the final piece of the puzzle to bring the system together and draw our geometry with the offset and scale applied.

**Resource Recommendations:**

For deepening your understanding of WebGPU buffer management, I recommend exploring the official WebGPU specification. Look for the sections that discuss buffer creation, buffer usages, memory mapping, and the specific details on the data layout expected by vertex shaders. A good grasp of the rendering pipeline and its stages is also important. In addition, resources that delve into GPU memory management are beneficial, as this knowledge directly influences efficient dynamic buffer handling. Furthermore, examining the various buffer usage flags such as `COPY_DST` and `VERTEX` is important to understand what the device is able to do with the buffer. Lastly, tutorials and examples from the WebGPU community can offer practical insights into more complex scenarios involving dynamic data. These resources should provide the necessary knowledge for any further projects.
