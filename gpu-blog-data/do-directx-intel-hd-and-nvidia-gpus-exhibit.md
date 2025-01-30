---
title: "Do DirectX Intel HD and NVIDIA GPUs exhibit different geometry shader behavior?"
date: "2025-01-30"
id: "do-directx-intel-hd-and-nvidia-gpus-exhibit"
---
The fundamental difference in shader compilation and execution pathways between Intel HD Graphics and NVIDIA GPUs can lead to observable variations in geometry shader behavior, particularly concerning resource allocation, thread divergence, and performance characteristics. My experience porting a complex fluid simulation from a workstation equipped with an NVIDIA RTX card to a laptop using integrated Intel HD Graphics highlighted these disparities.

Specifically, the Intel HD integrated graphics often exhibit more constrained resource management for geometry shaders compared to dedicated NVIDIA GPUs. Geometry shaders operate after the vertex shader stage, capable of generating new primitives—points, lines, or triangles—based on incoming vertex data. This amplification process demands significant memory bandwidth and processing power. Intel HD solutions, typically sharing system RAM, tend to have a more limited pool of resources, potentially leading to bottlenecks when geometry shader outputs become substantial. This limitation can translate into noticeable performance degradation or even shader execution failures under heavy load.

Furthermore, divergence within threads in a geometry shader, where different threads executing the same shader code take distinct paths due to conditional branching, can disproportionately impact performance on Intel HD graphics. NVIDIA GPUs, with their more sophisticated thread scheduling and warp architecture, generally handle divergent execution with greater efficiency. On Intel HD, this divergence may lead to a higher number of idle threads, essentially wasting computation capacity. In essence, code which works effectively and efficiently on an NVIDIA GPU may encounter issues concerning speed and resource limits on the integrated graphics.

The observed differences also extend to implicit behaviors of the shader compiler and driver. Subtle variations in the optimizations applied by Intel and NVIDIA's respective graphics drivers, particularly relating to the allocation of varying registers and the generation of intermediate data, can lead to variances in performance and even behavior. For instance, where a geometry shader relies heavily on temporary variables, Intel's shader compiler might use register spilling more aggressively than an NVIDIA compiler, affecting overall throughput.

These differences aren't typically about whether a given shader will function or not, given they are compiled according to DirectX requirements; rather, it concerns how efficiently and correctly the driver will perform the operations that are implied by the compiled shader program.

To illustrate these nuances, consider a simplified geometry shader that expands a single triangle into multiple triangles forming a 2D grid:

**Example 1: Simple Triangle Grid Generation**

```hlsl
#version 450

layout(triangles) in;
layout(triangle_strip, max_vertices = 16) out;

in vec3 inPosition[];
out vec3 outPosition;

void main() {
    int gridSize = 3;
    for (int i = 0; i <= gridSize; i++) {
        for (int j = 0; j <= gridSize; j++) {
            float u = float(i) / float(gridSize);
            float v = float(j) / float(gridSize);
            vec3 p = inPosition[0] + (inPosition[1] - inPosition[0]) * u + (inPosition[2] - inPosition[0]) * v;
            outPosition = p;
            gl_Position = gl_in[0].gl_Position;
            EmitVertex();
        }
       EndPrimitive();
    }
}
```

This straightforward shader takes a single triangle and tessellates it, using a nested loop structure to form a grid of triangles. On a high-performance NVIDIA card, this shader, even with a large *gridSize* value, runs relatively smoothly. However, on Intel HD integrated graphics, increasing *gridSize* significantly can cause a pronounced performance drop. This occurs partly because each invocation of `EmitVertex()` can create new vertices within a vertex buffer, and on an Intel integrated GPU, this operation can be resource-constrained, affecting the shader performance due to memory access overhead. The amount of memory that must be accessed and moved becomes the primary restriction for efficient execution.

**Example 2: Introducing Branching**

```hlsl
#version 450

layout(triangles) in;
layout(triangle_strip, max_vertices = 16) out;

in vec3 inPosition[];
out vec3 outPosition;
out vec4 outColor;

uniform float threshold;

void main() {
    int gridSize = 3;
    for (int i = 0; i <= gridSize; i++) {
        for (int j = 0; j <= gridSize; j++) {
            float u = float(i) / float(gridSize);
            float v = float(j) / float(gridSize);
            vec3 p = inPosition[0] + (inPosition[1] - inPosition[0]) * u + (inPosition[2] - inPosition[0]) * v;
            outPosition = p;

            //Introduce divergence using a threshold:
            if (distance(p, inPosition[0]) > threshold)
                outColor = vec4(1.0, 0.0, 0.0, 1.0);  //Red
            else
                outColor = vec4(0.0, 1.0, 0.0, 1.0);  //Green

            gl_Position = gl_in[0].gl_Position;
            EmitVertex();
        }
        EndPrimitive();
    }
}
```

In this modified example, a conditional branch based on the distance of the generated vertex to the initial position is introduced. Such branching increases the execution path diversity. On NVIDIA, with its efficient warp processing capabilities, the cost of this divergence is minimized. But on Intel HD, a significant performance penalty may arise, as the threads might exhibit a serialized execution pattern for divergent blocks, reducing the efficiency of parallel processing. This problem is further accentuated by a higher *gridSize* value, as more branching threads need to be managed by the shader execution pipeline.

**Example 3: Geometry Amplification**

```hlsl
#version 450

layout(triangles) in;
layout(triangle_strip, max_vertices = 32) out;

in vec3 inPosition[];
out vec3 outPosition;

uniform int amplificationFactor;

void main() {
    for(int k = 0; k < amplificationFactor; k++)
    {
      for(int i=0; i < 3; i++) {
          outPosition = inPosition[i];
          gl_Position = gl_in[0].gl_Position;
          EmitVertex();
        }
        EndPrimitive();
    }
}
```

This final example amplifies the input triangle geometry by simply emitting it several times defined by an amplification factor. While simple, on a system with shared resources like those of integrated graphics, the amplification creates a heavier memory access burden than would be felt on a system with more resources. This highlights how a seemingly benign geometry amplification process, when combined with limited memory resources available in integrated graphics solutions, can lead to significant slowdowns and potential resource exhaustion. The simple process of writing each triangle to memory becomes a significant bottleneck.

In conclusion, differences in architecture and driver implementation between Intel HD and NVIDIA GPUs result in noticeable variations in geometry shader behavior. Intel HD GPUs are more susceptible to resource constraints, thread divergence, and other performance penalties associated with geometry shader operations. To mitigate these issues, developers targeting a broad range of hardware need to consider techniques like pre-tessellation on the CPU, reduction of geometry complexity where possible, or using compute shaders to perform certain geometry operations instead of relying on the geometry shader pipeline, as these can circumvent some of the limitations encountered on integrated graphics solutions. Furthermore, careful profiling of shader performance on a range of platforms can reveal and help correct issues. Additionally, utilizing profiling tools for both platforms can identify bottlenecks and help to optimize the execution flow for different GPUs.

For further information, I recommend exploring documentation on the DirectX graphics pipeline, particularly concerning the geometry shader stage. I would recommend researching articles detailing the architecture of integrated graphics chips and how their shader units operate and interact with system memory. Additionally, the performance analysis guides for both Intel and NVIDIA GPUs provide valuable insight into best practices for shader optimization, specifically targeting each architecture.
