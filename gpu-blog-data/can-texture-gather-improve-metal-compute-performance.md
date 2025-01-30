---
title: "Can texture gather improve metal compute performance?"
date: "2025-01-30"
id: "can-texture-gather-improve-metal-compute-performance"
---
Yes, texture gather can significantly improve compute performance in certain metal scenarios, particularly when dealing with spatially correlated data. I've seen this firsthand while working on particle-based simulations using Metal; the performance gains from utilizing texture gathers instead of repeated texture fetches were substantial. The key advantage lies in how texture gathers leverage hardware-accelerated data access patterns, reducing the overall memory bandwidth bottleneck common in GPU computations.

Specifically, a texture gather operation, in Metal’s context, allows a fragment shader to sample multiple texels (typically four, forming a 2x2 region) from a texture around a specified coordinate in a single operation. The hardware fetches these neighboring texels in an optimized manner, often utilizing cache locality effectively. This contrasts with the less efficient approach of individually fetching each texel using multiple texture sampling calls. The impact becomes particularly noticeable when the same neighboring values are needed by many threads or fragments executing in parallel, such as during a convolution filter or when calculating local properties based on adjacent data points. In scenarios without texture gather, these reads would each trigger a separate memory fetch, leading to severe performance degradation due to increased latency and memory bus saturation.

Therefore, the improvement is not just about fewer lines of code; it's about optimizing memory access patterns in the highly parallel environment of the GPU. We're essentially moving from a dispersed read pattern (multiple independent texture fetches) to a consolidated one (a single gather operation), thereby leveraging hardware-specific design principles to minimize memory latency.

Consider a scenario where we're implementing a simple blur effect on an image within a compute shader. Without texture gather, we might sample each neighboring pixel separately. Let’s examine this approach:

```metal
kernel void blurKernel(texture2d<float, access::read> inputTexture [[texture(0)]],
                        texture2d<float, access::write> outputTexture [[texture(1)]],
                        uint2 gid [[thread_position_in_grid]]) {
    int2 pos = int2(gid);
    float blurValue = 0.0;
    int blurRadius = 1;
    int samples = 0;

    for (int y = -blurRadius; y <= blurRadius; ++y) {
        for (int x = -blurRadius; x <= blurRadius; ++x) {
            int2 samplePos = pos + int2(x, y);
            if (samplePos.x >= 0 && samplePos.x < inputTexture.get_width() &&
                 samplePos.y >= 0 && samplePos.y < inputTexture.get_height()) {
                    blurValue += inputTexture.read(samplePos).r;
                    samples++;
            }

        }
    }

    blurValue /= (float)samples;
    outputTexture.write(float4(blurValue, 0.0, 0.0, 1.0), pos);
}

```

In this code, we loop through neighboring pixel coordinates and perform a `read` for each, accumulating their red channel values for the blur. While conceptually straightforward, the numerous `inputTexture.read()` calls per pixel incur significant overhead. Each read operation translates to an individual memory fetch, which is not optimal for performance. This type of code results in a bandwidth-intensive operation.

Now, let's contrast this with an approach utilizing texture gather:

```metal
kernel void blurKernelGather(texture2d<float, access::read> inputTexture [[texture(0)]],
                             texture2d<float, access::write> outputTexture [[texture(1)]],
                             uint2 gid [[thread_position_in_grid]]) {
    int2 pos = int2(gid);
    float2 uv = float2(float(pos.x) + 0.5f, float(pos.y) + 0.5f) / float2(inputTexture.get_width(), inputTexture.get_height());

    float4 gatheredValues = inputTexture.gather(texture::coord::pixel, uv);
    float blurValue = (gatheredValues.x + gatheredValues.y + gatheredValues.z + gatheredValues.w) / 4.0;

    outputTexture.write(float4(blurValue, 0.0, 0.0, 1.0), pos);
}
```

Here, we use `inputTexture.gather(texture::coord::pixel, uv)`. This single function call returns a `float4` containing the red channel values from the four texels surrounding the specified coordinate. The `texture::coord::pixel` argument specifies that the gather operation is to be performed in pixel coordinates. We then average the four returned values and write the result to the output texture. This second method executes fewer texture access operations and offloads the neighbor fetch logic to the GPU hardware level. The memory access pattern is now coalesced, leading to a significant performance improvement, particularly noticeable with large textures and complex blur operations.

The performance boost derived from texture gather is not uniform across all applications and textures. It’s highly beneficial when:

*   **Spatial Coherence Exists:** The data being accessed is spatially correlated, as adjacent data points are relevant for calculations.
*   **Many Parallel Operations Need Neighbor Values:**  Numerous threads or fragments need the same neighboring values, leading to redundancy if each accesses them individually.
*   **Bandwidth Limitation is a Bottleneck:** Memory access patterns are the primary constraint on performance, not arithmetic calculations.

Another scenario highlighting the utility of texture gather is in calculating the normals of a heightmap. We typically require a directional derivative to approximate the surface normal at a point. We can obtain this derivative by analyzing neighboring height values. Let's implement this using a compute shader:

```metal
kernel void calculateNormals(texture2d<float, access::read> heightMap [[texture(0)]],
                              texture2d<float, access::write> normalMap [[texture(1)]],
                              uint2 gid [[thread_position_in_grid]]) {

    int2 pos = int2(gid);
    float2 uv = float2(float(pos.x) + 0.5f, float(pos.y) + 0.5f) / float2(heightMap.get_width(), heightMap.get_height());

    // Gather height values
    float4 heights = heightMap.gather(texture::coord::pixel, uv).xxxx;
    // Calculate approximate height gradients using gathered height values.
    float dx = (heights.y - heights.x) ;
    float dy = (heights.w - heights.x);

    float3 normal = normalize(float3(-dx, 1, -dy));

    normalMap.write(float4(normal, 0.0), pos);
}

```

In this example, the gather operation retrieves the height values of the four texels forming a square around the pixel at `uv`. Using these heights, we estimate derivatives and construct a normal vector.  This approach is significantly more efficient than manually sampling neighboring height values individually and results in more performant normal map generation, an important step in real-time rendering.

It's crucial to understand the limitations. Texture gather typically only works with 2x2 texel regions, although some hardware extensions may support different layouts. Gather operations only return the red component of gathered texels. Consequently, if we require other components of those texels, separate texture fetches may be unavoidable for those non-red components. Despite such limitations, in situations where data exhibit the necessary spatial coherence and when sampling multiple neighbors is a common task, the performance improvement derived from texture gather can be considerable.

To further enhance your understanding, I would recommend reviewing the official Metal Shading Language documentation, specifically sections discussing texture access and sampling.  In addition, Apple's developer documentation on Metal provides numerous articles covering optimization techniques for compute and graphics pipelines, often including usage examples related to texture access.  Finally, experimenting with different use cases yourself will always remain the most effective method to gain hands-on experience and deepen your comprehension of these performance optimizations.
