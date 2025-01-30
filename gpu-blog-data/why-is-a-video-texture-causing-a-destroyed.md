---
title: "Why is a video texture causing a 'destroyed texture' error in Chrome/WebGPU submit?"
date: "2025-01-30"
id: "why-is-a-video-texture-causing-a-destroyed"
---
When a video texture exhibits a "destroyed texture" error in Chrome/WebGPU during submit, it signals a critical synchronization issue between the video frame source and the GPU texture resource. Specifically, this error occurs when the WebGPU pipeline attempts to utilize a texture that has already been invalidated or released, typically due to the video element's lifecycle or an unexpected update in the video stream itself. This differs from more straightforward texture loading errors; it's about timing and management. Having debugged similar issues in the context of high-performance interactive video art installations, I've found this problem often boils down to the interplay between JavaScript's asynchronous operations and the GPU's buffered command execution.

The root cause lies in WebGPU’s deferred rendering model and its management of texture resources. Unlike immediate-mode APIs, WebGPU relies on a command buffer system. Your application submits commands, and the GPU executes these commands later. In the case of video textures, you typically create a WebGPU texture object that is dynamically updated with the current frame from a `<video>` element. This is where the fragility arises.

The sequence usually follows these steps: First, a WebGPU texture is created, often initialized with a single frame from the video element at instantiation. Second, the application triggers a rendering pass that uses this texture in shaders and other processing. Third, the application, via a `requestAnimationFrame` loop or similar, updates the texture by copying a new video frame. This copying process might use `copyExternalImageToTexture`.

The problem manifests when the video element's state or playback cycle is disrupted *during* a rendering pass, and the texture is no longer valid. A common scenario is the video looping or pausing/playing asynchronously within the requestAnimationFrame loop, before WebGPU executes all of its prior commands that are still referencing an older version of the texture. When WebGPU reaches the command that uses the now "destroyed" texture, it throws the "destroyed texture" error.

The error does not mean the texture object itself is destroyed on the JavaScript side; it means the *backing memory* on the GPU is no longer available. This often occurs in cases like: the video changes source, the video restarts its playback after reaching the end, a resource is re-allocated, or even a very quick succession of updates. Crucially, the video element’s update can happen *before* or *after* the WebGPU rendering command utilizing the older texture, leading to inconsistent behaviour depending on timing. Because command buffer execution is not instantaneous, the render commands are not immediately synchronized with the video element’s update.

Here are a few code examples demonstrating common scenarios and solutions:

**Example 1: Naive Update Without Synchronization**

This example shows a problematic scenario where texture updates are applied without explicitly waiting for the rendering queue to complete using await. Note that while the Javascript API `copyExternalImageToTexture` may return after it has submitted the texture update to the queue, it does not imply it has been processed by the GPU yet.

```javascript
async function render() {
    const videoFrame = videoElement; // Assuming videoElement is a <video> element
    device.queue.copyExternalImageToTexture(
        { source: videoFrame },
        { texture: videoTexture, origin: [0, 0] },
        [videoFrame.videoWidth, videoFrame.videoHeight]
    );

    renderPass.beginRenderPass();
    renderPass.setPipeline(renderPipeline);
    renderPass.setBindGroup(0, bindGroup);
    renderPass.draw(3);
    renderPass.endRenderPass();

    device.queue.submit([commandEncoder.finish()]);
    requestAnimationFrame(render);
}
```

*Commentary*: In this simplified example, `copyExternalImageToTexture` updates the texture with a new frame within each `render` call, but these updates can happen asynchronously with the render pass using the previous texture. Because the `device.queue.submit` is not awaited, and because the `device.queue.copyExternalImageToTexture` is not an explicit synchronization mechanism (it simply copies the frame into a queue), the GPU may process the render commands before the texture is updated, or *after* a subsequent texture update. This leads to a race condition that can result in the "destroyed texture" error under certain timing constraints, like when the video loops or rapidly changes. This is not a guaranteed error every single frame, but occurs under specific conditions.

**Example 2: Using `submit` and Explicit Synchronization**

This example includes an `await` to ensure that previous render passes are resolved before a new texture update is triggered, as well as implementing an explicit fence mechanism.

```javascript
async function render() {
  const videoFrame = videoElement; // Assuming videoElement is a <video> element

  await device.queue.onSubmittedWorkDone();

    device.queue.copyExternalImageToTexture(
        { source: videoFrame },
        { texture: videoTexture, origin: [0, 0] },
        [videoFrame.videoWidth, videoFrame.videoHeight]
    );


    renderPass.beginRenderPass();
    renderPass.setPipeline(renderPipeline);
    renderPass.setBindGroup(0, bindGroup);
    renderPass.draw(3);
    renderPass.endRenderPass();

    device.queue.submit([commandEncoder.finish()]);
    requestAnimationFrame(render);

}
```

*Commentary:* The key addition here is the `await device.queue.onSubmittedWorkDone();`. This forces JavaScript to pause until all previously submitted commands are completed by the GPU before updating the video texture and rendering a new frame. This creates an explicit synchronization point, eliminating the race condition described previously, but at the cost of potentially reduced performance in some scenarios. This will force the Javascript thread to wait for all previously rendered frames to complete on the GPU, and if the GPU is taking a while, this will stall the main Javascript thread.

**Example 3: Double Buffering and Fence Object Synchronization**

This more robust implementation of texture updates uses two texture objects and double buffers rendering. In some cases, double buffering is not possible to guarantee no errors because there is often the initial setup of one texture for rendering before the second one is setup and initialized, and this setup may be the source of errors still. In other words, if a video changes sources and you want to setup a new texture, both the primary and secondary texture objects may be destroyed as they must be resized. For a more robust solution one would use "triple" buffering which would involve three textures, and the setup would only occur while the application is rendering to a texture that is not being updated.

```javascript
let currentTextureIndex = 0;
const videoTextures = [videoTexture1, videoTexture2]; // Assuming two texture objects

async function render() {
    const videoFrame = videoElement;

    const currentTexture = videoTextures[currentTextureIndex];
    const nextTextureIndex = 1 - currentTextureIndex;
    const nextTexture = videoTextures[nextTextureIndex];

    device.queue.copyExternalImageToTexture(
        { source: videoFrame },
        { texture: nextTexture, origin: [0, 0] },
        [videoFrame.videoWidth, videoFrame.videoHeight]
    );

  await device.queue.onSubmittedWorkDone();

    renderPass.beginRenderPass();
    renderPass.setPipeline(renderPipeline);
    // Update the bind group to use the current texture
    renderPass.setBindGroup(0, currentTexture === videoTexture1 ? bindGroup1 : bindGroup2);
    renderPass.draw(3);
    renderPass.endRenderPass();

    device.queue.submit([commandEncoder.finish()]);


    currentTextureIndex = nextTextureIndex;
    requestAnimationFrame(render);
}
```

*Commentary:* This advanced technique utilizes two textures. While the GPU renders with the “current” texture, the next frame is copied into the “next” texture. Then, the textures switch roles. By using two textures (double-buffering) and an explicit fence using the `await` to wait for work to complete, the code avoids the issue of texture invalidation by always writing the next frame to an unused texture.  The downside to this is an increase in memory usage, and if a texture's source is changing rapidly, the application must destroy the old textures, create and setup the new textures, before the application can start rendering again, which can be another source of errors.

For effective debugging and management, I recommend these resources: The official WebGPU specification documents provide a deep understanding of the underlying API. The Chrome DevTools WebGPU inspector is essential for inspecting live resources, especially to see the state of the textures during execution. Lastly, researching practical examples of WebGPU video rendering can provide valuable solutions and insights into best practices.
