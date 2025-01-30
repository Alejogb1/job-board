---
title: "How can Python calculate frames per second (FPS) using a GPU?"
date: "2025-01-30"
id: "how-can-python-calculate-frames-per-second-fps"
---
The most reliable method for determining frames per second (FPS) with GPU involvement in Python necessitates monitoring the time taken for rendering operations that are heavily processed by the graphics processing unit. Unlike CPU-bound FPS calculations, where simply timing the rate of loop iterations might suffice, GPU-dependent scenarios require a deeper understanding of when rendering commands are executed and completed on the GPU itself. My own experience developing a real-time simulation using PyOpenGL involved precisely this challenge, leading me to adapt several techniques for accurate GPU-bound FPS measurement.

The fundamental issue stems from the asynchronous nature of modern GPU pipelines. When a program issues a rendering command, it often doesn't immediately block and wait for that command to be fully executed on the GPU. Instead, commands are queued, and the application thread continues execution while the GPU independently processes them. This separation makes direct CPU timing of frame rendering inaccurate, as the CPU might move on to the next frame’s processing while the GPU is still rendering the previous one. We need a means of synchronizing with the GPU’s execution timeline to ascertain the actual rendering time per frame.

The core approach involves employing GPU event queries, often provided through APIs like OpenGL or Vulkan (indirectly accessed via Python libraries such as PyOpenGL or PyVulkan). These queries allow us to mark specific points in the GPU execution pipeline and retrieve the elapsed time between them. This is crucial because we need the time difference between the end of rendering for one frame and the end of rendering for the subsequent frame to calculate FPS accurately.

A typical procedure involves the following:

1. **Issuing a query:** At the beginning (or immediately after the rendering command dispatch) of a rendering sequence for a frame, a query is issued to the GPU. This captures a timestamp representing the start time.
2. **Completing rendering:** After the rendering commands for that frame are submitted, the rendering process continues.
3. **Issuing another query:** Once that render sequence is completed (as indicated by specific sync mechanisms such as glFinish() in OpenGL), another query is issued, capturing a timestamp representing the end time.
4. **Retrieving the elapsed time:** The elapsed time between these two query timestamps, measured in nanoseconds, provides the rendering duration.
5. **Calculating FPS:** Frame rate is then calculated by taking the reciprocal of the rendering duration (converted to seconds) and averaging that value over a number of frames to avoid erratic measurements.

Let’s explore some code examples. The following example demonstrates how to calculate FPS using PyOpenGL and a query object.

```python
import OpenGL.GL as gl
import time

class GPUFrameTimer:
    def __init__(self):
        self.query = gl.glGenQueries(1)
        self.frame_times = []

    def start_timer(self):
        gl.glBeginQuery(gl.GL_TIME_ELAPSED, self.query)

    def end_timer(self):
        gl.glEndQuery(gl.GL_TIME_ELAPSED)

    def get_frame_time(self):
        available = False
        while not available:
            available = gl.glGetQueryObjectiv(self.query, gl.GL_QUERY_RESULT_AVAILABLE)
        time_elapsed = gl.glGetQueryObjectui64v(self.query, gl.GL_QUERY_RESULT)
        self.frame_times.append(time_elapsed / 1000000000.0) # Convert to seconds
        return time_elapsed / 1000000000.0

    def calculate_fps(self, sample_size=10):
        if len(self.frame_times) < sample_size:
          return None # Not enough frame data
        frame_times_avg = sum(self.frame_times[-sample_size:]) / sample_size
        return 1.0 / frame_times_avg
```

This class encapsulates the query process. `start_timer` initiates the timing, `end_timer` ends the query, and `get_frame_time` retrieves the time difference.  `calculate_fps` calculates the average frames per second over a predefined number of frames. This simple setup allows you to embed the timer directly within your rendering loop. The conversion factor accounts for the measurement of the time delta in nanoseconds.

Here's an example of how to integrate it with a rendering loop, assuming a simple PyOpenGL rendering context is already initialized:

```python
def render_frame(timer):
    # Assume some kind of rendering process here (drawing a triangle, etc.)
    gl.glClear(gl.GL_COLOR_BUFFER_BIT)
    # ... rendering code here ...
    gl.glBegin(gl.GL_TRIANGLES)
    gl.glColor3f(1.0,0.0,0.0)
    gl.glVertex3f(0,1,0)
    gl.glColor3f(0.0,1.0,0.0)
    gl.glVertex3f(-1,-1,0)
    gl.glColor3f(0.0,0.0,1.0)
    gl.glVertex3f(1,-1,0)
    gl.glEnd()
    gl.glFlush()

    timer.end_timer()  # End the timer after all drawing operations
    frame_time = timer.get_frame_time()

    fps = timer.calculate_fps()
    if fps is not None:
        print(f"FPS: {fps:.2f}, Frame time: {frame_time*1000:.2f} ms")

    timer.start_timer() # Begin timer for the next frame

if __name__ == '__main__':
    # Initialize OpenGL Context using PyOpenGL and GLFW
    import glfw
    if not glfw.init():
      raise Exception("glfw can not be initialized")

    window = glfw.create_window(640, 480, "GPU FPS Measurement", None, None)
    glfw.make_context_current(window)
    gl.glClearColor(0.0, 0.0, 0.0, 1.0)
    timer = GPUFrameTimer()
    timer.start_timer() #Initial start

    while not glfw.window_should_close(window):
        render_frame(timer)
        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()

```

This code snippet shows a barebones setup, rendering a triangle in each frame while also using `GPUFrameTimer` to calculate and display FPS. In this example, the `gl.glFlush()` call makes the rendering command more synchronous, which is necessary to make sure the GPU has processed the draw calls. The `start_timer` method call begins the timer for the next frame right after the timer data for the current frame is retrieved. This is important since you want to know the *complete* frame time.

A more advanced technique involves using fence objects or sync objects available in more modern graphics APIs like Vulkan (via Python bindings like `PyVulkan`). Sync objects can provide a finer control over GPU/CPU synchronization and offer improved performance by eliminating CPU busy-waiting (as can happen with the `glGetQueryObjectiv`).  Although this introduces added complexity, it's especially valuable when performance is critical. The following is a conceptual example that isn't executable (as the `vulkan` bindings require an initialized vulkan device context), to demonstrate the usage of fence objects to compute FPS on the GPU.

```python
# Conceptual code (requires initialized Vulkan context, and specific Vulkan Python bindings)
# Assume vulkan_device and command_queue objects exist
import time
import vklib # Hypothetical Python Vulkan bindings
class VulkanFrameTimer:
    def __init__(self, device):
        self.device = device
        self.frame_times = []
        self.frame_fence = vklib.Fence(self.device)
    
    def start_timer(self, command_buffer):
       # Record a timestamp in the command buffer
       self.start_timestamp = time.perf_counter_ns()
       command_buffer.write_timestamp( vklib.PipelineStage.TOP_OF_PIPE, vklib.TimestampQueryPool, vklib.TimestampQuery )


    def end_timer(self, command_buffer):
      command_buffer.write_timestamp( vklib.PipelineStage.BOTTOM_OF_PIPE, vklib.TimestampQueryPool, vklib.TimestampQuery+1 )

    def get_frame_time(self, command_queue):
       command_queue.submit(command_buffer, fence=self.frame_fence)
       self.frame_fence.wait()
       self.frame_fence.reset()
       start = vklib.getQueryPoolResults() # Function to get results from querypool.
       end   = vklib.getQueryPoolResults()+1
       time_elapsed = end - start
       self.frame_times.append(time_elapsed/1000000000.0)
       return time_elapsed / 1000000000.0

    def calculate_fps(self, sample_size=10):
        if len(self.frame_times) < sample_size:
          return None
        frame_times_avg = sum(self.frame_times[-sample_size:]) / sample_size
        return 1.0 / frame_times_avg

# Example within rendering loop:
# timer = VulkanFrameTimer(vulkan_device)
# command_buffer = get_command_buffer() #Obtain a command buffer to issue drawcalls.
# while True:
#   command_buffer.reset()
#   timer.start_timer(command_buffer) # Issue timestamp record at the beginning of the frame
#   #... Perform Vulkan Rendering ...
#   timer.end_timer(command_buffer)
#   frame_time = timer.get_frame_time() # Time the frame.
#   fps = timer.calculate_fps()
#    if fps is not None:
#     print(f"FPS: {fps:.2f}, Frame time: {frame_time*1000:.2f} ms")
```

This conceptual example uses a `vklib.Fence` and an assumed `vklib.QueryPool` to calculate frame times.  The time elapsed calculation uses device-specific timestamp results, which must be retrieved in a vulkan specific manner using `getQueryPoolResults`. The crucial element here is the efficient synchronisation of CPU and GPU via fences. Note that Vulkan initialization (not shown) requires much more code than OpenGL.

For further exploration, I would recommend focusing on documentation for the specific graphics API you are using: OpenGL, or Vulkan. The Khronos Group maintains extensive documentation for these APIs. Additionally, examining code examples for graphics engines like Unity, Unreal Engine or Godot (albeit in C++, or their respective languages) may provide deeper insights into practical implementations of GPU-driven FPS measurement. Consulting relevant graphics programming textbooks will also build a deeper understanding of the GPU pipeline and event timing. There are many online tutorials and forums that contain valuable practical advice, but prioritize official documentation for a sound technical foundation.
