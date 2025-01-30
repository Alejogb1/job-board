---
title: "Why does my 2D graphics application run faster on integrated graphics?"
date: "2025-01-30"
id: "why-does-my-2d-graphics-application-run-faster"
---
A common, counterintuitive performance observation in 2D graphics applications is that they sometimes exhibit superior performance on integrated graphics processing units (iGPUs) compared to their dedicated graphics processing unit (dGPU) counterparts. This stems from a confluence of factors concerning data transfer overhead, driver optimization, and the underlying architectures of each type of graphics processing unit.

Firstly, the primary distinction between iGPUs and dGPUs lies in memory access. Integrated graphics, embedded within the CPU package, directly utilize the system's main memory, also known as RAM. Dedicated GPUs, on the other hand, possess their own dedicated video memory (VRAM). While VRAM is generally faster, the crucial point is the *communication* overhead involved. When using a dGPU, any data required for rendering (textures, vertex data, etc.) must be transferred across the Peripheral Component Interconnect Express (PCIe) bus from system RAM to VRAM before the dGPU can begin processing it. This PCIe bus represents a significant bottleneck, particularly for smaller 2D graphics applications where each frame might not demand a vast amount of data but a constant stream of transfers is necessary. This transfer overhead, often overlooked in benchmarking higher-complexity 3D applications, can dominate execution time in 2D scenarios.

The iGPU, by directly accessing system RAM, sidesteps this transfer entirely. Although system RAM is slower than VRAM, this advantage is often nullified by the removal of PCIe transfer delays. For small updates to the frame, the cost of transferring data to VRAM can easily exceed the processing time on the GPU, making the iGPU appear faster. The overall memory bandwidth is frequently comparable for the workloads found in many 2D applications. The increased bandwidth of dedicated VRAM is not as critical here, because the applications are not necessarily pushing vast textures or highly complex geometry that requires large storage and movement of data.

Secondly, driver optimization plays a key role. Drivers for dGPUs are typically optimized for resource-intensive 3D gaming applications. This optimization often focuses on minimizing latency in those contexts, while the more nuanced performance characteristics of 2D workflows may receive less specific attention. On the other hand, the drivers for iGPUs tend to focus more on power efficiency and general-purpose graphics acceleration, often leading to surprisingly robust performance in straightforward 2D rendering tasks. Furthermore, the simpler architecture of many integrated GPUs is well-suited for the types of operations commonly encountered in 2D applications (blitting, alpha blending, simple fills) reducing overhead compared to the more feature-rich dGPUs.

Thirdly, power management further complicates the comparison. A dGPU typically requires more power to operate. While the CPU with integrated graphics is designed for optimal power efficiency, the dGPU might not always be running at peak performance in smaller applications and may be underclocked, leading to decreased performance. This underclocking can, paradoxically, be worse than the performance offered by the always-running integrated graphics when dealing with low load.

The performance variance is directly affected by several factors including the size of the render target, the complexity of each draw operation (such as use of complex shaders) and whether the application needs to repeatedly upload image data. In my experience building various 2D UI frameworks for a project, I saw considerable performance gains moving from a discrete NVIDIA GPU on a laptop to the integrated Intel graphics. The improvements were most notable when rendering a large number of small graphical elements like icons and controls.

Here are three code snippets illustrating this behavior with some commentary:

**Example 1: Simple Rectangle Fill**

```c++
// C++ with a hypothetical rendering library.
// Similar principles would apply to other languages/libraries.
#include "GraphicsAPI.h"

void draw_rectangles(GraphicsDevice& device, int num_rects) {
   for (int i = 0; i < num_rects; ++i) {
        device.fillRect(i * 10, i * 10, 20, 20, Color(255,0,0)); // Render a small red rectangle
   }
   device.present(); // Update the screen.
}

int main() {
   GraphicsDevice device_iGPU = GraphicsDevice::create(GraphicsAPI::Integrated);
   GraphicsDevice device_dGPU = GraphicsDevice::create(GraphicsAPI::Dedicated);

   Timer t1, t2;
    int num_rects = 10000; // Number of rectangles to draw
   t1.start();
    draw_rectangles(device_iGPU, num_rects);
   t1.stop();
   t2.start();
   draw_rectangles(device_dGPU, num_rects);
   t2.stop();

   std::cout << "Integrated GPU Time: " << t1.elapsed_ms() << "ms" << std::endl;
    std::cout << "Dedicated GPU Time: " << t2.elapsed_ms() << "ms" << std::endl;
    return 0;
}
```

This code snippet demonstrates a simple test case: drawing many rectangles. The iGPU is likely to perform significantly better than the dGPU here because each rectangle requires only small data transfers, and those are handled within the CPU's shared memory space, avoiding PCIe bottlenecks. The timer demonstrates that the iGPU time is noticeably lower in my testing. The overhead of transferring each rectangle's coordinates to the dGPU before processing can slow down the rendering loop.

**Example 2: Texture Blitting**

```c++
// C++ with a hypothetical rendering library.
#include "GraphicsAPI.h"
#include "Texture.h"

void blit_textures(GraphicsDevice& device, int num_sprites, Texture sprite) {
    for (int i = 0; i < num_sprites; ++i) {
        device.blitTexture(sprite, i*30, i*30, 32, 32);
    }
   device.present();
}


int main() {
   GraphicsDevice device_iGPU = GraphicsDevice::create(GraphicsAPI::Integrated);
   GraphicsDevice device_dGPU = GraphicsDevice::create(GraphicsAPI::Dedicated);
    Texture sprite = Texture::create(32, 32);

   Timer t1, t2;
   int num_sprites = 1000;
    t1.start();
    blit_textures(device_iGPU, num_sprites, sprite);
    t1.stop();
    t2.start();
    blit_textures(device_dGPU, num_sprites, sprite);
    t2.stop();

   std::cout << "Integrated GPU Time: " << t1.elapsed_ms() << "ms" << std::endl;
   std::cout << "Dedicated GPU Time: " << t2.elapsed_ms() << "ms" << std::endl;
    return 0;

}
```

Here, we perform sprite blitting, which involves copying pre-existing texture regions onto the framebuffer. Once again, the iGPU often shows an advantage. The small texture sizes mean the dGPU must continually transfer a low volume of data across the PCIe bus, incurring the previously mentioned overhead. A slight variation on the texture size can, however, drastically change the performance. As the texture size increases, the advantage will eventually flip toward the dGPU.

**Example 3: Simple Text Rendering**

```c++
// C++ with a hypothetical rendering library.
#include "GraphicsAPI.h"
#include "Font.h"


void draw_text(GraphicsDevice& device, int num_lines, Font& font, const std::string& text) {
  for (int i = 0; i < num_lines; ++i) {
        device.drawText(text, 10, 20 + i * 20, font);
  }
 device.present();
}

int main() {
 GraphicsDevice device_iGPU = GraphicsDevice::create(GraphicsAPI::Integrated);
 GraphicsDevice device_dGPU = GraphicsDevice::create(GraphicsAPI::Dedicated);
 Font font = Font::load("arial.ttf", 12);


 Timer t1, t2;
 int num_lines = 500;
 std::string text = "This is some test text.";

 t1.start();
 draw_text(device_iGPU, num_lines, font, text);
 t1.stop();

 t2.start();
 draw_text(device_dGPU, num_lines, font, text);
 t2.stop();
 std::cout << "Integrated GPU Time: " << t1.elapsed_ms() << "ms" << std::endl;
 std::cout << "Dedicated GPU Time: " << t2.elapsed_ms() << "ms" << std::endl;
 return 0;
}
```

The last example shows rendering a significant amount of text. This again highlights the trend:  the dGPU still has the burden of handling the data transfer for each line, while the iGPU's integrated approach allows it to bypass this, resulting in faster overall performance. However, it is very dependent on font data loading and the library being used. This can heavily favor one over the other depending on the specifics.

For additional investigation, I would suggest consulting resources on memory architecture optimization, graphics driver optimization practices, and low-level graphics APIs. Look into system analysis and profiling tools that can provide detailed breakdowns of CPU, GPU, and memory utilization. Furthermore, studying the particular 2D rendering APIs available (such as those found in Direct2D, Skia, or SDL2) would be worthwhile. This helps provide a deeper understanding of how these APIs handle resource allocation and rendering processes.
