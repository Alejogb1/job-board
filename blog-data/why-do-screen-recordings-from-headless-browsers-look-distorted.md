---
title: "Why do screen recordings from headless browsers look distorted?"
date: "2024-12-23"
id: "why-do-screen-recordings-from-headless-browsers-look-distorted"
---

,  I’ve spent a fair bit of time debugging rendering issues with headless browsers, particularly when it comes to video capture, so I'm familiar with the challenges this presents. The distortion you're seeing in screen recordings from headless browsers isn't some mystical quirk; it stems from a confluence of factors primarily related to how headless environments differ from typical graphical display setups. These differences impact rendering pathways, leading to artifacts and visual discrepancies. It's a fairly common problem, and it’s one I've addressed more than once.

Firstly, the heart of the issue often lies with the absence of a physical display device. A traditional browser, operating in a graphical environment, renders content by communicating with the system's graphics hardware (GPU) through an operating system's graphics drivers. This process involves sophisticated optimizations, including hardware acceleration, that leverage the GPU to offload computationally intensive rendering tasks, resulting in smooth and accurate visual output. However, headless browsers by their very nature lack this physical display. This means they must rely on software rendering techniques. In most cases, they use a software implementation of the rendering pipeline which, while generally functional, is not as optimized as hardware-accelerated rendering and may introduce discrepancies in how pixels are processed and presented, specifically relating to pixel precision or subpixel rendering which is often optimized for specific physical pixel grid.

Secondly, the framebuffer, or the memory location where rendered pixels are stored before being displayed, is typically managed differently. With a graphical display, the framebuffer is directly associated with a screen's physical characteristics such as pixel resolution, pixel density, and color profile. Headless browsers, not being directly connected to a display, must either emulate this or rely on a virtual framebuffer. How this virtual framebuffer is initialized, and the rendering process used with it, contributes significantly to distortions during screen recordings. For instance, differences in how pixel scaling and anti-aliasing are handled in software rendering pipelines can introduce blurring, jagged edges, or incorrect colors. Another common cause is the incorrect application of color management policies, especially when converting from a working color space to video container color spaces. These differences frequently arise due to variation in image and video codecs, container formats and pixel storage methods.

Thirdly, the frame capturing mechanism itself can be a source of distortion. When recording from a headless browser, you're often essentially taking snapshots of the virtual framebuffer and encoding them into a video format. The way this capture is timed and managed in relation to rendering completion will influence the integrity of the recorded video. If the frame capturing occurs before the content is fully rendered or while it's in an intermediate rendering state, it can lead to visual artifacts like tearing or incomplete updates in the final video file. Furthermore, the chosen recording resolution, aspect ratio, and the encoding parameters can compound distortions if not handled meticulously. For example, downscaling or upscaling performed during capture might not match the render resolution, leading to artifacts depending on the scaling method used.

Let’s look at some code examples. These illustrations are simplified but convey the key ideas.

**Example 1: Canvas Rendering and Resolution Discrepancies**

This snippet demonstrates how a basic canvas drawing might appear when rendered in a headless context and then captured, revealing how resolution mismatches and lack of anti-aliasing can lead to distortion.

```python
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import base64
import time

def generate_canvas_image(resolution=(600, 400)):
    options = Options()
    options.add_argument("--headless")
    driver = webdriver.Chrome(options=options)
    driver.set_window_size(*resolution)

    driver.get("data:text/html;charset=utf-8," +
               "<canvas id='myCanvas'></canvas><script> "
               "var canvas = document.getElementById('myCanvas');"
               "canvas.width = " + str(resolution[0]) + ";"
               "canvas.height = " + str(resolution[1]) + ";"
               "var ctx = canvas.getContext('2d');"
               "ctx.fillStyle = '#ff0000';"
               "ctx.fillRect(50, 50, 100, 100);"
               "ctx.font = '30px sans-serif';"
               "ctx.fillStyle = '#0000ff';"
               "ctx.fillText('Test Text', 200, 200); </script>")

    time.sleep(1) # allow time for render
    image_b64 = driver.execute_script("return document.querySelector('canvas').toDataURL('image/png').substring(22);")
    driver.quit()

    with open('canvas_output.png', 'wb') as f:
        f.write(base64.b64decode(image_b64))

generate_canvas_image()

```

In this code, the headless browser renders a simple canvas with a rectangle and some text. If the virtual buffer doesn't perfectly align with canvas size and proper anti-aliasing methods during text rendering are absent, the recorded image could appear with jagged edges or pixel-level artifacts. The lack of hardware acceleration is typically the main culprit here. The output will often show that text will appear pixelated because it was not properly sub-pixel rendered.

**Example 2: Frame Capture Timing Issues**

This Python snippet highlights how using Javascript to trigger frame capture can be problematic due to potential racing conditions if the render pipeline isn't properly synchronised.

```python
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time
import base64

def capture_with_time_delay():
    options = Options()
    options.add_argument("--headless")
    driver = webdriver.Chrome(options=options)
    driver.set_window_size(800, 600)

    driver.get("data:text/html;charset=utf-8," +
               "<div style='width:400px; height:300px; background-color:blue; display:none;' id='myDiv'></div>"
                "<script>"
                "setTimeout(() => { document.getElementById('myDiv').style.display = 'block'; }, 500);"
                "</script>"
                "<script>window.captureImage = function() { return document.documentElement.toDataURL();}</script>")

    time.sleep(0.3)  # Short delay before triggering capture
    image_b64 = driver.execute_script("return window.captureImage().substring(22);")
    driver.quit()
    with open('delayed_capture.png', 'wb') as f:
        f.write(base64.b64decode(image_b64))


capture_with_time_delay()

```

Here, a blue div initially hidden then shown after a 500ms delay. The capture is triggered soon afterwards using javascript `toDataUrl()` method. If the browser’s rendering hasn't completed by the time we execute the capture function, we could get a partial render, which appears distorted in a video. Proper signalling and waiting for element rendering via javascript is essential to avoid issues like this.

**Example 3: Incorrect Scaling Configuration**

This Java example shows how a headless browser might inadvertently apply scaling, leading to image distortions if the browser's zoom or scaling settings aren't set properly.

```java
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.chrome.ChromeDriver;
import org.openqa.selenium.chrome.ChromeOptions;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Base64;

public class ScalingIssues {

  public static void main(String[] args) throws IOException {

      ChromeOptions options = new ChromeOptions();
      options.addArguments("--headless");
      options.addArguments("--disable-gpu"); // recommended for headless mode
    options.addArguments("--force-device-scale-factor=1"); // prevents accidental scaling
    options.addArguments("--window-size=800,600");
    WebDriver driver = new ChromeDriver(options);
    driver.get("data:text/html;charset=utf-8,<h1>Hello, World!</h1>");
    String base64EncodedImage = (String)((org.openqa.selenium.JavascriptExecutor) driver)
            .executeScript("return document.documentElement.toDataURL('image/png').substring(22);");
    driver.quit();
    byte[] imageBytes = Base64.getDecoder().decode(base64EncodedImage);
        try (FileOutputStream fos = new FileOutputStream("scaling_output.png")) {
        fos.write(imageBytes);
        }
  }
}
```

The example uses Java and demonstrates how `--force-device-scale-factor=1` is important in preventing the browser from scaling the output to various DPI settings. Without this, and with default scaling settings, the recorded output may show the content scaled up or down with corresponding distortions.

To delve deeper into these issues, I recommend a few sources. For a thorough understanding of web rendering pipelines, look into “HTML5 Canvas” by Steve Fulton and Jeff Fulton or any material specifically discussing advanced web graphics. The official documentation for Chromium, the engine behind many headless browsers like Chrome and Puppeteer, offers detailed insights into its rendering architecture. Additionally, the "OpenGL Programming Guide" by Dave Shreiner et al. gives an in-depth understanding of graphics rendering, even though the browser might be using software implementation. Furthermore, papers and specifications around video codecs like H.264/AVC and VP9 will help you understand how frame data and encoded video formats differ from raw pixel representations. Finally the W3C specification for the html5 Canvas element provides an authoritative source for its behavior and rendering methods.

In short, achieving high-fidelity screen recordings from headless browsers requires careful attention to the subtleties of software rendering, frame capture timing, scaling, and how your capture pipeline works. By understanding the core problems and implementing techniques to avoid them, you can significantly improve the quality and reduce distortions in your recordings.
