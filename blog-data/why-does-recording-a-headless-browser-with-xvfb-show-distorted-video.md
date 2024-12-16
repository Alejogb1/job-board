---
title: "Why does recording a headless browser with XVFB show distorted video?"
date: "2024-12-16"
id: "why-does-recording-a-headless-browser-with-xvfb-show-distorted-video"
---

Okay, let's tackle this. It’s a problem I’ve certainly run into more than a few times over the years, especially when dealing with automated UI testing and web scraping pipelines. The issue you're describing—distorted video output when recording a headless browser using xvfb—isn't entirely straightforward, but it usually boils down to a combination of factors related to framebuffer management, resolution mismatch, and how the video encoding process interacts with this simulated display environment.

From my experience, several key components contribute to this distortion, and understanding them is critical to finding a solution. First, consider xvfb itself. Xvfb, or X virtual framebuffer, is essentially an X server that operates entirely in memory. It doesn't have a physical display; it creates a virtual buffer to simulate the display, which is crucial for running GUI applications without a monitor, like headless browsers. This virtual framebuffer is where the browser renders its output. Now, when you try to record the output, what you are actually recording is the contents of this framebuffer, captured at specific intervals.

The distortion we observe often arises when the video capture software—like ffmpeg, which is my tool of choice—misinterprets or improperly scales the framebuffer data. This can happen for several reasons, including:

1.  **Resolution Mismatch:** The most frequent cause is a mismatch between the resolution of the xvfb framebuffer and the resolution or pixel aspect ratio assumed by the video encoder. If you initialize xvfb with, say, a 1024x768 resolution but the recording tool expects a different format, it'll try to squeeze or stretch the pixel data, resulting in a distorted image. This can manifest as stretched, compressed, or otherwise warped visuals. It isn't necessarily just the absolute resolution but also the assumed aspect ratio. The browser might be rendering correctly within the xvfb framebuffer, but the encoder interprets the pixels incorrectly.

2.  **Color Space and Pixel Format Issues:** Another contributing factor is incorrect handling of color spaces and pixel formats. The framebuffer stores pixels using a specific format (e.g., rgb24, rgba32). If the video encoder expects a different format (e.g., yuv420p, often used in video compression), then a conversion process is necessary. Errors during this conversion can introduce visual artifacts or distortions. Incorrect bit depth is another aspect that is closely related to pixel format issues. If the xvfb is set up with a pixel depth that differs from what the capture process expects this can lead to a corrupted visual output.

3.  **Frame Rate Discrepancies:** Issues can arise if the capture framerate doesn't match the browser's rendering rate. if you are capturing at a higher framerate than the browser is rendering to the framebuffer, you might see duplicated frames or strange artifacts. Conversely, capturing at a lower framerate could lead to a choppy or inconsistent recording.

4.  **Buffer Copying Errors:** Occasionally, especially in more complex capture pipelines, there might be issues related to how data is read from the framebuffer and passed to the encoder. If these buffer copies are done incorrectly or are poorly synchronized, distortions can occur. This issue is not as common if using a standard tool like ffmpeg, but when using custom capture mechanisms, this becomes an area of consideration.

Let's look at some code examples, focusing on ffmpeg since that is a common tool for the capture process.

**Example 1: Basic Capture with Resolution Mismatch**

Here's an example where distortion is *likely* to occur due to mismatched resolutions:

```bash
#!/bin/bash

xvfb-run --server-args="-screen 0 1024x768x24" \
    google-chrome --headless --disable-gpu --screenshot /tmp/screenshot.png https://example.com &

sleep 5 # wait for page to load.

ffmpeg -f x11grab -video_size 1280x720 -i :0.0 \
    -c:v libx264 -pix_fmt yuv420p /tmp/distorted_video.mp4
```

In this case, `xvfb-run` sets up a virtual screen with 1024x768 resolution, but ffmpeg tries to record a 1280x720 window. The resulting video will likely be stretched and distorted.

**Example 2: Correcting Resolution and Pixel Format**

Here's a corrected version of the previous example. This one should yield better results:

```bash
#!/bin/bash

xvfb-run --server-args="-screen 0 1280x720x24" \
    google-chrome --headless --disable-gpu --screenshot /tmp/screenshot.png https://example.com &

sleep 5 # wait for page to load.

ffmpeg -f x11grab -video_size 1280x720 -i :0.0 \
    -c:v libx264 -pix_fmt yuv420p /tmp/correct_video.mp4
```
Here we ensure that the xvfb resolution matches the `video_size` parameter passed to ffmpeg, greatly reducing the likelihood of a distorted output.

**Example 3: Specifying Frame Rate for Consistent Capture**
This example shows how to specify both the frame rate and the video resolution when capturing video from XVFB. This will address issues around potential frame duplication or choppiness.

```bash
#!/bin/bash
xvfb-run --server-args="-screen 0 1280x720x24" \
    google-chrome --headless --disable-gpu --screenshot /tmp/screenshot.png https://example.com &
sleep 5 # wait for page to load.

ffmpeg -f x11grab -video_size 1280x720 -i :0.0 \
    -r 30 -c:v libx264 -pix_fmt yuv420p /tmp/correct_video_fps.mp4
```

In this example, we add the `-r 30` parameter which enforces a frame rate of 30 frames per second. This will help avoid framerate related distortions.

To dig deeper, I would highly recommend several resources. First, for understanding the inner workings of the X Window System and framebuffers, the "X Window System Protocol" documentation (you can often find it online) provides detailed explanations. For more information on xvfb, the man page (`man xvfb`) is an excellent starting point. When it comes to video encoding, "The FFmpeg Handbook" by Roger Pack provides an extremely thorough dive into how ffmpeg works, including details on various codecs, pixel formats, and best practices. Additionally, "H264 and MPEG-4 Video Compression" by Iain E.G. Richardson is invaluable for grasping the complexities of video encoding.

In summary, the distorted video issues when recording a headless browser using xvfb often stem from mismatched resolutions, incorrect pixel formats, inconsistent framerates, or buffer handling errors. Careful attention to xvfb’s display parameters, ffmpeg’s encoding options, and keeping the capture pipeline simple helps prevent these problems. Always ensure that your output resolution matches your xvfb resolution and that you select a suitable pixel format that both the browser renders to and the video encoder expects. With proper configuration and attention to the finer details, you can effectively record headless browser sessions with quality video.
