---
title: "Why is ScreenRecording a headless browser using xvfb with ffmpeg or jmf jar(java) showing distorted video in container?"
date: "2024-12-23"
id: "why-is-screenrecording-a-headless-browser-using-xvfb-with-ffmpeg-or-jmf-jarjava-showing-distorted-video-in-container"
---

,  It's a familiar frustration, the distorted headless screen recording. I've spent many a late night chasing down similar artifacts myself, particularly back when I was working on automated UI testing pipelines that required video evidence. What might appear straightforward – capturing the output of a virtual display using `xvfb` and then encoding it with `ffmpeg` (or, in your case, the less common but still valid, `jmf` jar in java) – often devolves into a battle against subtle configuration mismatches and unforeseen interactions between these tools. The core issue, more often than not, lies in how the various pieces negotiate resolution, pixel format, and frame rate.

Let's break down the common culprits and then look at some practical solutions with code.

First, `xvfb` (X Virtual Framebuffer) provides a virtual display server. It's the foundation for running GUI applications without a physical screen attached. When setting it up, the resolution and color depth are *crucial*. Mismatches here, where your browser renders at one resolution but `xvfb` is configured differently, are a primary cause of distortion. The rendering pipeline might be scaling inappropriately or misinterpreting pixel data. Furthermore, the pixel format used by the browser can differ from what `ffmpeg` or `jmf` expects, leading to color anomalies and visual noise.

Next, frame rate is critical. If the rate at which your browser renders frames doesn’t align with how `ffmpeg` or `jmf` captures them, you'll get juddering, skipped frames, or the perception of a “slow-motion” recording. It is not always as simple as setting a specific framerate. The encoding tools are processing frames at a given rate and, if the browser or `xvfb` are not providing new frames at or above that rate, the tools will either repeat or extrapolate the data, which causes visual issues.

Then there is the encoding step. Both `ffmpeg` and `jmf` have a plethora of configuration options that, if not correctly aligned with the incoming data, can result in distortions. `ffmpeg`'s encoding settings, especially the codec, pixel format, and bitrate, have a direct impact on visual quality. `jmf`, while capable, can be less forgiving, often requiring very specific configuration settings to achieve stable and accurate video output. It's also worth mentioning that `jmf`, being somewhat older technology, can sometimes exhibit compatibility issues with modern browsers and operating systems.

Finally, containers add an extra layer of complexity. Resource limitations or configuration settings of the container environment could impact the stability of your headless browser or the performance of `xvfb`. Shared memory limitations inside the container might cause rendering bottlenecks if not properly addressed.

Now, let's go through some examples. I'll use `ffmpeg` in my examples since it's the most commonly used tool in this scenario.

**Example 1: Correct `xvfb` Configuration**

In my experience, a solid starting point is always ensuring the virtual display is set up appropriately:

```bash
#!/bin/bash
# Set DISPLAY
export DISPLAY=:99

# Xvfb setup
Xvfb :99 -screen 0 1920x1080x24 &
sleep 2 # Give xvfb time to start
# Now you can start your browser or other application that uses X display

# ... after you launch the browser, start your ffmpeg command to record. For now, we assume it starts somewhere after the browser
#ffmpeg recording here: ffmpeg -f x11grab -video_size 1920x1080 -i :99 -r 30 -pix_fmt yuv420p output.mp4
```

Here, I am specifying `1920x1080` resolution and a 24-bit color depth, which are fairly standard. I've included the `sleep 2` because `xvfb` needs a moment to initialize, and launching processes before it’s fully ready can cause unpredictable results. Importantly, I've also used the command line flag `-screen` and it's index `0` to specify the screen which is key. Note, the screen size here should match what your headless browser is configured to use. In `puppeteer` and `selenium` you can explicitly configure the size of the browser window. The flag `-pix_fmt yuv420p` is also important as it ensures that the encoding step is done correctly. If this is not present or set to something that cannot be encoded it causes the encoding tools to fail or generate bad outputs.

**Example 2: `ffmpeg` Recording with proper settings**

Now let's move to an `ffmpeg` example. Again, assuming the browser is already running on `DISPLAY :99`.

```bash
#!/bin/bash
# ffmpeg recording example with a bitrate of 10Mb/s and 30 frames per second
ffmpeg -f x11grab -video_size 1920x1080 -i :99 -r 30 -c:v libx264 -pix_fmt yuv420p -preset veryfast -crf 23 output.mp4
```
The flag `-r 30` ensures the video is recorded at 30fps. `-c:v libx264` uses the common h.264 codec and `-pix_fmt yuv420p` sets a widely compatible pixel format to avoid pixel format issues. The `-preset veryfast` and `-crf 23` parameters are for better encoding performance with acceptable visual quality, which you may want to tune for different results. It's important that the `-video_size` matches exactly what was setup with `xvfb`. This is a common place where things can go wrong.

**Example 3: Troubleshooting with Pixel Format**

Often, it's not the resolution but the pixel format that's the culprit. Sometimes, explicitly stating the pixel format for `xvfb` using an xorg.conf helps.

```bash
#!/bin/bash
# xvfb with custom xorg.conf setup.

cat << EOF > xorg.conf
Section "Device"
        Identifier "Configured Video Device"
        Driver  "dummy"
EndSection

Section "Monitor"
        Identifier "Configured Monitor"
        HorizSync 31.5-48.5
        VertRefresh 50-70
EndSection

Section "Screen"
        Identifier "Default Screen"
        Device "Configured Video Device"
        Monitor "Configured Monitor"
        DefaultDepth 24
        SubSection "Display"
                Depth 24
                Modes "1920x1080"
        EndSubSection
EndSection

Section "ServerLayout"
        Identifier "Default Layout"
        Screen "Default Screen"
EndSection
EOF

# Setting display number
export DISPLAY=:99

# Start Xvfb with xorg.conf
Xvfb :99 -screen 0 1920x1080x24 -config xorg.conf &

sleep 2 # Allow xvfb to start
# now you can launch your browser and start your capture.
#ffmpeg recording here: ffmpeg -f x11grab -video_size 1920x1080 -i :99 -r 30 -pix_fmt yuv420p output.mp4
```

Here, we're explicitly defining the pixel depth and the modes using xorg configurations. This ensures a more controlled and predictable environment. Sometimes just forcing `Xvfb` to use 24bit is the most impactful. The code snippet then uses the same `ffmpeg` capture settings as before, assuming the browser started successfully.

Beyond these examples, if you are using `jmf`, it’s useful to check for specific codecs supported by the Java Media Framework and ensure the recording parameters match those that `jmf` can reliably process. I've often found the `jmf` documentation less clear and sometimes have to resort to trial and error to get it working reliably with modern browsers.

As for further reading, I highly recommend going through the ffmpeg documentation, especially the section on x11 grabbing, and x264 encoding. A good book, although a bit dated now, is "ffmpeg Basics" which explains the basics of ffmpeg. Also, spend some time looking at the Xorg documentation regarding Xserver configurations, as this helps in understanding how to configure `xvfb` in more detail.

In the context of containers, be sure that memory limitations are not causing any performance issues, using something like `docker stats` to monitor your container usage can be helpful.

In summary, when you're encountering distorted headless recordings, systematically verify the resolution, frame rate, and pixel format at each stage – from `xvfb` to your recording tool. It’s often the small mismatches that cause the most significant headaches. Don't be afraid to test one parameter at a time to pin down the cause of the issue, and remember to look into the container environment as a possible source of problems. It is an issue with many potential pitfalls, but with a bit of methodical problem solving, it's certainly solvable.
