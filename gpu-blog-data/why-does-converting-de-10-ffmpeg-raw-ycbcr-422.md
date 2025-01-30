---
title: "Why does converting DE-10 FFMPEG Raw YCbCr 4:2:2 frames to PNG or video produce poor results?"
date: "2025-01-30"
id: "why-does-converting-de-10-ffmpeg-raw-ycbcr-422"
---
The issue stems from a fundamental mismatch between the DE-10 Standard's raw YCbCr 4:2:2 data representation and the assumptions made by common image and video codecs when handling such data.  My experience working with high-speed video capture and processing for embedded systems, particularly involving the DE-10 platform and FFMPEG, has highlighted this several times.  The problem isn't inherently in FFMPEG, but rather in the lack of explicit metadata within the raw DE-10 stream concerning color space, pixel range, and chroma subsampling specifics.  FFMPEG, therefore, defaults to assumptions that often prove incorrect, leading to color distortion and artifacts.

**1.  Clear Explanation**

DE-10 Standard's raw YCbCr 4:2:2 output typically lacks crucial metadata embedded within the data stream itself.  Most codecs expect this metadata, explicitly defining parameters like:

* **Color Range:**  This specifies whether the Y, Cb, and Cr components utilize a limited range (e.g., 16-235 for Y) or a full range (0-255). DE-10 may use a full range or a limited range, and this isn't consistently communicated.  FFMPEG, by default, might assume a limited range, leading to a crushed black level and washed-out whites if the actual range is full.

* **Color Space:** YCbCr itself is a family of color spaces.  While DE-10 outputs YCbCr, the precise definition (e.g., BT.601, BT.709, xvYCC) is usually absent from the raw data. This omission forces FFMPEG to guess, potentially leading to color inaccuracies.

* **Chroma Subsampling:** Although stated as 4:2:2, the exact implementation details within the DE-10 output might deviate subtly from the standard.  Slight variations in how chroma samples are interspersed could cause artifacts during conversion. This is especially problematic with less sophisticated upscaling algorithms used by some codecs during conversion to higher chroma resolutions.

The absence of this metadata necessitates manual intervention in the FFMPEG command line to specify the correct parameters.  Failing to do so results in FFMPEG employing default settings that misinterpret the raw data, yielding degraded visual quality. The resulting images or videos exhibit problems such as inaccurate colors, banding, and a general lack of fidelity.


**2. Code Examples with Commentary**

The following examples demonstrate how to use FFMPEG to convert DE-10 raw YCbCr 4:2:2 frames to PNG and video formats, addressing the metadata issues outlined above.  These examples assume a 1920x1080 resolution and a full-range YCbCr 4:2:2 input. Adjust the parameters as needed to match your DE-10 output specifications.

**Example 1: Converting a single frame to PNG**

```bash
ffmpeg -f rawvideo -pixel_format yuv422p -video_size 1920x1080 -framerate 30 -i input.raw -pix_fmt yuv420p -colorspace bt709 -color_range pc -vf "scale=1920:1080" output.png
```

* `-f rawvideo`: Specifies raw video input.
* `-pixel_format yuv422p`:  Declares the input pixel format.  Crucial for informing FFMPEG about the chroma subsampling.
* `-video_size 1920x1080`: Sets the input frame dimensions.
* `-framerate 30`:  Defines the frame rate (adjust as needed).
* `-i input.raw`: Specifies the input raw file.
* `-pix_fmt yuv420p`:  Sets the output pixel format (common for PNG).
* `-colorspace bt709`: Explicitly defines the target color space (adjust based on your needs).
* `-color_range pc`:  Specifies a full range (pc), modify to 'tv' for limited range if necessary.
* `-vf "scale=1920:1080"`: Ensures scaling matches input resolution. This step could be omitted if output resolution matches input.


**Example 2: Converting a sequence of frames to an MP4 video**

```bash
ffmpeg -f rawvideo -pixel_format yuv422p -video_size 1920x1080 -framerate 30 -i input%04d.raw -c:v libx264 -pix_fmt yuv420p -colorspace bt709 -color_range pc -preset medium -crf 23 output.mp4
```

This example builds upon the first, adding the video encoding:

* `input%04d.raw`: This wildcard indicates a sequence of files (input0001.raw, input0002.raw, etc.).
* `-c:v libx264`: Selects the x264 encoder (a widely used and efficient H.264 encoder).  Other encoders are available (e.g., libx265 for H.265).
* `-preset medium`: Controls the encoding speed/quality trade-off.  `-preset slow` yields higher quality but takes longer.
* `-crf 23`: Constant Rate Factor; a lower CRF value means higher quality (and a larger file size).


**Example 3: Handling a different color space (assuming BT.601)**

```bash
ffmpeg -f rawvideo -pixel_format yuv422p -video_size 1920x1080 -framerate 30 -i input.raw -c:v libx264 -pix_fmt yuv420p -colorspace bt601 -color_range pc -preset medium -crf 23 output_bt601.mp4
```

This illustrates changing the color space to BT.601.  The crucial part is accurately identifying the color space used by the DE-10 output.  Consult the DE-10 specifications or your capture device's documentation.  Incorrectly specifying the color space will still lead to flawed results.



**3. Resource Recommendations**

* The FFMPEG documentation:  This is indispensable.  Carefully study the sections on raw video input, pixel formats, color spaces, and video encoders.  Understanding the intricacies of each parameter is vital.
* A comprehensive guide to video codecs:  Such a guide will enhance your understanding of various video encoding standards and their capabilities.
* A digital signal processing textbook: This resource is valuable for understanding the fundamentals of color spaces, chroma subsampling, and other relevant signal processing concepts.  This deeper theoretical knowledge will help you troubleshoot more effectively.

By meticulously specifying the relevant metadata in your FFMPEG commands, using suitable options for color space and range, and choosing an appropriate encoder, you should achieve significantly improved results when converting DE-10 raw YCbCr 4:2:2 frames to other formats. Remember that accurate source information from your DE-10 system is crucial for successful conversion.  The examples provided serve as a starting point; adapting them to your specific DE-10 configuration is essential for optimal outcome.
