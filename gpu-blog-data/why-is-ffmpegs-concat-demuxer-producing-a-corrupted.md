---
title: "Why is FFmpeg's concat demuxer producing a corrupted video segment?"
date: "2025-01-30"
id: "why-is-ffmpegs-concat-demuxer-producing-a-corrupted"
---
The core challenge with FFmpeg's concat demuxer, which often manifests as corrupted or truncated video segments during concatenation, typically stems from inconsistencies in the underlying stream metadata or the way individual files are prepared prior to concatenation. This isn't an intrinsic flaw in the demuxer itself, but rather a consequence of it being a relatively “thin” layer that expects well-formed inputs, rather than attempting robust error correction. In my experience spanning numerous projects involving video stitching for content aggregation, I've found that meticulous attention to detail during the input preparation phase is absolutely crucial to ensure seamless transitions and avoid these corruption issues.

The concat demuxer, unlike more advanced editing tools, operates based on the principle of appending bitstreams. It assumes that the constituent video and audio streams have identical codecs, bitrates, resolutions, and other critical parameters. Divergences in these parameters, even seemingly insignificant ones, can lead to the issues you’re experiencing. Consider that the concat demuxer primarily works by parsing the ‘file’ metadata entries, not necessarily by performing a deep dive into the internal stream data. Therefore, if file A’s metadata indicates a certain frame rate but the actual video stream within has subtle variations (perhaps the first few frames were accidentally dropped), then the subsequent concatenation will likely result in either partial reads, playback artifacts, or corrupted output. Specifically, subtle temporal discontinuities are very difficult for most decoders to handle gracefully when presented as a single, continuous stream.

The problem usually occurs in one of several ways: first, different encoders might generate identical encodings but insert different metadata. Second, even if source files were supposedly created with identical encoders and settings, subtle encoding deviations can occur if the encoding process was not managed carefully (such as slight variations in encoding passes or using different library versions). Finally, an incorrect specification of the input metadata in the text file which the concat demuxer reads, will directly cause issues.

Let's examine three common scenarios and their associated solutions via FFmpeg commands.

**Example 1: Inconsistent Frame Rates**

Assume two video files, `part1.mp4` and `part2.mp4`, appear visually identical but, internally, have slight differences in frame rate. This could happen if the files originated from different devices or were processed by different software. Directly concatenating them via a `concat` demuxer config file (`mylist.txt`) like so:

```
file 'part1.mp4'
file 'part2.mp4'
```

and running:

```bash
ffmpeg -f concat -safe 0 -i mylist.txt -c copy output.mp4
```

often produces a corrupted output; frame drops and playback errors are common. The underlying issue here isn't a bug in `ffmpeg` but simply that the demuxer joins the bitstreams without accounting for the temporal discontinuity. The fix requires re-encoding both files to a single, shared frame rate before concatenation. Suppose, after inspection using `ffprobe`, that `part1.mp4` has a nominal frame rate of 29.97 fps, while `part2.mp4` was accidentally encoded at 30 fps. We can fix this by applying the following processing chain:

```bash
ffmpeg -i part1.mp4 -r 30 -c:v libx264 -preset veryfast -pix_fmt yuv420p -c:a copy part1_fixed.mp4
ffmpeg -i part2.mp4 -r 30 -c:v libx264 -preset veryfast -pix_fmt yuv420p -c:a copy part2_fixed.mp4
```

Here, we’re forcing both files to 30 fps (the most common of the two) using the `-r` parameter. The `-c:v libx264` forces a re-encode using the x264 codec, and `-preset veryfast` trades off compression for speed (adjust as needed for your use case); the `-pix_fmt yuv420p` is included as a general compatibility setting. Finally, `-c:a copy` ensures that the audio is copied directly without modification (assuming there is no audio sync issue). The fixed files can now be combined with the `concat` demuxer using the same file list approach (with the new file names) to produce a seamless output. I have found that failing to correct frame rate inconsistencies such as this one to be one of the single largest sources of concat demuxer errors I’ve observed.

**Example 2: Variations in Pixel Format and Codec Settings**

In another scenario, you might have two files seemingly encoded with the same codec, but have subtle variations. Imagine `partA.mov` and `partB.mov`, both H.264 encoded, but `partA.mov` was encoded using a lossless pixel format (like `yuv444p`), while `partB.mov` uses the common `yuv420p` pixel format. While both files might play independently, attempting to concatenate them directly through the demuxer using the same method as in the first example:

```
file 'partA.mov'
file 'partB.mov'
```

and running:
```bash
ffmpeg -f concat -safe 0 -i mylist.txt -c copy output_corrupt.mov
```
will also lead to problems since the underlying pixel format changes. The fix here lies in ensuring all parts have a consistent pixel format (and codec profile, when possible).

```bash
ffmpeg -i partA.mov -c:v libx264 -pix_fmt yuv420p -c:a copy partA_fixed.mov
ffmpeg -i partB.mov -c:v libx264 -pix_fmt yuv420p -c:a copy partB_fixed.mov
```

This re-encodes the pixel format of `partA.mov` to the more common `yuv420p`, mirroring what's already present in `partB.mov`, while keeping the same audio stream. This ensures consistency across the streams, resolving potential corruption at the demuxer level. Again, re-encoding is often crucial for resolving this issue.

**Example 3: Incorrect File Path or Metadata**

Finally, a more subtle cause is an error in the text list provided to the concat demuxer. Consider a list that includes a misspelt filename, like:

```
file 'correct_file_1.mp4'
file 'wrong_filename.mp4'
file 'correct_file_2.mp4'
```
FFmpeg will likely encounter an error and potentially return a partial video or crash, depending on how it handles the missing file.

Another more common scenario is when full file paths are used, especially across multiple operating systems, where path formats may differ. Using absolute paths like `file '/home/user/videos/part1.mp4'` within the text file makes that configuration non-portable if the file structure changes. Although, generally, the concat demuxer is robust, it can sometimes exhibit unexpected behavior based on different input paths. The most robust approach, and one which I have adopted over the years, is to always work within a common directory where all parts are located.
Using simple file paths like:
```
file 'part1.mp4'
file 'part2.mp4'
```
will resolve this specific case so long as both files reside in the same directory as where the command is executed. In short, the `file` entry within the configuration file needs to be verified and the system that executes the `ffmpeg` command needs to be able to resolve that filepath.

In conclusion, the concat demuxer does not introduce errors itself. Instead, it faithfully appends the raw video bitstreams as instructed, so if issues appear, they are indicative of source inconsistencies or incorrect specifications provided. The key to successful concatenation lies in the meticulous preparation of your input files ensuring that they are internally consistent and in providing the correct file list to the demuxer.

For further learning, I recommend studying: 1) the official FFmpeg documentation, particularly the demuxing section; 2) documentation concerning x264 and other common codecs (particularly their profiles and levels); and 3) resources that detail the principles behind video encoding, compression, and container formats. Additionally, experimenting with `ffprobe` to inspect the metadata of individual files prior to concatenation has been invaluable in my own workflows to identify these subtle differences before concatenating a large batch of videos, thus preventing unnecessary work further down the line.
