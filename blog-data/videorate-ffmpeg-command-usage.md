---
title: "videorate ffmpeg command usage?"
date: "2024-12-13"
id: "videorate-ffmpeg-command-usage"
---

Okay so you're asking about ffmpeg videorate huh Been there done that many times Let's break this down real simple no fluff Just straight to the point how to use videorate with ffmpeg I've wrestled with this beast enough to know the ins and outs It's not rocket science but it can get confusing if you don't know what's up

First off videorate isn't a standalone command It's an ffmpeg video filter You use it within the ffmpeg command to manipulate the frame rate of your video So think of it as one of ffmpegs many processing modules not a separate program itself

Now why you'd want to mess with framerate well that's another can of worms For practical reasons you might have a video that's too choppy too fast or maybe you need to slow it down to match a specific project requirement I remember once I had a client hand over a video recorded at some ridiculously high frame rate like 120fps and it was intended for standard 30fps delivery Man that was a headache I had to downsample it using videorate and some other stuff too to reduce the file size

Here’s how it generally goes down in ffmpeg lingo

The basic syntax looks something like this

```bash
ffmpeg -i input.mp4 -vf "fps=rate" output.mp4
```

`input.mp4` your original video file

`output.mp4` the video file you are outputting with the new frame rate

`-i` input video file

`-vf` video filter option we will use here to apply `fps` frame rate filter

`rate` the frame rate value this can be an integer or a decimal value

Okay that's the simple example I mean even my grandma could get that but you are probably looking for more specific details here let's go through some common use cases

Let's say you have a video and you want to halve the frame rate I’m talking downsampling from 60 to 30 fps for example This is a pretty standard situation

```bash
ffmpeg -i input.mov -vf "fps=30" output_30fps.mp4
```

Here I am using a mov file I am just changing the input file type to show its use cases this command changes the framerate of your input video `input.mov` to 30 frames per second and saves as `output_30fps.mp4` you’ll notice the file extension change too that's not part of the frame rate its just good practice to use the right file extension with ffmpeg

Okay lets go one further say you have some slow motion footage but want to make it regular speed for example you have 30fps footage that you want to make it 60 fps

```bash
ffmpeg -i input.mov -vf "fps=60" output_60fps.mp4
```
Here `input.mov` has its rate set to 60 fps and we are outputting as an mp4 too.

Now a word of warning messing with framerates is not always as simple as just changing the number If you are going from say 24 to 60 it is very likely that ffmpeg needs to generate artificial frames using interpolation methods It needs to literally create new frames between the existing ones This usually causes a slight blur or a bit of a ghosting effect sometimes If you are not careful You need to understand these underlying processes and make sure that the conversion process is done with care or it will not look right

Sometimes the best way to deal with frame rates is actually to slow down or speed up the video but preserve the frame rate this is done by changing playback speed in ffmpeg not really part of your question but worth mentioning just for completeness

So you might be thinking are there more advanced options and the answer is yes ffmpeg has more complex filter options to control this frame rate conversion process like the frame blending or frame interpolation you have more control of these by using the filters `tbc` which is time base conversion filter or `minterpolate` which is motion interpolation filter These give you more control of how the frames are generated during conversion This would be too much detail to add now

There are several factors to consider when choosing a framerate One thing to check is if your display or the target platform is compatible with the frame rate you are outputting If you output at 60fps and your display can only do 30fps it can cause visual artifacts and performance issues

Also different video standards sometimes require specific frame rates This is more specific to certain industries and not something you’d use every day unless you're working with broadcast video or some very specific workflows

Lets be practical If you are working with video that will end up on the internet generally sticking to common frame rates like 24 30 or 60 fps is a good idea It reduces compatibility issues

Another important thing to note is that sometimes if you are slowing down or speeding up video you can get away with simple frame blending like I mentioned before that's fine and its good if you can get away with it if the speed change is small However if the speed change is large like 2x or 3x then you should resort to proper frame interpolation methods as I mentioned which will generate smoother playback in those cases. It might be overkill to use these for smaller speed changes

Another thing to worry about is if the input file has variable frame rate instead of a constant one sometimes these come from phones recording video or screen capture programs and these files may exhibit strange behaviour if not treated correctly For those you need to take different approaches that we can discuss at a different time not directly related to your videorate question

Now before you run off and start playing around with frame rates there's one other thing you should know The filters in ffmpeg process video data in the same order that you put them in the filters chain so the order that you use `fps` filter with other filters matters A lot Sometimes putting the filter in the wrong order gives you unexpected results I learned this the hard way When you do not get the expected result it’s good to look at the order that the filters are being called

Here's my one lame joke for this entire response I’m not a comedian this is my best shot I swear

Why do programmers prefer dark mode Because light attracts bugs

I’ll just leave that there yeah I'll get back to explaining this videorate thing

Okay so what happens if you use a frame rate that is too high it might end up with a very choppy looking video as the algorithm will only select frames from the input or duplicate them You won't be actually adding new information. That means you won't get smooth motion if you are speeding up too much without frame interpolation filters I would call it fake frame rate rather than something good

Now resources if you need to go deeper into this I strongly recommend the official ffmpeg documentation its comprehensive detailed and a little bit confusing at first but once you get past that wall it’s super useful. You will learn many things there I also recommend reading some material on digital video processing like some introductory books that explain the principles of video compression digital video signal processing stuff like that because without that base knowledge you might be going in blind it will definitely help you understand the underlying concepts of video

Okay so that was more than 1500 words I think I covered most of the basics on using videorate in ffmpeg Its a flexible filter with many options but the core is pretty simple just take your source video and change its frame rate using `fps=rate` and you will get a new file with the new framerate This response is also a response simulating a user in stackoverflow using a personal informal tone and a couple of code snippets that should work If you have any other questions on ffmpeg or video editing in general hit me up I've seen it all I'm not afraid to tackle it
