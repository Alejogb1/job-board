---
title: "how to remove all metadata using ffmpeg?"
date: "2024-12-13"
id: "how-to-remove-all-metadata-using-ffmpeg"
---

 so you wanna strip metadata from your media files using ffmpeg right Been there done that let me tell you it’s a rabbit hole but a necessary one if you’re serious about cleaning up your digital life or prepping content for specific platforms

I've spent way too many hours wrestling with ffmpeg it’s powerful but it also has its quirks It wasn't always smooth sailing believe me I once spent a whole weekend trying to figure out why a seemingly simple video was crashing a mobile app turns out it was some obscure metadata tag that ffmpeg hadn’t touched a nightmare I tell you So I learned the hard way that sometimes getting rid of everything is the best approach

First things first the basic command you’re probably looking for is this

```bash
ffmpeg -i input.mp4 -map 0 -c copy -map_metadata -1 output.mp4
```

let’s break this down a little `-i input.mp4` that's your input file right the video or audio whatever you've got Then `-map 0` says take all the streams from that input `-c copy` is key it’s telling ffmpeg to copy the streams without re-encoding that’s critical if you want speed and to avoid any quality loss Now `-map_metadata -1` this is the magic right here it's telling ffmpeg to drop all metadata That `-1` means all metadata will be stripped completely Finally you've got `output.mp4` which is the name of your cleaned up file

Simple enough right Well not always let’s say you’ve got a file where even that doesn’t cut it Sometimes there’s weird embedded stuff that clings on or you have specific tags you absolutely don't want

In those cases you gotta get a little more specific

```bash
ffmpeg -i input.mov -map 0 -c copy -metadata:s:v:0 comment="" -metadata:s:a:0 comment="" -metadata:s:d:0 comment="" output.mov
```

Here we’re specifically targeting comment metadata within video audio and data streams `-metadata:s:v:0 comment=""` zeroes out the comment field of video stream zero that's stream index `0` the same applies to audio streams and data streams. Now I know that specific tag might not be the issue you are facing but this kind of command allows you to clean up very specific metadata you need gone and it might be useful

This approach is useful because you might not want to drop every metadata sometimes you might want to keep the basic tags like duration and codec information but get rid of the rest like descriptions location data or author information For example if you want to keep the title but nothing else you can specify the following

```bash
ffmpeg -i input.mkv -map 0 -c copy -metadata title="Title of my video" -metadata artist="" -metadata album="" -metadata comment="" output.mkv
```

This keeps the `title` metadata tag but removes all the others like `artist` `album` `comment` by setting them to empty string you can add more metadata tags to this command as you need to like `date` or `year` etc.

Now you might think "why not just delete every single tag that could exist" that would technically work if you had a list of every single metadata tag that exist but there are many many of them and they might change between different file types and container formats so it might lead to a long list of tags to erase and it will be error prone hence the approach using `-map_metadata -1` is usually the way to go unless you want to preserve certain tags.

Remember it’s all about understanding the structure of your input file the data streams that it has and the tags that are associated with those streams Using `-map_metadata -1` usually fixes it for 99% of my use cases and it should work for you too if not then you should target specific tags.

Now I know ffmpeg can seem scary but it's really about understanding the options and their implications A lot of times problems arise from the fact that the user is not sure about which stream contains the metadata that they are trying to get rid off or they are not using the correct syntax for the different metadata they are targeting.

I also know that you might want to go more advanced than just removing metadata Maybe you'll want to add new metadata or to change existing metadata to a different value I mean let’s be honest we’ve all wanted to rename an album or add a cool description to a video right?  maybe just me but there is also something called metadata injection which is a common thing to do in media production

But before we go there let's talk about more basic things you need to know about metadata

First off metadata is basically data about data In the context of media it's all the information that's not the actual video or audio that could be the title the author the date it was created the location where it was shot stuff like that It's stored inside the file and can be really helpful for organization and indexing but as we've seen it can also cause problems like the time I spent a full weekend debugging a simple crash in a mobile app caused by a strange metadata

There are different metadata container formats too you might have ID3 tags for MP3 files EXIF data for images and all sorts of proprietary containers some you might not even know about So removing metadata is not just about one simple line of code it is more complicated than that

If you want to learn more about metadata standards I would recommend checking the book "Understanding Digital Media: Concepts and Applications" by John Giffin that covers those topics quite well

And this reminds me of the time I was trying to figure out why my videos had wrong timestamps I spent hours going through my system logs only to discover that the camera that recorded the video was storing wrong timezone information in the metadata So here is a pro tip always check the metadata for the most obscure problems they might be the cause of all your issues and you can learn about it from the book "Metadata Management for Digital Media Assets" by David Riecks.

Now a joke to lighten the mood a programmer was found dead in the shower with a shampoo bottle that said "Lather Rinse Repeat" infinite loops are not fun especially when you are trying to remove video metadata

But seriously always double-check your commands before you execute them with ffmpeg a small error can lead to unexpected results For example if you forget the `-c copy` option it can cause your video to be re-encoded which is sometimes not what you want

Also if you plan to automate these commands especially if you are processing a lot of media you should check out shell scripting techniques This will help you streamline your workflows and avoid mistakes. I remember one time I had to process thousands of videos and I did it manually that was a HUGE mistake and the next time I used shell script to automate everything so I can recommend it

Now that you know how to remove metadata feel free to experiment with other commands and options that ffmpeg has to offer but be careful it has a learning curve as it is a powerful tool

And if you have more questions I'm here to help good luck stripping down your media files!
