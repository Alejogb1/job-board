---
title: "How can MediaInfo output data to the console?"
date: "2025-01-30"
id: "how-can-mediainfo-output-data-to-the-console"
---
MediaInfo, a powerful command-line tool and library for analyzing media files, defaults to outputting its analysis to standard output (stdout), which typically renders on the console. This core functionality provides the foundation for extracting a wide array of metadata from video and audio files. However, understanding how to effectively control and format this output is crucial for practical applications, ranging from simple file inspection to complex automated workflows. My extensive use of MediaInfo in building automated media processing pipelines has yielded a variety of techniques for leveraging its console output capabilities.

The standard invocation of the MediaInfo executable, `mediainfo <filename>`, directly prints a formatted report to the console. The default output includes a comprehensive collection of information organized into several sections, such as "General," "Video," "Audio," and other relevant tracks. This output is generally sufficient for a quick overview but may not be ideal for structured parsing or integration into automated systems. Control over the output format and content is therefore essential.

One primary method for altering MediaInfo's output is through the `--Inform` parameter. This parameter enables the user to specify a custom format string, detailing precisely what information to display and its structure. This string uses a unique syntax akin to a templating language, using predefined keywords representing various pieces of media metadata. For instance, `"%FileName%\n%FileSize%\n"` outputs the filename and filesize separated by newlines. The available keywords are extensive, encompassing a wide variety of technical data points associated with the media. Utilizing this mechanism significantly enhances control over output content.

Another valuable flag is `--Output`. This option directly affects the output format to predefined, structured outputs like CSV (comma separated values) or XML (Extensible Markup Language). Specifying `--Output=CSV` directs MediaInfo to format the metadata in a single comma delimited line, facilitating easy parsing by other tools or scripts. `--Output=XML`, on the other hand, outputs well-structured XML, which proves especially valuable for complex automated systems that require standardized data. These options circumvent the complexity of constructing a manual format string using `--Inform`, while still providing access to structured data. The selection between using `--Inform` or `--Output` often hinges on the desired level of granularity and whether a standard output structure is required.

Finally, MediaInfo supports multiple output streams. While typically stdout is the destination, the output stream can be redirected using standard shell redirection operators. For instance, `mediainfo <filename> > output.txt` redirects the entire output to a file named `output.txt`. This proves beneficial when dealing with large media libraries or when the output requires further processing outside the console, or when needing to store logs. Combined with `--Inform` and `--Output`, these redirection operations provide robust and flexible handling of MediaInfo's output.

Below are code examples illustrating these various output techniques.

**Example 1: Using `--Inform` for custom output.**

```bash
mediainfo --Inform="General;File name: %FileName%\nFile size: %FileSize% bytes\nDuration: %Duration/String3%\nVideo codec: %Video_Codec_Name%" my_video.mp4
```

**Commentary:**

This example demonstrates the use of the `--Inform` flag. The provided format string specifies that the output should include the filename, filesize, duration in a user friendly format, and the video codec, each on a new line. The string begins with `General;`, indicating that general metadata fields will be accessed. The `%` symbol precedes each metadata keyword, which allows MediaInfo to extract that information. By specifying the fields precisely using this methodology, the output becomes tailored for specific parsing needs. The output of this command executed on a file called `my_video.mp4` could be something like:

```
File name: my_video.mp4
File size: 12345678 bytes
Duration: 00:01:23.456
Video codec: AVC
```

**Example 2: Using `--Output` for CSV output.**

```bash
mediainfo --Output=CSV my_audio.mp3
```

**Commentary:**

This example showcases the use of the `--Output=CSV` option. MediaInfo will output a single comma separated line of various metadata fields of the audio file `my_audio.mp3`. This example does not specify which exact fields to include; rather, the default set of attributes are included in the comma separated format. The lack of control over exact output fields is a trade off for the benefit of a standardized CSV structure, however this can be further refined using `--Inform` in conjunction with `--Output=CSV`. The output from this example might resemble:

```
File name,File size,Duration,Bitrate,Codec,
my_audio.mp3,5123456,123.456,128000,MP3
```

**Example 3: Redirecting output to a file.**

```bash
mediainfo --Inform="General;%FileName%\t%FileSize%\t%Duration/String3%" large_video_collection/*.mov > metadata_log.txt
```

**Commentary:**

Here, the output of MediaInfo is redirected to a file named `metadata_log.txt`. This example executes the MediaInfo command on multiple video files matching the wildcard pattern `large_video_collection/*.mov`. The `--Inform` flag specifies that each line in the output includes filename, filesize, and duration of each file, each tab delimited. The shell redirection operator `>` redirects the standard output stream to the specified file. This is a common approach when extracting metadata from many files and storing them for further analysis, or when needing to keep a log of processing steps. A truncated example of `metadata_log.txt` would resemble:

```
file1.mov	10000000	00:05:00.000
file2.mov	20000000	00:10:00.000
file3.mov	15000000	00:07:30.000
...
```

In summary, MediaInfo provides robust and flexible control over its console output via the `--Inform` and `--Output` flags as well as standard shell redirection techniques. Using `--Inform` enables creation of custom-formatted reports by explicitly listing required information. Employing `--Output=CSV` or `--Output=XML` simplifies integration with other software by generating standardized structured outputs. Directing output to files through redirection allows for processing large collections and keeping records of MediaInfo analyses. The combination of these features renders MediaInfo highly adaptable for a variety of tasks involving media metadata extraction and processing.

For advanced understanding, I would recommend studying the MediaInfo command-line interface documentation and testing various parameters. Practical experience is essential for mastering the intricacies of the tool. Understanding the available metadata fields and the templating syntax used with the `--Inform` parameter is also crucial for maximizing utility. Additionally, exploring existing community forums and articles pertaining to specific use cases would further enhance proficiency. A thorough grasp of basic shell scripting or programming fundamentals would also be highly beneficial when using the command line tool for complex automation purposes. These approaches will ensure a well-rounded understanding and practical capability in extracting metadata from media files using MediaInfo.
