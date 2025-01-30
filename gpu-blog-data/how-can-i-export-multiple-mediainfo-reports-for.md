---
title: "How can I export multiple MediaInfo reports for video and audio files using pymediainfo?"
date: "2025-01-30"
id: "how-can-i-export-multiple-mediainfo-reports-for"
---
Pymediainfo, a Python wrapper for MediaInfo, offers robust capabilities for extracting detailed metadata from media files; however, natively, it operates on a per-file basis. Extending this to bulk processing requires implementing an iterative approach, handling each file individually and aggregating the results. My experience processing large media libraries has shown that an organized, script-based solution is far more efficient than manual processing, especially when dealing with hundreds or thousands of files.

The core challenge lies in managing the instantiation of the MediaInfo object for each file, then extracting the desired information and formatting it into a usable structure. Here, I will demonstrate three primary methods using `pymediainfo` to achieve this, focusing on different output formats: textual, JSON, and CSV. I will assume a directory of media files for demonstration; a real-world application would, of course, require appropriate file path management.

**Method 1: Basic Text Export**

This approach focuses on a straightforward text-based output, printing the information to the console or directing it to a file. It does not involve explicit data structures but instead relies on the `to_data()` method of the `MediaInfo` object. This is the simplest method for quickly inspecting the results of `pymediainfo`.

```python
from pymediainfo import MediaInfo
import os

def export_text_reports(directory):
    for filename in os.listdir(directory):
        if filename.endswith((".mp4", ".mov", ".mkv", ".avi", ".mp3", ".wav")): # Add desired extensions
            filepath = os.path.join(directory, filename)
            try:
                media_info = MediaInfo.parse(filepath)
                print(f"---- MediaInfo Report for: {filename} ----")
                print(media_info.to_data())
                print("\n")
            except Exception as e:
               print(f"Error processing {filename}: {e}")


if __name__ == "__main__":
    media_directory = "/path/to/your/media/directory" # Replace with your directory
    export_text_reports(media_directory)
```

*Explanation:* The code iterates through the specified directory, filtering files based on common media extensions. For each file, it attempts to create a `MediaInfo` object and then prints the raw output using `to_data()`. Error handling is included using a `try-except` block to gracefully manage any parsing issues, logging any problematic file for later investigation. The `if __name__ == "__main__":` block ensures that the function is executed only when the script is run directly. The raw output from `to_data()` is a string that can be redirected into a text file for later parsing using a different script.

**Method 2: JSON Export**

JSON, due to its structured format, allows for better automation and integration with other tools.  Here, I will demonstrate how to process media files and store their metadata as JSON objects, suitable for database insertion, web API processing or any task that requires structured data.

```python
from pymediainfo import MediaInfo
import os
import json

def export_json_reports(directory, output_file):
    all_media_data = []
    for filename in os.listdir(directory):
        if filename.endswith((".mp4", ".mov", ".mkv", ".avi", ".mp3", ".wav")):
            filepath = os.path.join(directory, filename)
            try:
                media_info = MediaInfo.parse(filepath)
                media_data = media_info.to_json()
                all_media_data.append(json.loads(media_data))
            except Exception as e:
                print(f"Error processing {filename}: {e}")


    with open(output_file, 'w') as outfile:
        json.dump(all_media_data, outfile, indent=4)


if __name__ == "__main__":
    media_directory = "/path/to/your/media/directory"  # Replace with your directory
    output_json = "media_reports.json"  # Replace with your desired file name
    export_json_reports(media_directory, output_json)
```

*Explanation:* This code utilizes the `to_json()` method of the `MediaInfo` object which returns a JSON formatted string.  This string is then parsed using `json.loads()` before being appended to the list called `all_media_data`. This ensures that data is processed correctly, which `json.dump()` needs. Once all media files have been processed, it writes this data into the specified JSON file with an indent of 4 for readability. This method produces a single JSON file that contains a list of individual reports, simplifying further analysis.

**Method 3: CSV Export**

CSV is often necessary when working with spreadsheet tools or other data analysis applications that prefer tabular data. This method demonstrates how to extract specific attributes and store them in a CSV format.  This method uses the `tracks` property of the `MediaInfo` object to process only the information that I want, avoiding the full report.

```python
from pymediainfo import MediaInfo
import os
import csv

def export_csv_reports(directory, output_file):
    header = ["File Name", "Format", "Duration", "Bit Rate", "Width", "Height", "Audio Codec", "Audio Bit Rate", "Sample Rate"]
    all_media_data = []

    for filename in os.listdir(directory):
        if filename.endswith((".mp4", ".mov", ".mkv", ".avi", ".mp3", ".wav")):
            filepath = os.path.join(directory, filename)
            try:
                media_info = MediaInfo.parse(filepath)
                file_data = [filename] # Add the file name

                for track in media_info.tracks:
                  if track.track_type == "General": # Collect information from the general track
                    file_data.extend([track.format, track.duration, track.overall_bit_rate])
                  if track.track_type == "Video": # Collect video information
                    file_data.extend([track.width, track.height])
                  if track.track_type == "Audio": # Collect audio information from the first audio track found
                      file_data.extend([track.codec_id, track.bit_rate, track.sampling_rate])
                      break # Stop looking for other audio tracks to keep output simple

                all_media_data.append(file_data)
            except Exception as e:
               print(f"Error processing {filename}: {e}")


    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(all_media_data)


if __name__ == "__main__":
    media_directory = "/path/to/your/media/directory" # Replace with your directory
    output_csv = "media_reports.csv" # Replace with your desired file name
    export_csv_reports(media_directory, output_csv)
```

*Explanation:* This script creates a header row specifying the relevant attributes for each media file. The code then extracts information from each track in the file, such as general track format, duration, and bitrate, as well as video track resolution and audio track codecs and bitrate and sample rate. An important feature of this code is it only captures the first audio track, which greatly simplifies the output. The extracted information is stored as a list, and then the writerow method writes the header and then all the data to the csv file. This facilitates easy import into spreadsheets or further processing with tools that require tabular data.

**Resource Recommendations:**

For a deep dive into the MediaInfo library itself, consult its official documentation, which provides granular details about the available parameters and data points. For general Python programming, I'd recommend exploring resources that focus on file system interactions with the os and pathlib modules. Specifically, for working with structured data, examine online guides for JSON and CSV processing in Python.  Understanding the fundamentals of looping, error handling using try-except blocks, list and dict handling, and reading and writing files with open() are prerequisites to being able to modify these examples as needed.

In conclusion, `pymediainfo` offers a robust framework for media metadata extraction.  By combining it with Python’s scripting capabilities, it is possible to streamline the process of generating multiple reports in various formats.  By focusing on different output formats, I have addressed a variety of use cases, from basic text inspections to complex data integrations with JSON or CSV. Each method provides a specific advantage based on the desired use case. The provided code can be extended to match specific requirements, and can be used as a foundation for customized media analysis workflows.
