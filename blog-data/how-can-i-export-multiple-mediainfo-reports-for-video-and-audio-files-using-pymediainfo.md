---
title: "How can I export multiple MediaInfo reports for video and audio files using pymediainfo?"
date: "2024-12-23"
id: "how-can-i-export-multiple-mediainfo-reports-for-video-and-audio-files-using-pymediainfo"
---

Let's tackle this challenge of automating media information extraction using pymediainfo. This is a task I've encountered several times over the years, particularly when building video processing pipelines, and the need to consistently pull metadata across large batches of files is quite common. The basic single-file extraction is fairly straightforward, but the requirements become different when dealing with potentially hundreds or thousands of files. We need efficiency, resilience, and a clean output format. I've often found that building a robust solution here saves considerable time in the long run.

First, let's talk about the inherent challenge: looping through many files while ensuring each `MediaInfo` instance is properly created, used, and then disposed of to avoid any unexpected memory usage or resource leaks. We cannot simply create one instance and reuse it, as the data is inherently tied to a given file. We'll also want to consider error handling. Media files can be corrupt, malformed, or missing entirely. Ignoring these can lead to inaccurate or incomplete results and a much less reliable process. And finally, for practical use, exporting to a readable format, such as a structured format like json or csv, is preferable to just printing to the console.

Okay, so let’s begin with the basic approach, moving through a more efficient and practical one, and then discussing a few caveats.

**Example 1: Basic Iteration and JSON Output**

This example focuses on looping through a list of files, calling `pymediainfo` for each, and producing a json output. It is not the most performant, as it opens each file sequentially, but it is a good starting point.

```python
import os
import json
from pymediainfo import MediaInfo

def extract_media_info_basic(file_list, output_file):
    all_media_data = []
    for file_path in file_list:
        try:
            media_info = MediaInfo.parse(file_path)
            if media_info.tracks:  # Ensure we got something useful
                all_media_data.append(media_info.to_json())
            else:
                print(f"Warning: No media info found for file: {file_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    with open(output_file, 'w') as f:
        json.dump(all_media_data, f, indent=4)


if __name__ == "__main__":
    # Example usage (replace with your actual files)
    example_files = ['video1.mp4', 'audio1.wav', 'video2.mkv', 'invalid_file.txt']  # Create dummy files
    for f in example_files:
      open(f,'a').close() #create empty dummy files
    
    extract_media_info_basic(example_files, 'media_info_output.json')

    for f in example_files:
      os.remove(f)  #cleanup dummy files
    
    
    print("Media info extraction complete. Check 'media_info_output.json'")

```

This code iterates through a list of files. Inside the loop, `MediaInfo.parse(file_path)` creates a `MediaInfo` object that contains information for each individual file. It is crucial to use the try-except block to catch issues with file access or parsing. The `to_json()` method serializes all the collected data to a json-compatible string, which can be aggregated to a list of data and then written to an output file. The indent argument in `json.dump` makes the output human-readable.

**Example 2: Improved Efficiency with Process Pool**

Now let's enhance performance by using multiprocessing. This approach leverages multiple cores to concurrently process several files in parallel, significantly speeding up processing when dealing with large file sets.

```python
import os
import json
from pymediainfo import MediaInfo
from multiprocessing import Pool, cpu_count

def process_file(file_path):
    try:
        media_info = MediaInfo.parse(file_path)
        if media_info.tracks:
             return media_info.to_json()
        else:
            print(f"Warning: No media info found for file: {file_path}")
            return None # returning None to filter out
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None # returning None to filter out

def extract_media_info_parallel(file_list, output_file):
    with Pool(cpu_count()) as pool: # creates pool with processes equals to the number of cores.
        results = pool.map(process_file, file_list)
    
    results = [data for data in results if data is not None ] # remove none results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
     # Example usage (replace with your actual files)
    example_files = ['video1.mp4', 'audio1.wav', 'video2.mkv', 'invalid_file.txt']  # Create dummy files
    for f in example_files:
      open(f,'a').close() #create empty dummy files

    extract_media_info_parallel(example_files, 'media_info_output_parallel.json')

    for f in example_files:
      os.remove(f)  #cleanup dummy files
      
    print("Parallel media info extraction complete. Check 'media_info_output_parallel.json'")

```

Here, `multiprocessing.Pool` creates a pool of worker processes, and the `pool.map()` method distributes the file processing across them. The `cpu_count()` ensures that we utilize the machine's full capabilities. Each file is processed concurrently using `process_file`, which now only returns the json-formatted string or None in case of an issue. The non None results are then filtered into a single output file.  This approach is substantially faster for many files.

**Example 3: CSV output with Customized Columns**

Finally, we might want to tailor the output columns specifically to our needs, and csv might be more suitable than json. This often helps when working with tabular data. We can utilize `csv.writer` module to create this output. Note that extracting the specific fields you need requires a more specific handling based on the structure of the `MediaInfo.parse()` output.

```python
import os
import csv
from pymediainfo import MediaInfo


def extract_media_info_csv(file_list, output_file):
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['filename','file_size', 'duration', 'video_codec', 'audio_codec']) # header row
        for file_path in file_list:
            try:
                media_info = MediaInfo.parse(file_path)
                if media_info.tracks:
                    file_data = {} #initialize per file dict
                    for track in media_info.tracks:
                       if track.track_type == 'General':
                           file_data['filename'] = os.path.basename(track.complete_name)
                           file_data['file_size'] = track.file_size
                           file_data['duration'] = track.duration
                       elif track.track_type == 'Video':
                           file_data['video_codec'] = track.codec
                       elif track.track_type == 'Audio':
                            file_data['audio_codec'] = track.codec
                    csv_writer.writerow([file_data.get('filename',''), file_data.get('file_size',''), file_data.get('duration',''), file_data.get('video_codec',''), file_data.get('audio_codec','') ])
                else:
                    csv_writer.writerow([os.path.basename(file_path), '','','',''])
                    print(f"Warning: No media info found for file: {file_path}")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                csv_writer.writerow([os.path.basename(file_path), '','','',''])


if __name__ == "__main__":
    # Example usage (replace with your actual files)
    example_files = ['video1.mp4', 'audio1.wav', 'video2.mkv', 'invalid_file.txt'] # Create dummy files
    for f in example_files:
      open(f,'a').close() #create empty dummy files
   
    extract_media_info_csv(example_files, 'media_info_output.csv')
    
    for f in example_files:
      os.remove(f)  #cleanup dummy files
    
    print("Media info extraction complete. Check 'media_info_output.csv'")
```

This example writes the extracted metadata to a CSV file. We are handling several `track_types` and accessing the desired properties. The header row is defined, and then the relevant information for each file is added to the csv using `csv.writer`. Notice that `track` properties may not always be present on all files, so accessing them directly may result in errors. I use `file_data.get('key','default')` method to access them with defaults, ensuring that incomplete entries still produce a complete row of data.

For a deeper understanding of `pymediainfo` itself, exploring the official documentation and example scripts is crucial. Specifically, dive into the structure of the output it generates for different media types; this will inform your metadata extraction and structuring. If you’re looking for material on handling large datasets and parallel processing in Python, look into "Fluent Python" by Luciano Ramalho which provides deep insights into python concurrency, among other advanced features, and "Effective Computation in Physics" by Anthony Scopatz and Kathryn D. Huff which discusses many performance-focused computation strategies. Additionally, the documentation for python `multiprocessing` provides all of the specifics regarding the use of process pools, and python's built in `json` and `csv` modules. These, along with the `pymediainfo` resources, provide a thorough foundation.

These examples are a starting point; there are many other adjustments you may need to make based on your specific circumstances. For instance, if you require very granular metadata, you will need to dig deeper into the `MediaInfo` object structure and tailor the output accordingly. Likewise, you may choose other data formats like XML or a database if you need advanced storage or querying capabilities. The key is to adapt these basic strategies to your specific needs, while maintaining a clean, efficient, and robust code base.
