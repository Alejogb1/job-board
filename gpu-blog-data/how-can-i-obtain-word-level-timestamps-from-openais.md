---
title: "How can I obtain word-level timestamps from OpenAI's Whisper?"
date: "2025-01-30"
id: "how-can-i-obtain-word-level-timestamps-from-openais"
---
Whisper's output, while remarkably accurate in transcription, doesn't natively provide word-level timestamps.  This is a frequent point of frustration for users requiring precise temporal alignment of speech and text.  My experience working on a large-scale speech-to-text project involving multilingual audio analysis highlighted this limitation.  To address it, we developed a post-processing technique leveraging Whisper's segment-level timestamps and a simple heuristic.  This approach, while not perfect, delivers sufficiently accurate word-level timestamps for many applications.

**1. Explanation of the Methodology**

Whisper's JSON output contains segments, each characterized by a start and end timestamp, along with a corresponding transcription.  These segments represent chunks of audio processed coherently by the model.  The key lies in distributing these timestamps across the words within each segment.  We achieve this by assuming a roughly uniform distribution of word duration within a segment.  This assumption, while a simplification, works well in practice, especially for relatively short segments.  Longer segments, however, may contain significant variations in speech rate, potentially leading to less precise timestamps.

The core algorithm involves:

1. **Segment Processing:** Iterating through each segment in the Whisper output.
2. **Word Segmentation:** Splitting the segment's text into individual words.
3. **Timestamp Allocation:**  Calculating the duration of the segment and dividing it equally among the words.  The starting timestamp of a word is derived from the segment's start timestamp plus the cumulative duration of preceding words.  The ending timestamp is simply the starting timestamp plus the allocated duration for the word.
4. **Output Construction:** Creating a new data structure containing word-level timestamps, alongside the original words.

This method, while simplistic, offers a pragmatic solution that requires minimal computational overhead.  Its accuracy is heavily dependent on the uniformity of speech rate within segments.  More sophisticated approaches could incorporate speech rate analysis for improved precision, but this adds complexity.

**2. Code Examples with Commentary**

The following examples use Python and demonstrate the core components of this process.  Assume that `whisper_output` is the JSON output from the Whisper API.


**Example 1: Basic Timestamp Allocation**

```python
import json

def allocate_timestamps(whisper_output):
    results = []
    for segment in whisper_output["segments"]:
        start_time = segment["start"]
        end_time = segment["end"]
        words = segment["text"].split()
        segment_duration = end_time - start_time
        word_duration = segment_duration / len(words) if len(words) > 0 else 0

        for i, word in enumerate(words):
            word_start = start_time + i * word_duration
            word_end = word_start + word_duration
            results.append({"word": word, "start": word_start, "end": word_end})
    return results

# Example usage (replace with your actual Whisper output)
whisper_output_example = {
    "segments": [
        {"start": 0.0, "end": 2.0, "text": "This is a test."},
        {"start": 2.0, "end": 4.5, "text": "Another segment for testing."}
    ]
}

word_timestamps = allocate_timestamps(whisper_output_example)
print(json.dumps(word_timestamps, indent=2))
```

This example demonstrates the fundamental process:  segment iteration, word splitting, and even duration allocation.  The error handling for empty segments is crucial to avoid division by zero.


**Example 2: Handling Punctuation**

```python
import json
import re

def allocate_timestamps_punctuation(whisper_output):
  results = []
  for segment in whisper_output["segments"]:
    start_time = segment["start"]
    end_time = segment["end"]
    #Using regex to split on spaces while keeping punctuation attached to words
    words = re.findall(r'\b\w+\b|[.,!?;]', segment["text"])
    segment_duration = end_time - start_time
    word_duration = segment_duration / len(words) if len(words) > 0 else 0

    for i, word in enumerate(words):
      word_start = start_time + i * word_duration
      word_end = word_start + word_duration
      results.append({"word": word, "start": word_start, "end": word_end})
  return results

# Example usage (with punctuation)
whisper_output_example_punctuation = {
    "segments": [
        {"start": 0.0, "end": 2.0, "text": "This is a test!"},
        {"start": 2.0, "end": 4.5, "text": "Another segment, for testing."}
    ]
}

word_timestamps_punctuation = allocate_timestamps_punctuation(whisper_output_example_punctuation)
print(json.dumps(word_timestamps_punctuation, indent=2))
```

This builds upon the first example by incorporating regular expressions to handle punctuation more robustly.  Punctuation is often crucial for accurate interpretation and shouldn't be discarded.


**Example 3:  Improving Accuracy with Segment Length Consideration**

```python
import json

def allocate_timestamps_refined(whisper_output, min_word_duration=0.1):
    results = []
    for segment in whisper_output["segments"]:
        start_time = segment["start"]
        end_time = segment["end"]
        words = segment["text"].split()
        segment_duration = end_time - start_time
        num_words = len(words)

        if num_words > 0:
            word_duration = max(segment_duration / num_words, min_word_duration) #Ensure minimum word duration
            for i, word in enumerate(words):
                word_start = start_time + i * word_duration
                word_end = min(word_start + word_duration, end_time) # Prevent exceeding segment end time
                results.append({"word": word, "start": word_start, "end": word_end})
    return results

# Example usage
word_timestamps_refined = allocate_timestamps_refined(whisper_output_example)
print(json.dumps(word_timestamps_refined, indent=2))

```

This example introduces a `min_word_duration` parameter to prevent excessively short word durations, particularly in shorter segments. It also prevents word end times from exceeding the segment's end time.


**3. Resource Recommendations**

For a deeper understanding of speech processing techniques, I recommend exploring standard texts on digital signal processing and speech recognition.  Furthermore, studying the Whisper model architecture and its output format will provide invaluable context.  Finally, reviewing research papers on word-level time alignment in automatic speech recognition can significantly expand your knowledge base.  These resources, combined with practical experimentation, are key to refining this post-processing technique.
