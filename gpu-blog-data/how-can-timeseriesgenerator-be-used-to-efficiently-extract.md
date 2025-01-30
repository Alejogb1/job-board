---
title: "How can TimeseriesGenerator be used to efficiently extract frames from large video files?"
date: "2025-01-30"
id: "how-can-timeseriesgenerator-be-used-to-efficiently-extract"
---
The primary challenge when working with video data for machine learning lies in memory management. Loading an entire video into RAM, especially for high-resolution or long-duration videos, becomes rapidly infeasible. This is where techniques for efficient, frame-by-frame processing are critical, and `TimeseriesGenerator` from Keras provides one approach when frames are treated as a time series. I've encountered this specific challenge multiple times during model training for action recognition and video anomaly detection, where the sheer size of the datasets made naive loading mechanisms unsustainable.

`TimeseriesGenerator`, traditionally used for sequential time series data such as sensor readings or stock prices, can be adapted to work effectively with video frames by treating the frame sequence as a temporal sequence. Its core functionality revolves around generating batches of time series data from a larger, ordered dataset. The key to efficient video processing using `TimeseriesGenerator` lies in understanding how to load video frames on demand, rather than preloading everything. Instead of directly feeding pixel data to it, we’ll feed the generator indices of frame positions, use the generator to access those indices and retrieve them from a disk based video reader. This allows us to keep a smaller memory footprint, with data loaded in batches as needed during model training. The Keras `TimeseriesGenerator` is designed to work with numerical data, and by transforming the frame indices to match its expected input, we use the generator to create sequences of frame locations which then are used to load image data.

Here’s a breakdown of the process:

1. **Video Frame Extraction (On Demand):**  We do not extract all frames beforehand. Instead, we employ a library like OpenCV (cv2) or moviepy which allow to open a video file and read frame by frame by specifying the location of the frame (or its index). These libraries offer frame-accurate reading which is crucial for avoiding inconsistencies in your data.
2. **Frame Index as Timeseries:** We establish a simple sequence of frame indices (0, 1, 2, 3…), essentially representing the "time" dimension for our video data. This is the input for the `TimeseriesGenerator`.
3. **`TimeseriesGenerator` Configuration:** We initialize the `TimeseriesGenerator` with this sequence of frame indices. Parameters such as `length` (the number of consecutive frames to include in each batch) and `stride` (the interval between the starting points of consecutive sequences) are configured.
4. **Batch Generation with Custom Data Loading:** As the `TimeseriesGenerator` produces batches of frame *indices*, we'll have our custom code fetch corresponding image frames from the video file. This is how on-demand loading is achieved, preventing the entire video from being loaded in the memory at the beginning.

Consider the following Python code example using OpenCV:

```python
import cv2
import numpy as np
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

def load_video_frames(video_path, frame_indices, target_size=(64, 64)):
    """Loads and preprocesses video frames based on provided indices.

    Args:
        video_path: Path to the video file.
        frame_indices: List of frame indices to load.
        target_size: Size of the image to resize frames to

    Returns:
        NumPy array of preprocessed frames, or None if error encountered.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video file: {video_path}")
            return None
        frames = []
        for frame_index in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, target_size)
                frame = frame / 255.0  # Normalize pixel values
                frames.append(frame)
            else:
                print(f"Warning: Could not read frame at index: {frame_index} in {video_path}")
        cap.release()
        return np.array(frames)
    except Exception as e:
        print(f"Error loading frames: {e}")
        return None


def video_data_generator(video_path, num_frames, batch_size, length, stride, target_size=(64,64)):
    """Generates batches of video data using TimeSeriesGenerator.

    Args:
        video_path: Path to the video file.
        num_frames: Total number of frames in the video.
        batch_size: Desired batch size.
        length: Number of consecutive frames in each sequence.
        stride: Spacing between sequence start points.
    """
    frame_indices = np.arange(num_frames)
    generator = TimeseriesGenerator(frame_indices, frame_indices, length=length, sampling_rate=1, stride=stride, batch_size=batch_size)
    
    while True:
        batch_indices = next(generator)
        batch_frame_indices = batch_indices[0]

        batch_images = []
        for indices in batch_frame_indices:
           frame_batch = load_video_frames(video_path, indices.tolist(), target_size=target_size)
           if frame_batch is not None:
               batch_images.append(frame_batch)
            
        if len(batch_images) == 0:
            continue
        yield np.array(batch_images)

# Example Usage
video_path = 'my_video.mp4' # Replace with the path to your video
cap = cv2.VideoCapture(video_path)
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

batch_size = 4
length = 10 # Number of consecutive frames in a sequence
stride = 5 # Offset between sequences
target_size = (64, 64)

video_gen = video_data_generator(video_path, num_frames, batch_size, length, stride, target_size)

for i in range(3):  # Example: Fetching and printing shapes of 3 batches
    batch = next(video_gen)
    print(f"Batch {i+1} shape: {batch.shape}")
```

This code defines `load_video_frames` to handle frame extraction and preprocessing. The `video_data_generator` sets up a `TimeseriesGenerator` with an array of frame indices. Instead of the typical time-series data, we treat the frame numbers themselves as the data that gets sequenced by the generator. It then uses this generator to access the frame indices and calls `load_video_frames` for each batch of indices and yields an image batch. The example usage section demonstrates how to call `video_data_generator` and outputs the shape of the created batches, showing the effective sequence and batch creation. The `try-except` block in `load_video_frames` handles potential frame read errors.

Here’s a variant using the `moviepy` library, which is more focused on video editing but also allows for precise frame access:

```python
from moviepy.editor import VideoFileClip
import numpy as np
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import math

def load_video_frames_moviepy(video_path, frame_indices, target_size=(64, 64)):
    """Loads and preprocesses video frames using moviepy.

    Args:
        video_path: Path to the video file.
        frame_indices: List of frame indices to load.
        target_size: Size of the image to resize frames to

    Returns:
        NumPy array of preprocessed frames, or None if error encountered.
    """
    try:
        clip = VideoFileClip(video_path)
        frames = []
        for frame_index in frame_indices:
            try:
                frame = clip.get_frame(frame_index/clip.fps) # Get frame at specific second
                frame = np.array(frame)
                frame = cv2.resize(frame, target_size)
                frame = frame / 255.0  # Normalize pixel values
                frames.append(frame)

            except IndexError:
                print(f"Warning: Could not read frame at index: {frame_index} in {video_path}")
        clip.close()
        return np.array(frames)
    except Exception as e:
        print(f"Error loading frames: {e}")
        return None


def video_data_generator_moviepy(video_path, num_frames, batch_size, length, stride, target_size=(64,64)):
    """Generates batches of video data using TimeSeriesGenerator and moviepy.

    Args:
        video_path: Path to the video file.
        num_frames: Total number of frames in the video.
        batch_size: Desired batch size.
        length: Number of consecutive frames in each sequence.
        stride: Spacing between sequence start points.
        target_size: Size of the image to resize frames to
    """
    frame_indices = np.arange(num_frames)
    generator = TimeseriesGenerator(frame_indices, frame_indices, length=length, sampling_rate=1, stride=stride, batch_size=batch_size)
    
    while True:
        batch_indices = next(generator)
        batch_frame_indices = batch_indices[0]

        batch_images = []
        for indices in batch_frame_indices:
           frame_batch = load_video_frames_moviepy(video_path, indices.tolist(), target_size=target_size)
           if frame_batch is not None:
               batch_images.append(frame_batch)
            
        if len(batch_images) == 0:
            continue
        yield np.array(batch_images)

# Example Usage
video_path = 'my_video.mp4' # Replace with the path to your video
clip = VideoFileClip(video_path)
num_frames = math.floor(clip.fps * clip.duration)
clip.close()
batch_size = 4
length = 10 # Number of consecutive frames in a sequence
stride = 5 # Offset between sequences
target_size = (64, 64)

video_gen = video_data_generator_moviepy(video_path, num_frames, batch_size, length, stride, target_size)


for i in range(3):  # Example: Fetching and printing shapes of 3 batches
    batch = next(video_gen)
    print(f"Batch {i+1} shape: {batch.shape}")

```

This `moviepy` version operates similarly, but uses `VideoFileClip` to load video and `get_frame` with frame indices converted to time to precisely extract individual frames. This method can sometimes be more convenient for frame-accurate reads, particularly if the video has variable frame rates (even though that is not taken into account).  Again, error handling is included.

Finally, for applications where you need specific data augmentation during loading, incorporate a separate function that can be called inside `load_video_frames` to apply necessary image transforms. This is essential to prevent overfitting and improve the robustness of your models. For example:

```python
import cv2
import numpy as np
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import math
import random

def random_brightness(frame, delta=0.2):
    """Applies a random brightness adjustment."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    random_bright = 1 + random.uniform(-delta, delta)
    hsv[:,:,2] = np.clip(hsv[:,:,2] * random_bright, 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def data_augment(frame, augment_chance = 0.5):
    """Applies random augmentation to the input frame."""
    if random.random() < augment_chance:
        frame = random_brightness(frame)
    return frame


def load_video_frames_augmented(video_path, frame_indices, target_size=(64, 64)):
    """Loads, augments and preprocesses video frames based on provided indices.

    Args:
        video_path: Path to the video file.
        frame_indices: List of frame indices to load.
        target_size: Size of the image to resize frames to

    Returns:
        NumPy array of preprocessed frames, or None if error encountered.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video file: {video_path}")
            return None
        frames = []
        for frame_index in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, target_size)
                frame = data_augment(frame)
                frame = frame / 255.0  # Normalize pixel values
                frames.append(frame)
            else:
                print(f"Warning: Could not read frame at index: {frame_index} in {video_path}")
        cap.release()
        return np.array(frames)
    except Exception as e:
        print(f"Error loading frames: {e}")
        return None

def video_data_generator_augmented(video_path, num_frames, batch_size, length, stride, target_size=(64,64)):
    """Generates batches of augmented video data using TimeSeriesGenerator.

    Args:
        video_path: Path to the video file.
        num_frames: Total number of frames in the video.
        batch_size: Desired batch size.
        length: Number of consecutive frames in each sequence.
        stride: Spacing between sequence start points.
    """
    frame_indices = np.arange(num_frames)
    generator = TimeseriesGenerator(frame_indices, frame_indices, length=length, sampling_rate=1, stride=stride, batch_size=batch_size)
    
    while True:
        batch_indices = next(generator)
        batch_frame_indices = batch_indices[0]

        batch_images = []
        for indices in batch_frame_indices:
           frame_batch = load_video_frames_augmented(video_path, indices.tolist(), target_size=target_size)
           if frame_batch is not None:
               batch_images.append(frame_batch)
            
        if len(batch_images) == 0:
            continue
        yield np.array(batch_images)

# Example Usage
video_path = 'my_video.mp4' # Replace with the path to your video
cap = cv2.VideoCapture(video_path)
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

batch_size = 4
length = 10 # Number of consecutive frames in a sequence
stride = 5 # Offset between sequences
target_size = (64, 64)

video_gen = video_data_generator_augmented(video_path, num_frames, batch_size, length, stride, target_size)

for i in range(3):  # Example: Fetching and printing shapes of 3 batches
    batch = next(video_gen)
    print(f"Batch {i+1} shape: {batch.shape}")

```

This example adds a `data_augment` function which calls `random_brightness` to adjust image brightness randomly. The `load_video_frames_augmented` function then incorporates data augmentation to the frames before they are normalized.

For further study, I suggest exploring the documentation and usage examples of the following resources: the OpenCV library for its video reading and writing capabilities; the moviepy library for a Pythonic interface to videos; and of course, Keras documentation for detailed information on `TimeseriesGenerator` and data preprocessing techniques. Furthermore, research on common data augmentation techniques (like those implemented above) are also very helpful. In practice, this combination of techniques yields a scalable way to work with even large video datasets.
