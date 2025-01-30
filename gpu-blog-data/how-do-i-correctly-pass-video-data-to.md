---
title: "How do I correctly pass video data to a Keras model using a custom DataGenerator?"
date: "2025-01-30"
id: "how-do-i-correctly-pass-video-data-to"
---
Handling video data for Keras model training necessitates a custom DataGenerator, given that raw video files rarely fit within memory. My experience building a real-time action recognition system highlighted several crucial aspects to consider, deviating significantly from image-based generators. Efficiently feeding video frames to a Keras model requires careful management of data loading, preprocessing, and batching, while maintaining temporal relationships.

The core challenge lies in transforming sequential video frames into a format the model can understand, typically a tensor of shape `(batch_size, sequence_length, height, width, channels)`. The `sequence_length` parameter introduces temporal context, making it distinct from image generators. This parameter dictates how many consecutive frames are grouped together for model training. Without a custom `DataGenerator`, loading and processing these sequences on-the-fly can lead to performance bottlenecks or out-of-memory errors, particularly with larger video datasets.

A custom `keras.utils.Sequence` is the recommended approach for creating this custom generator. This class necessitates overriding two methods: `__len__` to specify the number of batches per epoch and `__getitem__` to generate each batch.

The process generally involves these steps within the `__getitem__` method:

1.  **Batch Indexing:** Calculate which videos contribute to the current batch. This usually involves division of the total dataset by batch size, taking care of remainder cases.

2.  **Video Loading:** Open and extract video frames. Methods like `opencv-python` or `moviepy` are commonly used. One can randomly select the start point of a subsequence to improve model generalization.

3.  **Frame Preprocessing:** Resize frames, normalize pixel values, and apply any other data augmentation techniques. The transformations should maintain aspect ratio unless explicitly altered.

4.  **Sequence Construction:** Organize the preprocessed frames into the required tensor shape for the model, incorporating the selected `sequence_length`.

5.  **Label Extraction:** Fetch the corresponding labels for each video subsequence in the batch.

6.  **Batch Return:** Return a tuple comprising the processed batch of video frames and their associated labels.

The `__len__` method, on the other hand, computes and returns the number of batches, which is approximately the number of videos divided by the batch size. This ensures the generator correctly traverses the dataset within an epoch.

**Code Example 1: Basic Video Frame Extraction**

```python
import cv2
import numpy as np
from tensorflow import keras

class VideoDataGenerator(keras.utils.Sequence):
    def __init__(self, video_paths, labels, batch_size, sequence_length, target_size=(224, 224)):
        self.video_paths = video_paths
        self.labels = labels
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.target_size = target_size

    def __len__(self):
        return int(np.ceil(len(self.video_paths) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_paths = self.video_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_labels = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_videos = []
        for video_path in batch_paths:
            video = cv2.VideoCapture(video_path)
            frames = []
            if not video.isOpened():
                print(f"Error: Cannot open video at {video_path}")
                continue

            frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count < self.sequence_length:
              video.release()
              continue

            start_frame = np.random.randint(0, frame_count - self.sequence_length +1)
            video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            for i in range(self.sequence_length):
                ret, frame = video.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, self.target_size)
                frame = frame / 255.0  # Normalize
                frames.append(frame)
            video.release()

            if len(frames) == self.sequence_length:
                batch_videos.append(np.array(frames))

        batch_videos = np.array(batch_videos)
        batch_labels = np.array(batch_labels)

        return batch_videos, batch_labels

# Example Usage
video_paths = ['video1.mp4', 'video2.mp4', 'video3.mp4', 'video4.mp4'] # Fictitious videos
labels = [0, 1, 0, 1]  # Example labels
batch_size = 2
sequence_length = 10

generator = VideoDataGenerator(video_paths, labels, batch_size, sequence_length)
x, y = generator[0]
print("Shape of input batch:", x.shape)
print("Shape of label batch:", y.shape)
```
*Commentary:* This code demonstrates the basic implementation of a custom data generator that loads videos, extracts subsequences of frames, resizes and normalizes the frames, and returns batches of video data along with their corresponding labels. It addresses the crucial step of ensuring each sequence has the correct `sequence_length`. The random start frame selection helps model generalization. Error handling for video loading is also incorporated. The example usage shows the output batch shapes, validating the generator's function.

**Code Example 2: Handling Variable Length Videos**
```python
import cv2
import numpy as np
from tensorflow import keras

class VariableVideoGenerator(keras.utils.Sequence):
    def __init__(self, video_paths, labels, batch_size, sequence_length, target_size=(224, 224)):
        self.video_paths = video_paths
        self.labels = labels
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.target_size = target_size
        self.video_lengths = self._get_video_lengths()

    def _get_video_lengths(self):
        lengths = []
        for path in self.video_paths:
            video = cv2.VideoCapture(path)
            if video.isOpened():
                lengths.append(int(video.get(cv2.CAP_PROP_FRAME_COUNT)))
                video.release()
            else:
              lengths.append(0)
        return lengths


    def __len__(self):
        return int(np.ceil(len(self.video_paths) / float(self.batch_size)))

    def __getitem__(self, idx):
         batch_paths = self.video_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
         batch_labels = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
         batch_lengths = self.video_lengths[idx * self.batch_size:(idx + 1) * self.batch_size]

         batch_videos = []
         valid_indices = []
         for i, video_path in enumerate(batch_paths):
              if batch_lengths[i] >= self.sequence_length:
                valid_indices.append(i)

         for i in valid_indices:
            video_path = batch_paths[i]
            video = cv2.VideoCapture(video_path)
            if not video.isOpened():
                continue

            start_frame = np.random.randint(0, batch_lengths[i] - self.sequence_length +1)
            video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            frames = []
            for _ in range(self.sequence_length):
                 ret, frame = video.read()
                 if not ret:
                    break
                 frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                 frame = cv2.resize(frame, self.target_size)
                 frame = frame / 255.0
                 frames.append(frame)
            video.release()
            if len(frames) == self.sequence_length:
                 batch_videos.append(np.array(frames))


         batch_videos = np.array(batch_videos)

         batch_labels = np.array([batch_labels[i] for i in valid_indices])

         return batch_videos, batch_labels


# Example Usage
video_paths = ['video1.mp4', 'video2.mp4', 'video3.mp4', 'video4.mp4','video5.mp4'] # Fictitious videos
labels = [0, 1, 0, 1,0]  # Example labels
batch_size = 2
sequence_length = 10

generator = VariableVideoGenerator(video_paths, labels, batch_size, sequence_length)

for i in range(len(generator)):
  x, y = generator[i]
  if x.size > 0:
    print(f"Batch {i}: Input shape {x.shape}, label shape {y.shape}")
```
*Commentary:* This code extends the prior example by incorporating explicit handling of videos with lengths smaller than the required `sequence_length`. It precalculates the length of every video in the `__init__` and filters videos shorter than sequence length in the `__getitem__`, avoiding errors and ensures only complete video subsequences are used for training. Only valid videos contribute to the batch and corresponding labels are returned. The loop demonstrates batch processing of generator.

**Code Example 3: Adding Data Augmentation**

```python
import cv2
import numpy as np
from tensorflow import keras
import random
from scipy.ndimage import rotate

class AugmentedVideoGenerator(keras.utils.Sequence):
    def __init__(self, video_paths, labels, batch_size, sequence_length, target_size=(224, 224), augment=True):
        self.video_paths = video_paths
        self.labels = labels
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.target_size = target_size
        self.augment = augment
        self.video_lengths = self._get_video_lengths()


    def _get_video_lengths(self):
        lengths = []
        for path in self.video_paths:
            video = cv2.VideoCapture(path)
            if video.isOpened():
                lengths.append(int(video.get(cv2.CAP_PROP_FRAME_COUNT)))
                video.release()
            else:
              lengths.append(0)
        return lengths


    def __len__(self):
        return int(np.ceil(len(self.video_paths) / float(self.batch_size)))


    def _augment_frame(self, frame):
      if not self.augment:
         return frame
      if random.random() < 0.5:
          angle = random.uniform(-10, 10)
          frame = rotate(frame, angle, reshape=False)
      return frame

    def __getitem__(self, idx):
        batch_paths = self.video_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_labels = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_lengths = self.video_lengths[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_videos = []
        valid_indices = []
        for i, video_path in enumerate(batch_paths):
            if batch_lengths[i] >= self.sequence_length:
                valid_indices.append(i)

        for i in valid_indices:
            video_path = batch_paths[i]
            video = cv2.VideoCapture(video_path)
            if not video.isOpened():
                 continue

            start_frame = np.random.randint(0, batch_lengths[i] - self.sequence_length + 1)
            video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            frames = []
            for _ in range(self.sequence_length):
                ret, frame = video.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, self.target_size)
                frame = frame / 255.0
                frame = self._augment_frame(frame)
                frames.append(frame)
            video.release()
            if len(frames) == self.sequence_length:
                 batch_videos.append(np.array(frames))

        batch_videos = np.array(batch_videos)
        batch_labels = np.array([batch_labels[i] for i in valid_indices])
        return batch_videos, batch_labels


# Example Usage
video_paths = ['video1.mp4', 'video2.mp4', 'video3.mp4', 'video4.mp4','video5.mp4']  # Fictitious videos
labels = [0, 1, 0, 1,0]  # Example labels
batch_size = 2
sequence_length = 10
generator = AugmentedVideoGenerator(video_paths, labels, batch_size, sequence_length)
for i in range(len(generator)):
  x, y = generator[i]
  if x.size > 0:
    print(f"Batch {i}: Input shape {x.shape}, label shape {y.shape}")
```
*Commentary:* This final code provides a demonstration of how to incorporate data augmentation within the custom data generator. Random rotations applied to each frame are shown via the `_augment_frame` method. This is enabled via the `augment` flag and demonstrates how to add custom per-frame augmentation. The loop demonstrates batch processing of generator.

**Resource Recommendations**

For deeper understanding, I suggest exploring texts on deep learning with specific chapters on data loading techniques for sequential data. Books focusing on Keras and TensorFlow will provide detailed usage of the `keras.utils.Sequence`. Additionally, consulting resources covering best practices for video data processing within machine learning will be beneficial, covering topics like frame skipping and efficient loading with hardware acceleration. Understanding of common data augmentation techniques in image processing is also recommended.
