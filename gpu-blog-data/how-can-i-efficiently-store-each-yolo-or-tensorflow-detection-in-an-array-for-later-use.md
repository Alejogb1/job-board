---
title: "How can I efficiently store each YOLO or TensorFlow detection in an array for later use?"
date: "2025-01-26"
id: "how-can-i-efficiently-store-each-yolo-or-tensorflow-detection-in-an-array-for-later-use"
---

The fundamental challenge in storing YOLO or TensorFlow object detections lies in managing the potentially variable number of bounding boxes and associated confidence scores generated per frame. Each detection result is typically a collection of these elements, and naive storage directly in a standard Python list can become inefficient for larger or longer video sequences. My experience building real-time object tracking systems has shown the need for structured data handling to maintain performance.

Here’s the approach I’ve found most effective for storing these detections: I avoid the use of a simple list. Instead, I create a specialized array, typically a NumPy array, or when more dynamic resizing is needed, a dictionary-based structure that permits individual frame detection information to be easily associated with frame IDs, timestamps or another temporal key. This provides faster numerical computation, efficient memory access, and facilitates downstream processing. The key is structuring the information for efficient retrieval. This method contrasts to storing detection information as lists of lists, which I've found cumbersome for analysis, or worse, lists of dictionaries, which while flexible, lack the performance benefits of numerical arrays when I often perform computation on these values.

To break it down, each detection usually consists of the following:

1. **Bounding Box Coordinates:** Represented as four numbers: x_min, y_min, x_max, y_max (or often as x_center, y_center, width, height).
2. **Confidence Score:** A numerical value indicating the model's confidence in the detection.
3. **Class ID:** An integer representing the detected object's class.

My chosen storage structure directly maps these components to numerical array columns. For each video frame or image, detections are stored as rows. This approach also readily supports batch processing, which is vital in my use case.

**Code Example 1: NumPy Array for Static Frame Handling**

This example demonstrates how to create a NumPy array to store detection results when the number of detections per frame is reasonably constant or a maximum number of detections is assumed and padded to the end of array with ‘Nan’ values.

```python
import numpy as np

def store_detections_numpy(detections, frame_count, max_detections=5):
    """Stores detections in a NumPy array.

    Args:
        detections: A list of lists, where each sublist contains
                    [x_min, y_min, x_max, y_max, confidence, class_id].
        frame_count: Integer frame number.
        max_detections: Maximum detections to be stored per frame

    Returns:
        A NumPy array, or None if detections is an empty list.
    """

    if not detections:
        return None

    # Initialize an array to hold max number of detections with NaN padding.
    # Detection data has 6 columns: x_min, y_min, x_max, y_max, confidence, class_id
    frame_detections = np.full((max_detections, 6), np.nan)
    num_detections = len(detections)
    
    # Copy detection data into the numpy array if within the limit
    for i in range(min(num_detections, max_detections)):
        frame_detections[i, :] = detections[i]
    
    # Returns array with detections padded with NaN values if there are fewer than max_detections.
    return frame_detections

# Example usage:
frame_number = 1
detections = [[10, 20, 100, 120, 0.9, 1], [150, 50, 200, 200, 0.85, 2], [300, 100, 350, 250, 0.95, 3]]
stored_array = store_detections_numpy(detections, frame_number)
print(f"Stored Detections (frame {frame_number}):")
print(stored_array)

frame_number = 2
detections2 = [[10, 20, 100, 120, 0.9, 1], [150, 50, 200, 200, 0.85, 2]]
stored_array2 = store_detections_numpy(detections2, frame_number)
print(f"Stored Detections (frame {frame_number}):")
print(stored_array2)

frame_number = 3
detections3 = []
stored_array3 = store_detections_numpy(detections3, frame_number)
print(f"Stored Detections (frame {frame_number}):")
print(stored_array3)
```

This function, `store_detections_numpy`, takes a list of detections and frame number and converts the detections into a NumPy array, padding unused row slots with `NaN` values up to `max_detections`. This allows you to store the detections in a more structured manner than a simple list of lists. It is important that all sublists in detections have a fixed length. This is a good example when you know that there will always be less than a given number of detections for each frame.

**Code Example 2: Dictionary with NumPy Arrays for Flexible Frame Handling**

This example uses a dictionary to store the detection results where each key is the frame ID and each value is a NumPy array storing the detections associated with that frame. This structure allows a variable number of detections per frame. This is the structure that I use when I don’t know the max detections in advance, which is the most common use-case.

```python
import numpy as np

def store_detections_dict(detections, frame_number, stored_data=None):
    """Stores detections in a dictionary where each key is frame id.

    Args:
        detections: A list of lists, where each sublist contains
                    [x_min, y_min, x_max, y_max, confidence, class_id].
        frame_number: Integer frame id or a string representing the frame.
        stored_data: an optional dictionary containing previously stored data.

    Returns:
        A dictionary with keys representing frame IDs and values of NumPy arrays.
    """
    if stored_data is None:
        stored_data = {}

    if not detections:
      stored_data[frame_number] = None
      return stored_data

    # Convert detections list to a numpy array.
    frame_detections = np.array(detections)
    stored_data[frame_number] = frame_detections

    return stored_data

# Example usage:
stored_data = {}
frame_number = 1
detections = [[10, 20, 100, 120, 0.9, 1], [150, 50, 200, 200, 0.85, 2], [300, 100, 350, 250, 0.95, 3]]
stored_data = store_detections_dict(detections, frame_number, stored_data)
print(f"Stored Detections (frame {frame_number}):")
print(stored_data[frame_number])

frame_number = 2
detections2 = [[10, 20, 100, 120, 0.9, 1], [150, 50, 200, 200, 0.85, 2]]
stored_data = store_detections_dict(detections2, frame_number, stored_data)
print(f"Stored Detections (frame {frame_number}):")
print(stored_data[frame_number])

frame_number = 3
detections3 = []
stored_data = store_detections_dict(detections3, frame_number, stored_data)
print(f"Stored Detections (frame {frame_number}):")
print(stored_data[frame_number])

print(f"All stored detections: {stored_data}")
```

In this example, the `store_detections_dict` function creates a dictionary where the keys are frame numbers and values are NumPy arrays. This dictionary structure allows me to easily look up detection results associated with a specific frame number. If no detections are found, the key for that frame is still in the dictionary and associated with the `None` object.

**Code Example 3: Conversion to Pandas DataFrame**

This example illustrates the conversion of the previous dictionary to a Pandas DataFrame. The key benefit of this conversion is that the results can be readily analyzed. I usually do all my analysis after I have stored the detections using the `store_detections_dict` method.

```python
import pandas as pd
import numpy as np

def dict_to_dataframe(stored_data):
  """Converts a dictionary to a pandas DataFrame.

    Args:
        stored_data: A dictionary where keys are frame IDs and
                     values are NumPy arrays.

    Returns:
        A pandas DataFrame.
    """
  data_rows = []
  for frame, detections in stored_data.items():
    if detections is not None:
      for detection in detections:
        data_rows.append([frame, *detection])
    else:
        data_rows.append([frame, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])

  columns = ['frame_id', 'x_min', 'y_min', 'x_max', 'y_max', 'confidence', 'class_id']
  df = pd.DataFrame(data_rows, columns=columns)
  return df

stored_data = {}
frame_number = 1
detections = [[10, 20, 100, 120, 0.9, 1], [150, 50, 200, 200, 0.85, 2], [300, 100, 350, 250, 0.95, 3]]
stored_data = store_detections_dict(detections, frame_number, stored_data)

frame_number = 2
detections2 = [[10, 20, 100, 120, 0.9, 1], [150, 50, 200, 200, 0.85, 2]]
stored_data = store_detections_dict(detections2, frame_number, stored_data)

frame_number = 3
detections3 = []
stored_data = store_detections_dict(detections3, frame_number, stored_data)

df = dict_to_dataframe(stored_data)
print(df)
```

The `dict_to_dataframe` function iterates through the dictionary created in the previous example and constructs a Pandas DataFrame for analysis, and handles the case where no detections occur for a given frame.  This provides a table format that I've found easier to work with for analysis than working with nested lists or dictionary.  The Pandas library includes numerous tools to facilitate further processing.

In summary, for efficient storage of object detections, avoid unstructured lists. Employ NumPy arrays and dictionary structures for flexible and performant data management. When further analysis is required, the conversion to Pandas DataFrames is very helpful.

For further study, consider researching NumPy’s array broadcasting and vectorized operations to see how they can further optimize numerical operations on stored data. Explore using HDF5 to persist NumPy arrays for efficient large datasets, and delve deeper into Pandas for time-series analysis. These are all resources I frequently consult. Finally, the SciPy package contains algorithms for a wide variety of numerical processing routines that you may find useful for your analysis.
