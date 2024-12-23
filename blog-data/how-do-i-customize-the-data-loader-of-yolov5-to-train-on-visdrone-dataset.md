---
title: "How do I customize the data loader of yolov5 to train on VisDrone dataset?"
date: "2024-12-23"
id: "how-do-i-customize-the-data-loader-of-yolov5-to-train-on-visdrone-dataset"
---

Alright, let's tackle this. Customizing the data loader for yolov5, specifically for the VisDrone dataset, is a task I've personally navigated a few times, and it's rarely a one-size-fits-all solution. The core challenge isn’t the underlying yolov5 architecture itself, but rather, adapting its data ingestion mechanisms to align with VisDrone's unique format. We're going to need to delve into the `torch.utils.data.Dataset` class and modify it to handle the specifics of the VisDrone annotation style, which includes bounding boxes and, sometimes, additional metadata you might want to leverage.

Firstly, and perhaps most importantly, VisDrone doesn't come with label files formatted the way yolov5 expects them – which is typically one text file per image, with each line representing a bounding box in normalized (x_center, y_center, width, height) format, along with a class id. VisDrone, instead, commonly uses annotation files (often in text or xml format), containing information about each object present, including bounding boxes in absolute pixel coordinates. These coordinates are presented as (xmin, ymin, xmax, ymax) format, also the classes are in string format. My past projects with custom aerial datasets involved similar discrepancies, and the crucial point is to build a translator. The goal, in essence, is to create a PyTorch dataset that pre-processes these annotations into the format yolov5 is designed to ingest.

Here is an outline of the steps we are going to discuss:

*   **Data Preparation:** We need to process the annotation files to create matching image paths and label information.
*   **Custom Dataset Class Implementation:** Write a PyTorch dataset that extends `torch.utils.data.Dataset` to handle VisDrone data.
*   **Data Augmentation**: Integrate suitable transformations into the custom dataset.

Let's start by diving into the code:

**1. Data Preparation:**

Before we craft a custom dataset, we need some infrastructure to organize the VisDrone dataset. Assume you've already downloaded and extracted VisDrone, and that its structure looks something like this:
```
visdrone/
├── images/
│   ├── train/
│   │   ├── 0000001_00000_d_0000001.jpg
│   │   ├── ...
│   ├── val/
│   │   ├── ...
│   ├── test/
│   │   ├── ...
├── annotations/
│   ├── train/
│   │   ├── 0000001_00000_d_0000001.txt
│   │   ├── ...
│   ├── val/
│   │   ├── ...
│   ├── test/
│       ├── ...
```
VisDrone annotations typically provide bounding box coordinates as (xmin, ymin, xmax, ymax) and class names as strings. We need to:

    1.  **Load the Annotations:** Parse each annotation file and extract bounding box coordinates and class labels.
    2.  **Convert to Yolov5 Format:** Transform the coordinates to normalized values (x_center, y_center, width, height), relative to image dimensions. Convert class names to class ids.
    3.  **Store the processed info in appropriate data structure:** Create a data structure that we can easily pass to dataset initialization.

Here is the function to achieve this:

```python
import os
import numpy as np
from PIL import Image

def visdrone_data_preparation(image_path, annotation_path, class_mapping):
    """
    Prepares VisDrone data for use with YOLOv5.
    Args:
        image_path (str): The path to the folder containing the images.
        annotation_path (str): The path to the folder containing the annotations.
        class_mapping (dict): A mapping from class names to class ids.

    Returns:
        list: A list of tuples. Each tuple contains the path to an image
            and its label information
    """

    image_names = os.listdir(image_path)
    data_list = []

    for image_name in image_names:
        image_name_no_ext = os.path.splitext(image_name)[0]
        annotation_name = image_name_no_ext + '.txt'

        img_path = os.path.join(image_path, image_name)
        label_path = os.path.join(annotation_path, annotation_name)

        if not os.path.exists(label_path):
            continue

        img = Image.open(img_path)
        width, height = img.size

        labels = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 6:
                   continue  #Skip line if the annotations has incorrect format.
                xmin, ymin, xmax, ymax, class_name = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), parts[5]
                
                class_id = class_mapping.get(class_name, -1)  # Map class name to an id.
                if class_id == -1:
                  continue #Skip this annotation if the class is not in map
                
                x_center = (xmin + xmax) / (2 * width)
                y_center = (ymin + ymax) / (2 * height)
                bbox_width = (xmax - xmin) / width
                bbox_height = (ymax - ymin) / height

                labels.append([class_id, x_center, y_center, bbox_width, bbox_height])

        if labels: # Only add if image has annotations
          data_list.append((img_path, np.array(labels)))

    return data_list


# Example class_mapping
class_mapping_visdrone = {
    'pedestrian': 0,
    'people': 1,
    'bicycle': 2,
    'car': 3,
    'van': 4,
    'truck': 5,
    'tricycle': 6,
    'awning-tricycle': 7,
    'bus': 8,
    'motor': 9,
}

# Example usage of the preparation function
# train_image_path =  'visdrone/images/train/'
# train_annotation_path = 'visdrone/annotations/train/'
# train_data = visdrone_data_preparation(train_image_path, train_annotation_path, class_mapping_visdrone)
```

**2. Custom Dataset Class Implementation:**

Now that we have a function to prepare the data, we can implement our custom dataset. This class will inherit from `torch.utils.data.Dataset` and needs to implement `__len__` and `__getitem__` methods:

```python
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import numpy as np
from PIL import Image

class VisDroneDataset(Dataset):
    def __init__(self, data_list, transform=None):
      """
      Custom dataset for loading VisDrone data.
      Args:
        data_list:  list: A list of tuples. Each tuple contains the path to an image
            and its label information, as produced by visdrone_data_preparation.
        transform (callable, optional): Optional transform to be applied
            on a sample.

      """
      self.data_list = data_list
      self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img_path, labels = self.data_list[idx]
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)
        
        labels = torch.tensor(labels, dtype=torch.float32)
        return img, labels
# Example usage of the custom dataset
# dataset = VisDroneDataset(train_data, transform=T.ToTensor())
```
**3. Data Augmentation**

Data augmentation is a critical aspect of training robust object detection models. Using `torchvision.transforms` we can create a set of image augmentations. It is important to note that we are not performing any augmentations on the bounding box data. For the purpose of demonstration, we are using a simple set of augmentations:

```python
# Data augmentation example:
transform_visdrone = T.Compose([
    T.Resize((640,640)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

#Example usage with augmentations:
# dataset = VisDroneDataset(train_data, transform=transform_visdrone)

```

**Putting it together**

Now that we have the `visdrone_data_preparation` function to transform VisDrone annotations, `VisDroneDataset` custom class, and `transform_visdrone` augmentations, you can directly load your data in the yolov5 training script. This dataset will iterate through all of the images and their corresponding processed labels. In a typical yolov5 training loop you need to replace the default dataset loader with this custom dataset. In your training script it would look something like this:

```python
#Prepare the data
train_image_path =  'visdrone/images/train/'
train_annotation_path = 'visdrone/annotations/train/'
train_data = visdrone_data_preparation(train_image_path, train_annotation_path, class_mapping_visdrone)

#Transform the data
transform_visdrone = T.Compose([
    T.Resize((640,640)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

#Prepare the dataset
dataset = VisDroneDataset(train_data, transform=transform_visdrone)

#Then create your dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)
```
**Further Considerations:**

*   **Error Handling:** Add more comprehensive error handling to manage cases like missing annotation files or malformed annotations.
*   **Data Caching:** For larger datasets, consider caching pre-processed labels to avoid repetitive parsing, especially during early experimentation.
*   **Class Mapping:** Ensure the `class_mapping` dictionary is comprehensive and correct based on your specific needs, or make the class id configurable.
*   **Data Augmentation:** As shown, basic augmentations are integrated into the dataset. However, more advanced augmentations, especially those tailored to aerial imagery (like perspective transforms) can significantly enhance the training process.
*   **Label Smoothing:** For very noisy labels, consider using label smoothing techniques, as it can lead to more robust models.
*   **Memory Management:** For large datasets, the entire transformed dataset may not fit into the RAM, consider utilizing a proper data stream.

**Recommended Reading:**

*   **"Deep Learning with PyTorch" by Eli Stevens, Luca Antiga, and Thomas Viehmann:** This book offers a thorough understanding of PyTorch’s data loading mechanisms.
*   **"Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow" by Aurélien Géron:** While covering a broader range, it contains chapters that help in understanding the data pre-processing and augmentation aspects of deep learning.
*   **Papers related to the VisDrone dataset:** Exploring research publications that use VisDrone can give ideas about specific data handling techniques, such as in “The VisDrone-DET2021 challenge: Evaluation of object detection on drone-captured aerial images”.

Adapting a dataset to a particular detection model is a critical stage and can be quite nuanced. By understanding data structures, mastering class inheritance from `torch.utils.data.Dataset`, and applying necessary transformations, you can build a powerful data pipeline to train your object detector. This should provide a strong foundation for your task. Let me know if there are other areas you'd like to explore.
