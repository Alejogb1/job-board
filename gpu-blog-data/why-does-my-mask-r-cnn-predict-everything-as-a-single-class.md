---
title: "Why does my Mask R-CNN predict everything as a single class?"
date: "2025-01-26"
id: "why-does-my-mask-r-cnn-predict-everything-as-a-single-class"
---

A common pitfall in Mask R-CNN training, especially for custom datasets, is misconfiguration of the class mapping, leading the model to predict all objects as belonging to a single, often the background, class. I've encountered this exact scenario multiple times, most notably when building a defect detection system for a manufacturing line. The problem stems not necessarily from the model's core architecture, but rather how the training data's class labels are processed and interpreted.

The root cause generally lies within the disconnect between the annotation format provided, the expected input of the Mask R-CNN model, and ultimately, the data loader implementation. Specifically, a crucial aspect is the correct mapping between class indices used during annotation and the model's output classes. The model relies on numerical indices to distinguish between classes, with '0' typically reserved for background. If this mapping is absent, incorrect, or simply inconsistent, the model will effectively learn to predict only one class, typically the dominant one or the background if improperly handled. Consider a scenario where a dataset uses class names like 'defect_type_a,' 'defect_type_b,' and so on, while the training code expects purely numerical indices starting at zero. If the class mapping mechanism doesn't translate these names into appropriate numeric indices, the model will fail. The model’s output has a size of `num_classes + 1`, as the first class is usually background.

I will explain this through common issues related to data loading and mapping and provide examples based on experiences with various custom datasets. The most frequent cause is a failure to properly encode class labels during the data preparation phase. The annotations for a bounding box and segmentation mask often come as text labels, requiring them to be converted into numerical representations that the Mask R-CNN can understand. The most commonly used approach is to assign indices like 0, 1, 2, etc., to the different object categories, with the 0 often (but not always) representing the background. Inefficiently handled, text labels can be propagated, causing all masks to essentially be grouped under the same class – the background class if the model doesn't recognize the labels, or a random, unified class if they get incorrectly assigned an index other than the background.

Secondly, the data loading process must maintain the proper mapping through all the transformations involved, including any image augmentations and mask manipulations. If the labels are lost or re-assigned incorrectly during these transformations, the model may be exposed to erroneous data. This means the transforms that crop and resize, must also manage the associated masks and class IDs.

Thirdly, if the input dataset lacks diversity in its class distribution, or if the background class is heavily overrepresented, the model can skew its predictions towards a single class. Although this skew is often attributed to imbalanced data, it's sometimes compounded by incorrect class encoding, effectively collapsing all classes into the most prevalent one, often the background, which is already dominant by nature of most images containing far more background pixels.

Here are some code snippets demonstrating these principles with explanations:

**Example 1: Inconsistent Class Mapping**

Here, I show a snippet using a conceptual approach to loading data. Assume you have a dataset where the annotations are in JSON format, containing class names as strings. This example demonstrates how neglecting the conversion to numerical labels will cause the problem.

```python
import json
import numpy as np

def load_annotations(annotation_path):
    with open(annotation_path, 'r') as f:
        data = json.load(f)
    
    image_ids = []
    boxes = []
    masks = []
    labels = []
    
    for item in data:
      image_ids.append(item['image_id'])
      box = item['bbox'] #assuming bbox is in x1,y1,x2,y2 format
      boxes.append(np.array(box, dtype=np.float32)) #boxes as list of list
      
      mask = item['mask'] #simplified to a 2d numpy array for the mask, shape H,W
      masks.append(np.array(mask, dtype=np.uint8)) #masks as list of np arrays
      
      label = item['label'] #assuming it comes as a string name
      labels.append(label) # labels is a list of string labels

    return image_ids, boxes, masks, labels

#incorrect usage in data loader
def data_loader_incorrect(annotation_file):
    image_ids, boxes, masks, labels = load_annotations(annotation_file)
    return image_ids, boxes, masks, labels #labels is still string, model will not recognize it


#usage
image_ids, boxes, masks, labels  = data_loader_incorrect("ann.json")

print(labels) #output example : ['defect_type_a', 'defect_type_b', 'defect_type_a', ...]
```

This incorrect implementation directly uses string class names which the model can't process. The model expects indices and will thus default to the background class.

**Example 2: Correct Class Mapping**

This is a correct way using a dictionary map to change string class labels to integers:

```python
import json
import numpy as np

def load_annotations(annotation_path, class_mapping):
    with open(annotation_path, 'r') as f:
        data = json.load(f)
    
    image_ids = []
    boxes = []
    masks = []
    labels = []
    
    for item in data:
      image_ids.append(item['image_id'])
      box = item['bbox'] #assuming bbox is in x1,y1,x2,y2 format
      boxes.append(np.array(box, dtype=np.float32)) #boxes as list of list
      
      mask = item['mask'] #simplified to a 2d numpy array for the mask, shape H,W
      masks.append(np.array(mask, dtype=np.uint8)) #masks as list of np arrays
      
      label_name = item['label'] #assuming it comes as a string name
      label_index = class_mapping[label_name]
      labels.append(label_index) # labels is a list of ints
    return image_ids, boxes, masks, labels

# Correct Usage
def data_loader_correct(annotation_file, class_mapping):
    image_ids, boxes, masks, labels = load_annotations(annotation_file, class_mapping)
    return image_ids, boxes, masks, labels

# usage

class_mapping = {
    'defect_type_a': 1, # 0 is usually the background
    'defect_type_b': 2,
    'defect_type_c': 3,
}
image_ids, boxes, masks, labels  = data_loader_correct("ann.json", class_mapping)
print(labels) #output example : [1, 2, 1, ...]

```
Here, the `class_mapping` dictionary provides the necessary translation, turning `defect_type_a` into '1' and so on. This is crucial for the model to distinguish between different classes. The labels are now correctly in numeric format and assigned the correct integer.

**Example 3: Incorrect Mapping Within Data Transformation**

Here, I illustrate a scenario where the mapping is established initially but gets lost during transformations:

```python
import json
import numpy as np
from copy import deepcopy

def load_annotations(annotation_path, class_mapping):
    with open(annotation_path, 'r') as f:
        data = json.load(f)
    
    image_ids = []
    boxes = []
    masks = []
    labels = []
    
    for item in data:
      image_ids.append(item['image_id'])
      box = item['bbox'] #assuming bbox is in x1,y1,x2,y2 format
      boxes.append(np.array(box, dtype=np.float32)) #boxes as list of list
      
      mask = item['mask'] #simplified to a 2d numpy array for the mask, shape H,W
      masks.append(np.array(mask, dtype=np.uint8)) #masks as list of np arrays
      
      label_name = item['label'] #assuming it comes as a string name
      label_index = class_mapping[label_name]
      labels.append(label_index) # labels is a list of ints
    return image_ids, boxes, masks, labels

#Incorrect transform
def random_crop(image, boxes, masks, labels, crop_size): # crop the image, but also make sure the bounding boxes are within image
    h,w = image.shape[0:2]
    x_start = np.random.randint(0, w - crop_size)
    y_start = np.random.randint(0, h - crop_size)
    
    cropped_image = image[y_start:y_start + crop_size, x_start:x_start+crop_size]
    cropped_boxes = []
    cropped_masks = []
    cropped_labels = deepcopy(labels)
    
    for index, box in enumerate(boxes):
       #assuming box format is x1, y1, x2, y2
        x1, y1, x2, y2 = box
        
        
        x1_crop = np.maximum(0, x1 - x_start)
        y1_crop = np.maximum(0, y1 - y_start)
        x2_crop = np.minimum(crop_size, x2 - x_start)
        y2_crop = np.minimum(crop_size, y2 - y_start)
        
        if x2_crop > x1_crop and y2_crop > y1_crop:
            cropped_boxes.append(np.array([x1_crop,y1_crop,x2_crop,y2_crop]))
            cropped_masks.append(masks[index][y_start:y_start + crop_size, x_start:x_start+crop_size])
        else:
            cropped_labels.pop(index) # remove box with no overlap with crop
    return cropped_image, cropped_boxes, cropped_masks, cropped_labels

# Incorrect Usage in Data Loading pipeline:
def data_loader_incorrect_transformation(annotation_file, class_mapping):
    image_ids, boxes, masks, labels = load_annotations(annotation_file, class_mapping)
    image = load_image(image_ids[0]) #load the image here
    transformed_image, transformed_boxes, transformed_masks, transformed_labels = random_crop(image, boxes, masks, labels, crop_size = 512)

    return transformed_image, transformed_boxes, transformed_masks, transformed_labels #labels are no longer aligned correctly

def load_image(img_id):
    #simulate loading image
    return np.random.rand(1024,1024,3)
# usage
class_mapping = {
    'defect_type_a': 1,
    'defect_type_b': 2,
    'defect_type_c': 3,
}
transformed_image, transformed_boxes, transformed_masks, transformed_labels = data_loader_incorrect_transformation("ann.json", class_mapping)
print(transformed_labels) #output example : [1, 2, 1] - it may not have the same mapping with mask and bbox
```

In this flawed `random_crop` function, boxes are appropriately cropped and the labels are copied, but the deletion of boxes is not in sync with the labels. This results in misaligned labels with its associated masks after the crop operation. The labels would be incorrect for any box/mask that were excluded during the transformation stage. It highlights that careful design is needed to maintain class ID integrity during the transformations.

To diagnose these problems, examine the data loader output thoroughly, checking that class mappings are correctly applied and remain synchronized, even after image and mask transformations. Visualizing the masks alongside their corresponding class labels will help spot discrepancies. If the problem persists after addressing these points, ensure the training dataset has balanced class representation and not an overwhelmingly large background class.

For further study, I would recommend exploring resources focusing on image data preprocessing, object detection dataset handling, and Mask R-CNN model configuration. Material relating to data augmentation for object detection is also very useful, in particular paying special attention to operations that involve both the image and the mask, and corresponding labels.
