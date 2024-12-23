---
title: "How can I convert a JPEG mask to COCO JSON format?"
date: "2024-12-23"
id: "how-can-i-convert-a-jpeg-mask-to-coco-json-format"
---

Alright,  I've definitely been down this road before, several times actually, usually in projects involving image segmentation or object detection annotation pipelines. Converting a jpeg mask to coco json isn't inherently complex, but it requires a good understanding of both the jpeg mask format (often a single-channel grayscale image representing the mask) and the coco json structure, especially the segmentation data format. We need to move from pixels to polygons or RLE encoded masks. I'll walk through the process, focusing on the technical details and giving you a few code examples to illustrate.

First, let’s establish the core challenge. Jpeg masks, typically, are pixel-based representations. Each pixel has a grayscale value (often 0 for background, and a value, possibly 255 or 1, for the object). Coco json, on the other hand, uses either polygons or run-length encoding (RLE) to represent the segmentation. Polygons are sequences of x, y coordinates that define the outline of the object, while RLE provides a compressed representation of the mask using run lengths. The conversion involves figuring out how to extract either the polygon or RLE representation from the pixel data.

My initial go-to has often been polygon extraction. This involves boundary tracing or contour detection. We can consider the mask image as a contour map where 0 is the "ground level" and the value representing the object represents a 'hill'. We can trace the outer edges of that hill.

Here’s an example using python with `opencv` and `numpy`:

```python
import cv2
import numpy as np
import json

def mask_to_coco_polygon(mask_path, image_id, category_id):
    """Converts a jpeg mask to coco polygon format.

        Args:
            mask_path (str): Path to the jpeg mask image.
            image_id (int): The image id from coco.
            category_id (int): The coco category id for the object in the mask.

        Returns:
            dict: A dictionary representing the coco annotation.
            None: If no contours are found.
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
      return None # or raise an exception as needed

    _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY) # Adjust threshold if needed

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    segmentations = []

    for contour in contours:
        contour = contour.reshape(-1, 2)
        segmentation = contour.flatten().tolist()
        segmentations.append(segmentation)

    if not segmentations:
       return None

    annotation = {
        "image_id": image_id,
        "category_id": category_id,
        "segmentation": segmentations,
        "iscrowd": 0,  # set iscrowd to 0 unless you are dealing with crowd annotations.
        "area": cv2.contourArea(contours[0]) # Assuming only single contour is present and area is important
    }
    
    # You can also include a bounding box which I'm skipping here
    # For bounding box calculation check the opencv documentation
    
    return annotation

#example
if __name__ == '__main__':
    # dummy mask image creation for testing
    dummy_mask = np.zeros((100,100),dtype = np.uint8)
    cv2.circle(dummy_mask,(50,50),40,255,-1)
    cv2.imwrite("dummy_mask.jpg",dummy_mask)
    
    annotation_data = mask_to_coco_polygon("dummy_mask.jpg", image_id=1, category_id=1)

    if annotation_data:
         print(json.dumps(annotation_data, indent=2))

```

In the snippet above, we use `cv2.imread` to load the mask, apply thresholding to binarize it, and then use `cv2.findContours` to find the contours, the boundary. The resulting contour coordinates are flattened into a list representing the polygon, and included into the annotation dictionary.

Now, polygon annotations aren't always ideal, especially for intricate objects with complex boundaries. RLE can be more efficient in such cases. Let’s see how to convert to RLE:

```python
import cv2
import numpy as np
import json
from pycocotools import mask as maskUtils

def mask_to_coco_rle(mask_path, image_id, category_id):
  """Converts a jpeg mask to coco RLE format.

      Args:
          mask_path (str): Path to the jpeg mask image.
          image_id (int): The image id from coco.
          category_id (int): The coco category id for the object in the mask.

      Returns:
          dict: A dictionary representing the coco annotation.
          None: If the mask could not be processed.
    """
  mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
  if mask is None:
     return None

  _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY) # Adjust threshold if needed
  
  fortran_mask = np.asfortranarray(thresh)
  rle = maskUtils.encode(fortran_mask)

  if not rle:
      return None
  
  annotation = {
      "image_id": image_id,
      "category_id": category_id,
      "segmentation": rle, # changed to a simple dict from previous example
      "iscrowd": 0, #set to 0 if not a crowed annotation
      "area": int(maskUtils.area(rle)), # calculate the mask area directly from rle
    }
  
  # You can also include a bounding box which I'm skipping here

  return annotation


# Example of usage:
if __name__ == '__main__':
  # dummy mask image creation for testing
  dummy_mask = np.zeros((100,100),dtype = np.uint8)
  cv2.circle(dummy_mask,(50,50),40,255,-1)
  cv2.imwrite("dummy_mask.jpg",dummy_mask)
    
  annotation_data = mask_to_coco_rle("dummy_mask.jpg", image_id=1, category_id=1)
  if annotation_data:
       print(json.dumps(annotation_data, indent=2))
```

Here, the core difference is the use of the `pycocotools` library. We create a fortran array which is the expected input of the `encode` function and then directly get the rle mask object from the function and put it as the segmentation in coco format.

A crucial detail when working with RLE is the ‘fortran array’ requirement. This refers to the order in which the pixel data is arranged in memory, and is a memory layout of arrays where elements are stored column-major wise. By converting the binary mask into a fortran array, we prepare the data correctly for the RLE encoding algorithm within `pycocotools`.

Finally, let’s consider a case where you might have multiple objects in a single mask. This means the grayscale might not just be 0 and 255, but also different grayscale values to denote different objects. In such cases, we need to loop through each object in the mask image and then extract the annotation for each object. Let’s stick with RLE for the sake of brevity:

```python
import cv2
import numpy as np
import json
from pycocotools import mask as maskUtils

def multi_mask_to_coco_rle(mask_path, image_id, category_mapping):
    """Converts a multi-object jpeg mask to coco RLE format.

        Args:
            mask_path (str): Path to the jpeg mask image.
            image_id (int): The image id from coco.
            category_mapping (dict): Mapping between the grayscale value in the mask and coco category id

        Returns:
            list: A list of dictionaries, each representing a coco annotation.
            None: If the mask could not be processed.
    """

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None

    annotations = []
    unique_values = np.unique(mask) #find the different grayscale values
    unique_values = unique_values[unique_values != 0] #ignore background (grayscale value 0)

    for val in unique_values:
        binary_mask = np.uint8(mask == val)
        category_id = category_mapping.get(val)

        if category_id is None:
            continue # if category_id not present in mapping

        fortran_mask = np.asfortranarray(binary_mask)
        rle = maskUtils.encode(fortran_mask)

        if not rle:
          continue # no mask found for the value.

        annotation = {
            "image_id": image_id,
            "category_id": category_id,
            "segmentation": rle,
            "iscrowd": 0,
            "area": int(maskUtils.area(rle)),
        }
        annotations.append(annotation)

    return annotations

# example usage
if __name__ == '__main__':
  # dummy mask image creation for testing
  dummy_mask = np.zeros((100,100),dtype = np.uint8)
  cv2.circle(dummy_mask,(30,30),20,100,-1)
  cv2.circle(dummy_mask,(70,70),20,200,-1)
  cv2.imwrite("dummy_mask.jpg",dummy_mask)

  category_mapping = {100: 1, 200: 2}  # mapping from mask value to coco category id.

  annotation_data = multi_mask_to_coco_rle("dummy_mask.jpg", image_id=1, category_mapping=category_mapping)

  if annotation_data:
        print(json.dumps(annotation_data, indent=2))
```

In this last example, I iterate through the unique grayscale values found in the mask, creating separate masks for each object by applying thresholding and then using the category_mapping dictionary to find the correct category id and then extracting the rle and forming the annotation.

When diving deeper into coco specifications, I strongly suggest the 'coco dataset' paper by Tsung-Yi Lin et al. (accessible through most research databases or on the coco website) for a detailed understanding of the data format. Additionally, the official pycocotools repository is the source of truth for everything regarding coco format manipulation and is very useful in understanding the finer details about rle masks.

These examples should provide a solid base for converting your jpeg masks to coco json format. The choice of polygons vs. RLE will depend on the specific use case and the nature of the masks you have, but regardless of your choice, remember that thorough testing of your conversion process is essential to ensure the integrity of your annotations.
