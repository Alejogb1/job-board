---
title: "How can I selectively detect specific labels with YOLOv5's detect.py?"
date: "2024-12-23"
id: "how-can-i-selectively-detect-specific-labels-with-yolov5s-detectpy"
---

Okay, let’s tackle this. I’ve spent a considerable amount of time working with YOLO models, and selectively detecting labels is a common need, especially when you're dealing with diverse datasets. It's not immediately obvious in the standard `detect.py` script, so let me walk you through how to achieve this effectively.

The core challenge arises because `detect.py` by default processes all classes trained in your model. It doesn't inherently offer a flag to filter output based on class labels. We need to leverage the underlying mechanics of the prediction and detection process to implement this filtering. Fundamentally, we're aiming to modify how detections are handled *after* the model makes its predictions. This is crucial to understand; we're not retraining the model or altering the prediction phase itself, but just filtering the output.

My initial foray into this was on a project a few years back involving aerial imagery. The model, trained to identify various types of vehicles, often produced a large number of detections, and I needed to focus solely on cars for a specific analysis. That's where I started exploring these selective filtering methods, and I've refined them over time.

The core idea revolves around modifying the `non_max_suppression` function's output. When YOLOv5 generates predictions, it outputs bounding boxes and their corresponding class predictions. The `non_max_suppression` or `NMS` step filters out redundant, overlapping bounding boxes, keeping the most confident detection. This results in a tensor containing the bounding box coordinates, confidence scores, and class indices. It's at this point that we can apply our selective filtering.

I've found three common and useful techniques:

**1. Filtering During Post-Processing:**

   This approach involves iterating through the `non_max_suppression` output and keeping only the detections for our desired labels. The key advantage here is that it's flexible and allows for multiple label filtering without modifying core YOLOv5 functions. Let me present an example code snippet for demonstration:

```python
import torch

def filter_detections_by_label(detections, desired_labels):
    filtered_detections = []
    for *xyxy, conf, cls in detections.tolist():
      if int(cls) in desired_labels:
        filtered_detections.append([xyxy, conf, int(cls)])
    if filtered_detections:
       return torch.tensor([torch.tensor(d[0]).to(detections.device).float() + torch.tensor([0,0,0,0]).float().to(detections.device), torch.tensor([d[1]]).to(detections.device).float(), torch.tensor([d[2]]).to(detections.device).int()], device=detections.device)
    else:
       return torch.empty((0, 6), device=detections.device)
  

# Assume 'pred' is the output of your YOLOv5 model's forward pass
def run_detection(model, img, desired_labels=[0, 1]):
    pred = model(img)[0]
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic_nms=False, max_det=1000)
    filtered_results = filter_detections_by_label(pred[0], desired_labels)

    return filtered_results

# Example usage
# Load your model and image (omitted for brevity)
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
# img = torch.rand(1, 3, 640, 640)
# # Set your desired labels (e.g., class indices 0 and 1)
# desired_labels = [0, 2]
# output = run_detection(model, img, desired_labels)
#
# # 'output' now only contains detections from the specified classes.
```

   In this example, `filter_detections_by_label` is where the filtering occurs. It iterates through detections, only keeping those whose class index is in `desired_labels`. The logic of creating the tensor is necessary due to the way the detections are structured after NMS (a single tensor of detections). The function takes in `detections` from the NMS process and `desired_labels`. It creates a new tensor from the filtered detections and outputs it.
   This approach avoids any changes to the core YOLOv5 library code.
    
**2. Modifying the `non_max_suppression` Function (More Involved):**

   Another method involves modifying the non-max suppression function directly. This is more intrusive but can be more efficient if you are doing frequent label-specific detection. You'd need to inject code *before* the NMS step.

```python
import torch
from utils.general import non_max_suppression

def selective_non_max_suppression(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic_nms=False,
        max_det=1000,
        desired_classes=None
    ):

    if desired_classes is not None:
        # Filtering logic before NMS
        filtered_prediction = []
        for i in range(prediction.shape[0]):
            if (prediction[i, 5] in desired_classes) :
              filtered_prediction.append(prediction[i,:].unsqueeze(0))
        if not filtered_prediction:
          return  torch.empty((0, 6), device=prediction.device)
        prediction = torch.cat(filtered_prediction, dim=0)
    
    
    # Call the original NMS function with the filtered prediction
    return non_max_suppression(
        prediction,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        classes=classes,
        agnostic_nms=agnostic_nms,
        max_det=max_det,
    )
    

# Modify the main function to use the custom version
def run_detection_nms(model, img, desired_labels=[0,1]):
    pred = model(img)[0]
    pred = selective_non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic_nms=False, max_det=1000, desired_classes=desired_labels)
    return pred

# Example usage (similar to above, but using `selective_non_max_suppression`)
# Load your model and image
# output = run_detection_nms(model, img, desired_labels)
```
    
    This modified NMS function includes an `if desired_classes` clause that checks the classes before applying the nms filtering. If `desired_classes` is set, then it filters the predictions using the included labels.
    
    **3. Pre-Filtering of Predictions (Least Efficient)**

    The least efficient of the methods, but included for completeness, would be filtering the raw predictions directly. In general, it is best to filter after NMS is applied. This pre-filtering would apply before any NMS filtering. It’s generally not advised due to performance implications, but here's how you'd approach it:

```python
import torch

def filter_raw_predictions(predictions, desired_labels):
    filtered_predictions = []
    for prediction in predictions:
        if int(prediction[5]) in desired_labels:
            filtered_predictions.append(prediction.unsqueeze(0))
    if not filtered_predictions:
      return  torch.empty((0, 6), device=predictions.device)
    return torch.cat(filtered_predictions, dim=0)

def run_detection_prefilter(model, img, desired_labels=[0,1]):
    pred = model(img)[0]
    filtered_pred = filter_raw_predictions(pred, desired_labels)
    pred = non_max_suppression(filtered_pred, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic_nms=False, max_det=1000)

    return pred


# example usage
# Load your model and image
# output = run_detection_prefilter(model, img, desired_labels)
```
    This method creates a function `filter_raw_predictions` that filters the raw predictions based on the desired labels prior to NMS. While this might seem intuitive, it isn't as efficient as filtering the results after NMS. This approach adds an additional step before the usual pipeline and should be avoided in favor of the first two examples.

**Resource Recommendations:**

For a more in-depth understanding of object detection, I highly recommend reading "Deep Learning for Vision Systems" by Mohamed Elgendy. It provides a comprehensive overview of various object detection algorithms, including YOLO, and delves into the intricacies of NMS. For a more hands-on, code-oriented perspective, explore the official PyTorch documentation. Additionally, the original YOLO papers by Joseph Redmon are invaluable resources; these give critical insights into the architecture and functionality of the model. Finally, the Ultralytics documentation, although not a single reference point, is quite informative, especially the sections related to the `detect.py` script and its components.

In closing, selective label detection with YOLOv5 isn't a native feature, but using the methods I've detailed (post-filtering, NMS modification, or pre-filtering), you can achieve it effectively. Based on my experience, the first method is most versatile for quick changes or debugging, while the second can be optimal for production settings, although it requires modifying some library code. As always, make sure to evaluate performance and choose the method best suited to your specific needs. And always, make sure your results are interpretable and meaningful within the context of your project. Good luck!
