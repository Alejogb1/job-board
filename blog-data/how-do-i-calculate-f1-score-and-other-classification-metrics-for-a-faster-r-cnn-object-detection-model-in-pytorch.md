---
title: "How do I calculate F1-score and other classification metrics for a Faster R-CNN object detection model in PyTorch?"
date: "2024-12-23"
id: "how-do-i-calculate-f1-score-and-other-classification-metrics-for-a-faster-r-cnn-object-detection-model-in-pytorch"
---

Okay, let's delve into calculating F1-score and other classification metrics for a Faster R-CNN object detection model. This is a recurring challenge, and I recall a particularly frustrating project back in '18 where our model was nailing the bounding boxes but the overall classification performance, as assessed by our client, seemed... off. It turned out we were focusing too much on the box regression and neglecting proper classification analysis. We weren't looking beyond simple accuracy, which as you likely know, is a very misleading metric in imbalanced datasets.

The core issue here isn't *just* about applying formulas, but fundamentally about understanding how object detection outputs must be translated into the context of classification metrics. We need to bridge the gap between bounding box predictions and class predictions.

Here’s the breakdown, focusing on the common case where you have class probabilities for each predicted bounding box.

Firstly, remember that object detection is fundamentally a *localization* and *classification* task, with classification happening for each predicted bounding box. We typically use the classification scores associated with each bounding box, after non-maximum suppression (NMS), to evaluate classification performance. Just obtaining class probabilities after NMS isn't enough; we need to compare those probabilities with the actual ground truth classifications, but *aligned* properly with ground truth boxes. This alignment is the crucial step. It’s about establishing which predictions are *true positives*, *false positives*, and *false negatives*.

The first thing to clarify is that there isn’t a single, universally agreed upon, way of calculating classification performance in the context of object detection. We typically assess the classification scores for *only those bounding boxes which overlap significantly* with a ground-truth box. We use a threshold (usually 0.5, but tunable) for the Intersection over Union (IoU) between the prediction and the ground-truth box to qualify as a "match." If an IoU match exists, and the prediction is correctly classified, this becomes a true positive. If there is a match, but the classification is incorrect, it’s a false positive *with respect to the classification, but not the localization*. If no match is found, and a ground-truth object exists, it's a false negative.

Now, let’s solidify this with some code examples and I'll use PyTorch tensor operations for efficiency. Suppose we have a function which does the bounding box matching as described above, using the Intersection over Union to find matches, given by `match_predictions_to_groundtruth` (we will not write this function for simplicity, but assume it handles the assignment). Assume this returns a list of tuples, each tuple containing the following structure: (prediction index, ground truth index, boolean indicating matched with IoU > threshold, classification prediction for match, classification ground truth for match)

```python
import torch
from sklearn.metrics import f1_score, precision_score, recall_score

def calculate_metrics(predictions, ground_truths, iou_threshold=0.5, num_classes=None):
    """Calculates classification metrics for object detection results.

    Args:
        predictions: List of tensors, where each tensor contains bounding box
                     predictions and their corresponding class probabilities.
        ground_truths: List of tensors, where each tensor contains ground truth
                       bounding boxes and their corresponding class labels.
        iou_threshold: IoU threshold for matching predictions with ground truths.
        num_classes: The number of classes. If None will be derived from the prediction classes.

    Returns:
         Dictionary containing F1 score, precision, recall for each class
    """

    matched_pairs = match_predictions_to_groundtruth(predictions, ground_truths, iou_threshold)

    true_labels = []
    pred_labels = []

    for _, _, matched, pred_class, gt_class in matched_pairs:
        if matched:
            true_labels.append(gt_class)
            pred_labels.append(pred_class)

    if not true_labels:
       return {"f1_score": {}, "precision": {}, "recall": {}}

    if not num_classes:
        num_classes = max(max(true_labels), max(pred_labels)) + 1
    
    f1 = f1_score(true_labels, pred_labels, average=None, labels=range(num_classes))
    precision = precision_score(true_labels, pred_labels, average=None, labels=range(num_classes))
    recall = recall_score(true_labels, pred_labels, average=None, labels=range(num_classes))

    return {"f1_score": {c:f for c, f in zip(range(num_classes), f1)},
            "precision": {c:p for c, p in zip(range(num_classes), precision)},
            "recall": {c:r for c, r in zip(range(num_classes), recall)}
            }
```

This first code snippet shows how, *after* we have a list of matched predictions and ground truths, we can compute the F1-score, recall, and precision using the scikit-learn methods. Note the use of `average=None`, which forces these to return per-class metrics, rather than overall averages, which are often less insightful. If `num_classes` is not specified, it will be inferred.

Now for a practical example, lets use a slightly more complex function which includes a dummy placeholder for how we might load this, and assume `match_predictions_to_groundtruth` is written elsewhere.

```python
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def eval_fasterrcnn(model, dataloader, num_classes, device):

  all_metrics = []

  for images, targets in dataloader:

    images = list(image.to(device) for image in images)
    with torch.no_grad():
        outputs = model(images)
    
    predictions = []
    ground_truths = []

    for i,output in enumerate(outputs):
      boxes = output['boxes'].cpu().detach()
      scores = output['scores'].cpu().detach()
      labels = output['labels'].cpu().detach()
      
      # filter by score
      filtered_indices = scores > 0.5
      filtered_boxes = boxes[filtered_indices]
      filtered_labels = labels[filtered_indices]
      
      predictions.append((filtered_boxes, filtered_labels))
      
      ground_truth_boxes = targets[i]['boxes'].cpu().detach()
      ground_truth_labels = targets[i]['labels'].cpu().detach()
      
      ground_truths.append((ground_truth_boxes, ground_truth_labels))

    metrics = calculate_metrics(predictions, ground_truths, num_classes=num_classes)
    all_metrics.append(metrics)

  
  final_metrics = {}
  for k in all_metrics[0]:
    class_metric = {}
    for class_num in all_metrics[0][k]:
        class_metric[class_num] = sum([d[k][class_num] for d in all_metrics]) / len(all_metrics)
    final_metrics[k] = class_metric

  return final_metrics
```

This code snippet provides a more concrete example of how you might evaluate your model. The `eval_fasterrcnn` function iterates through your dataloader, performs inference, applies a simple score threshold of 0.5, then calls `calculate_metrics`. This would be part of your training loop. The important thing here, note how we only are feeding the scores and labels to `calculate_metrics` after we have performed inference and applied a score threshold. This is critical because we are not interested in classifying bounding boxes which do not have a high confidence of containing an object of a class. We finally average the metrics over the dataset to obtain an aggregated measure of performance.

Finally, let’s add some code which can allow you to experiment on a dummy dataset

```python
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class DummyDataset(Dataset):
    def __init__(self, num_samples=10, num_classes=3):
        self.num_samples = num_samples
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = torch.rand(3, 256, 256)
        num_boxes = torch.randint(1, 5, (1,)).item()
        boxes = torch.rand(num_boxes, 4) * 256
        labels = torch.randint(0, self.num_classes, (num_boxes,))
        targets = {"boxes": boxes, "labels": labels}
        return image, targets


if __name__ == '__main__':

    num_classes = 3
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # load model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, progress=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes+1) # for background
    model = model.to(device)
    model.eval()

    dataset = DummyDataset(num_samples=10, num_classes=num_classes)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    metrics = eval_fasterrcnn(model, dataloader, num_classes, device)
    print(metrics)
```

This final snippet provides some code that actually loads a model, creates a dummy dataset, and prints the classification metrics. This provides a very fast way to prototype and debug your workflow, and then replace the dummy dataset with your real data.

Crucially, note these steps:

1.  **Match Predictions:** Correctly associate predicted bounding boxes to ground-truth bounding boxes, usually using IoU and a chosen threshold.
2.  **Collect Labels:** Once matched, gather the predicted class and ground truth class labels and discard those which are not matched.
3.  **Compute Metrics:** Use metrics from libraries like scikit-learn, particularly the `f1_score`, `precision_score`, and `recall_score` functions, ensuring `average=None` for per-class metrics.
4.  **Iterate over your dataset:** Perform the above steps for each batch, storing all metrics. Average these metrics to obtain your final dataset metrics.

For deeper understanding of the underlying math, I'd strongly suggest delving into the classic Pattern Recognition and Machine Learning textbook by Christopher Bishop. Also, the work on object detection evaluation from the COCO dataset challenge (often discussed in their yearly workshop proceedings) is extremely relevant. For a more focused look at bounding box matching and IoU, look for papers describing the evaluation protocols in popular object detection benchmarks like PASCAL VOC and MS COCO. In particular, pay careful attention to the definition of *True Positive*, *False Positive*, and *False Negative* in the context of object detection, as this will clear up any confusion about what to include in the metrics. Remember, mastering this analysis is a critical step in building robust object detection systems. Good luck!
