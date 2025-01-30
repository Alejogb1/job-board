---
title: "What multiclass metrics are suitable for image segmentation using fastai?"
date: "2025-01-30"
id: "what-multiclass-metrics-are-suitable-for-image-segmentation"
---
The inherent ambiguity in evaluating multi-class image segmentation performance necessitates a nuanced approach beyond simple accuracy.  My experience working on medical image analysis projects, specifically analyzing microscopic tissue samples for cancer detection, highlighted the crucial role of selecting appropriate metrics that reflect the specific challenges of the task.  Simply put, while overall accuracy might seem straightforward, it can be misleading when class imbalances exist, as is often the case in medical imaging.  Therefore, selecting appropriate multi-class metrics for image segmentation with fastai requires careful consideration of the problem's characteristics.


**1.  Clear Explanation of Suitable Metrics**

Image segmentation, particularly in a multi-class context, involves assigning each pixel in an image to one of several predefined classes.  The evaluation of such models necessitates metrics that go beyond simple classification accuracy, as they must account for the spatial distribution of predictions.  For this, we typically employ metrics that consider both the precision and recall of each class, and then aggregate these class-wise scores to obtain an overall performance indicator.  These metrics include:

* **Intersection over Union (IoU) or Jaccard Index:** This metric calculates the ratio of the intersection to the union of the predicted and ground truth segmentation masks for each class. A higher IoU indicates better segmentation accuracy for that class.  It's less sensitive to class imbalances than accuracy.  The mean IoU (mIoU) is then computed by averaging the IoU across all classes, providing a single score summarizing the overall segmentation performance.

* **Dice Coefficient:**  Similar to IoU, the Dice coefficient measures the overlap between the predicted and ground truth segmentation masks.  It's defined as twice the intersection divided by the sum of the sizes of the two sets.  A value of 1 indicates perfect overlap, while 0 indicates no overlap.  Like IoU, the mean Dice coefficient (mDice) is often used for summarizing performance across classes.  I’ve found it particularly useful when dealing with highly fragmented regions, which are common in certain biological image analysis tasks.

* **Precision and Recall (Class-wise and Macro-averaged):** Precision reflects the accuracy of positive predictions for a specific class, while recall measures the model's ability to identify all instances of that class.  Calculating these metrics for each class individually offers valuable insights into the model's strengths and weaknesses per class.  Macro-averaging then combines the class-wise scores, giving equal weight to each class, regardless of its frequency in the dataset. This is crucial when dealing with imbalanced datasets, as it prevents the dominant classes from disproportionately influencing the overall score.  A harmonic mean of precision and recall, the F1-score, provides a single combined metric for each class, offering a balanced assessment.


**2. Code Examples with Commentary**

The following code examples demonstrate how to compute these metrics using fastai and relevant libraries.  I've assumed you've already trained your segmentation model and have access to predictions and ground truth masks.  These examples leverage the flexibility of fastai's `Learner` object, which is tailored to various tasks, including segmentation.


**Example 1: Using fastai's built-in metrics (if available)**

```python
from fastai.metrics import Dice, IoU
learn.metrics = [Dice(), IoU()]
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix() #This can also be useful for insights into individual class performance
```

*Commentary:*  If your specific fastai version includes Dice and IoU metrics directly, this is the simplest approach.  The `ClassificationInterpretation` object provides valuable visualization tools to complement the numerical metrics.  This method is dependent on the availability of these metrics within your fastai version, hence its inclusion is conditional.


**Example 2: Manual Calculation using NumPy**

```python
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

def calculate_iou(y_true, y_pred, num_classes):
    iou = []
    for i in range(num_classes):
        intersection = np.logical_and(y_true == i, y_pred == i).sum()
        union = np.logical_or(y_true == i, y_pred == i).sum()
        iou.append(intersection / union if union > 0 else 0)  #Avoid division by zero
    return np.mean(iou)

def calculate_dice(y_true, y_pred, num_classes):
    dice = []
    for i in range(num_classes):
        intersection = np.logical_and(y_true == i, y_pred == i).sum()
        dice_coef = (2. * intersection) / (np.sum(y_true == i) + np.sum(y_pred == i))
        dice.append(dice_coef)
    return np.mean(dice)

# Assuming y_true and y_pred are your ground truth and prediction masks, flattened to 1D arrays
y_true_flattened = y_true.flatten()
y_pred_flattened = y_pred.flatten()
num_classes = len(np.unique(y_true_flattened))

iou = calculate_iou(y_true_flattened, y_pred_flattened, num_classes)
dice = calculate_dice(y_true_flattened, y_pred_flattened, num_classes)

precision = precision_score(y_true_flattened, y_pred_flattened, average='macro', zero_division=0) #macro-averaging
recall = recall_score(y_true_flattened, y_pred_flattened, average='macro', zero_division=0) #macro-averaging
f1 = f1_score(y_true_flattened, y_pred_flattened, average='macro', zero_division=0) #macro-averaging

print(f"IoU: {iou}, Dice: {dice}, Precision: {precision}, Recall: {recall}, F1: {f1}")
```

*Commentary:* This example demonstrates a manual calculation of IoU, Dice, precision, recall, and F1-score using NumPy.  It's crucial to flatten the prediction and ground truth masks to use `sklearn.metrics` functions correctly.  The `zero_division=0` argument handles potential division by zero errors.  This approach offers more granular control and allows for easy extension to other metrics.


**Example 3: Leveraging segmentation-specific libraries**

```python
import segmentation_models_pytorch as smp
from sklearn.metrics import classification_report

# Assuming 'preds' contains your model's predictions and 'targets' contains the ground truth masks
dice_score = smp.metrics.IOUScore(threshold=0.5).compute(preds, targets) #Assuming binary segmentation for simplicity
iou_score = smp.metrics.IOUScore(threshold=0.5, class_indexes=[0, 1, 2]).compute(preds, targets) #Adjust class_indexes according to your number of classes

#For precision, recall and F1-score the inputs should be flattened similarly to the above method
#This example is illustrative to showcase an alternative library
report = classification_report(targets.flatten(), preds.flatten(), target_names=['class0', 'class1', 'class2'],zero_division=0)  # Replace class names as needed
print(report)
```

*Commentary:* This utilizes the `segmentation_models_pytorch` library, which provides specialized metrics for semantic segmentation tasks. This example shows how to compute IoU using the library, although it requires adapting the input format to align with its expectations. The use of `classification_report` from `sklearn` is demonstrated, emphasizing that while libraries like `segmentation_models_pytorch` may offer convenient IoU/Dice computation, other metrics often require the use of standard machine learning libraries like `sklearn`.


**3. Resource Recommendations**

For further in-depth understanding, I recommend consulting standard machine learning textbooks covering evaluation metrics and the relevant fastai documentation.  Examining research papers on medical image segmentation and exploring the source code of established segmentation libraries like those mentioned above will provide a deeper understanding of the implementation details and the nuances involved.  Focusing on literature specific to your application domain (e.g., medical imaging, satellite imagery) is key to understanding the practical implications of different metrics in specific contexts.
