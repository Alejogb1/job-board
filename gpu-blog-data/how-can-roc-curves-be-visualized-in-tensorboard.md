---
title: "How can ROC curves be visualized in TensorBoard?"
date: "2025-01-26"
id: "how-can-roc-curves-be-visualized-in-tensorboard"
---

TensorBoard, a visualization tool within the TensorFlow ecosystem, does not inherently render Receiver Operating Characteristic (ROC) curves directly as a specific panel type. Instead, ROC curve data needs to be structured and logged in a manner that leverages TensorBoard’s existing capabilities, specifically scalar and image summaries, to achieve a visual representation. My experience building several fraud detection models has ingrained in me this particular approach due to the crucial nature of evaluating classifier performance at varying thresholds.

The core challenge lies in the fact that an ROC curve is not a single scalar value but a two-dimensional plot illustrating the trade-off between the True Positive Rate (TPR) and False Positive Rate (FPR) across a range of classification thresholds. TensorBoard primarily handles scalar metrics and images. Thus, we must log sufficient data to recreate this plot. I found the most practical approach is to compute TPR and FPR at discrete thresholds and store these pairs as scalar summaries. While this allows for a reconstruction of the curve, another complimentary method would be to actually render the ROC plot as an image, especially for more complex models where threshold discretization may be too sparse.

To log the necessary data, one must calculate true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN) at various thresholds using the model's predicted probabilities. These statistics are then utilized to compute TPR (TP / (TP + FN)) and FPR (FP / (FP + TN)). This process needs to be performed periodically during training or evaluation. In essence, we are generating the data needed to draw the curve manually; TensorBoard simply offers the canvas for that generated data.

Here's how to achieve this in practice with three code examples illustrating different techniques:

**Example 1: Logging TPR and FPR as Scalar Summaries**

This is the foundational method, using scalar summaries for threshold-based metrics. The goal is to provide a series of (FPR, TPR) coordinates that can be visualized.

```python
import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_curve

def log_roc_scalars(writer, y_true, y_prob, step, num_thresholds=20):
    """Logs TPR and FPR as scalar summaries for ROC curve reconstruction.

       Args:
          writer: TensorBoard summary writer.
          y_true: Ground truth labels (binary).
          y_prob: Predicted probabilities.
          step: Global training step.
          num_thresholds: Number of thresholds to evaluate.
    """
    thresholds = np.linspace(0, 1, num_thresholds)
    fpr, tpr, _ = roc_curve(y_true, y_prob) #Use sklearn to get the actual tpr and fpr values at given thresholds

    with writer.as_default():
        for i in range(len(thresholds)):
            tf.summary.scalar(f"roc/fpr_threshold_{i}", fpr[i], step=step)
            tf.summary.scalar(f"roc/tpr_threshold_{i}", tpr[i], step=step)

# Example Usage:
if __name__ == '__main__':
    log_dir = "logs/roc_scalars"
    writer = tf.summary.create_file_writer(log_dir)
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    y_prob = np.array([0.1, 0.3, 0.8, 0.6, 0.9, 0.2, 0.7, 0.4])

    for step in range(5):
      #This would be inside the loop for training the model, simulating results on evaluation set
      y_prob=np.random.rand(8) #Generate random probs for illustration
      log_roc_scalars(writer, y_true, y_prob, step)
    writer.close()
```

In this code, `log_roc_scalars` calculates and logs the FPR and TPR for a specific set of thresholds. We obtain the (fpr, tpr) points directly from `sklearn.metrics.roc_curve` given true and predicted labels. Each pair is then logged with a unique suffix, making it possible to plot them against each other in TensorBoard’s scalar tab after grouping based on a common prefix. After executing, you would start TensorBoard, navigate to scalar graphs, filter by the prefix 'roc' and then plot all the TPR values against all the FPR values. Though not a single curve, it allows a step-by-step reconstruction of the ROC plot over training steps.

**Example 2: Logging the ROC Curve as an Image Summary**

This approach avoids manual curve reconstruction by generating the ROC plot as a matplotlib figure and logging it as an image summary.

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from io import BytesIO

def log_roc_image(writer, y_true, y_prob, step):
  """Logs ROC curve as an image.

      Args:
          writer: TensorBoard summary writer.
          y_true: Ground truth labels (binary).
          y_prob: Predicted probabilities.
          step: Global training step.
    """

  fpr, tpr, _ = roc_curve(y_true, y_prob)
  plt.figure()
  plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
  plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Receiver Operating Characteristic')
  plt.legend(loc="lower right")

  buf = BytesIO()
  plt.savefig(buf, format='png')
  buf.seek(0)
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  image = tf.expand_dims(image, 0) #Add batch dim

  with writer.as_default():
    tf.summary.image("roc_curve", image, step=step)

  plt.close() # Avoid memory leakage

#Example Usage:
if __name__ == '__main__':
  log_dir = "logs/roc_image"
  writer = tf.summary.create_file_writer(log_dir)
  y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
  y_prob = np.array([0.1, 0.3, 0.8, 0.6, 0.9, 0.2, 0.7, 0.4])
  for step in range(5):
      y_prob=np.random.rand(8)
      log_roc_image(writer, y_true, y_prob, step)
  writer.close()
```

In this example, we utilize Matplotlib to plot the ROC curve using the scikit-learn's generated fpr and tpr. The matplotlib plot is converted to a PNG image using `io.BytesIO`, and decoded into a TensorFlow tensor. Then, it is logged to TensorBoard via an image summary. The primary advantage here is the direct visualization of the curve itself, making it intuitive to interpret classifier performance. When navigating to the image tab in TensorBoard, the visualized plot will appear over different training steps. The disadvantage is that you lose the ability to interact with the data points in TensorBoard itself.

**Example 3: Combining Scalar and Image Logging**

This example combines both scalar and image summaries to provide flexibility:

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from io import BytesIO

def log_roc_metrics(writer, y_true, y_prob, step, num_thresholds=20):
   """Logs both scalar summaries and image summary for ROC curve.
      Args:
          writer: TensorBoard summary writer.
          y_true: Ground truth labels (binary).
          y_prob: Predicted probabilities.
          step: Global training step.
          num_thresholds: Number of thresholds to evaluate.
   """

   thresholds = np.linspace(0, 1, num_thresholds)
   fpr, tpr, _ = roc_curve(y_true, y_prob)

   with writer.as_default():
      for i in range(len(thresholds)):
          tf.summary.scalar(f"roc/fpr_threshold_{i}", fpr[i], step=step)
          tf.summary.scalar(f"roc/tpr_threshold_{i}", tpr[i], step=step)

   plt.figure()
   plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
   plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
   plt.xlim([0.0, 1.0])
   plt.ylim([0.0, 1.05])
   plt.xlabel('False Positive Rate')
   plt.ylabel('True Positive Rate')
   plt.title('Receiver Operating Characteristic')
   plt.legend(loc="lower right")

   buf = BytesIO()
   plt.savefig(buf, format='png')
   buf.seek(0)
   image = tf.image.decode_png(buf.getvalue(), channels=4)
   image = tf.expand_dims(image, 0)

   with writer.as_default():
      tf.summary.image("roc_curve", image, step=step)

   plt.close()

# Example Usage:
if __name__ == '__main__':
   log_dir = "logs/roc_combined"
   writer = tf.summary.create_file_writer(log_dir)
   y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
   y_prob = np.array([0.1, 0.3, 0.8, 0.6, 0.9, 0.2, 0.7, 0.4])

   for step in range(5):
      y_prob = np.random.rand(8)
      log_roc_metrics(writer, y_true, y_prob, step)
   writer.close()
```

This example integrates both techniques, logging both discrete scalar values that can be plotted and a fully rendered image of the curve. This offers the best of both worlds, allowing detailed inspection of individual threshold values through scalars while offering a holistic overview through the images, in TensorBoard.

For expanding on this in other contexts, I'd recommend the following resources: The TensorFlow documentation provides comprehensive details on working with summary operations and TensorBoard itself. Scikit-learn’s documentation provides a deep understanding of classification metrics, particularly the ROC calculation, which underpins these visualisations. Lastly, matplotlib’s documentation covers plot customization and fine tuning graph generation, which are useful when producing more polished ROC plots. While the use of scalar and image summaries may seem a bit indirect, with careful planning and design they prove flexible enough to meet the demands of visualising an ROC in TensorBoard.
