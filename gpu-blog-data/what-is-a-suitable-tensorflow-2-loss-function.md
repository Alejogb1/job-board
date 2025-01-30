---
title: "What is a suitable TensorFlow 2 loss function for semantic segmentation with two classes?"
date: "2025-01-30"
id: "what-is-a-suitable-tensorflow-2-loss-function"
---
Semantic segmentation, particularly when dealing with only two classes, presents a specific set of considerations when choosing an appropriate loss function. The most crucial aspect is that standard cross-entropy loss can be significantly skewed by imbalanced class representation, a common scenario in binary segmentation problems like medical imaging (e.g., tumor vs. background). Therefore, selecting a loss function that mitigates the impact of this imbalance is essential. Based on several projects I've worked on involving satellite image analysis and lesion detection, I've found binary cross-entropy with class weighting and, in certain cases, Dice loss to be particularly effective.

The inherent problem with basic binary cross-entropy is that it treats all pixels equally, regardless of their class membership. If, for instance, the positive class (e.g., the object of interest) constitutes a small fraction of the overall image, the model can achieve reasonably low loss by predicting most pixels as the negative class. The gradient updates in this case become biased towards the dominant class, hindering the learning of accurate segmentation boundaries for the minority class.

A simple yet impactful improvement involves incorporating class weights into the binary cross-entropy calculation. These weights inversely correspond to the frequency of each class in the training set. For example, if the negative class occupies 90% of the pixels and the positive class 10%, the negative class could be assigned a weight of 0.11 and the positive class 1.0 (or multiples thereof). This modification forces the model to pay more attention to errors made on the positive class, resulting in more balanced learning.

The formula for weighted binary cross-entropy is as follows, applied on a per-pixel basis:

`- w_positive * y * log(p) - w_negative * (1 - y) * log(1 - p)`

Where:
*   `y` is the true label (0 or 1)
*   `p` is the predicted probability of belonging to class 1
*   `w_positive` is the weight for the positive class
*   `w_negative` is the weight for the negative class

Another valuable alternative to cross-entropy, especially when addressing class imbalance, is Dice loss. Unlike cross-entropy, which assesses the per-pixel error rate, Dice loss measures the overlap between predicted and ground-truth segmentation maps. This metric directly correlates with how well the segmented regions match the target regions. It is therefore a more directly applicable metric for evaluating the quality of the segmentation result.

The Dice coefficient, which serves as the basis for Dice loss, is calculated as:

`2 * |A ∩ B| / (|A| + |B|)`

Where:
*   `A` is the set of pixels belonging to class 1 in the predicted segmentation map
*   `B` is the set of pixels belonging to class 1 in the ground-truth segmentation map
*   `| |` denotes the number of elements in a set

Dice loss is simply the negation of the Dice coefficient or 1 minus the Dice Coefficient. It is defined as:

`1 - (2 * |A ∩ B| / (|A| + |B|))`

In practice, a small smoothing term (epsilon) is typically added to the numerator and denominator to avoid division by zero when the true and predicted segmentations do not overlap. This can help prevent gradient explosions. Additionally, Dice loss is usually averaged across the batch size.

Here are three code examples demonstrating both of these loss functions within the TensorFlow 2 framework:

**Example 1: Weighted Binary Cross-Entropy**

```python
import tensorflow as tf

def weighted_binary_crossentropy(y_true, y_pred, w_positive=1.0, w_negative=1.0):
  """
    Calculates the weighted binary cross-entropy loss.

    Args:
        y_true: The ground truth labels (batch_size, height, width, 1).
        y_pred: The predicted probabilities (batch_size, height, width, 1).
        w_positive: Weight for positive class.
        w_negative: Weight for negative class.

    Returns:
       Weighted binary cross-entropy loss.
  """
  y_true = tf.cast(y_true, tf.float32) # Convert y_true to float for numerical stability
  epsilon = 1e-7 # Small value for preventing log(0)
  y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon) # Clip prediction values
  bce = - (w_positive * y_true * tf.math.log(y_pred) + w_negative * (1 - y_true) * tf.math.log(1 - y_pred))
  return tf.reduce_mean(bce)


# Example usage
y_true_example = tf.constant([[[[1.0]], [[0.0]]], [[[0.0]], [[1.0]]]]) # Example labels for a batch of two 2x1 images
y_pred_example = tf.constant([[[[0.9]], [[0.1]]], [[[0.2]], [[0.8]]]]) # Example predictions
positive_weight = 1.0
negative_weight = 0.2 # Lower weight for negative class

loss_example = weighted_binary_crossentropy(y_true_example, y_pred_example, w_positive=positive_weight, w_negative=negative_weight)
print(f"Weighted Binary Cross-Entropy Loss: {loss_example.numpy()}")
```

This function `weighted_binary_crossentropy` takes the ground truth, predictions, and weights as inputs. It applies the weighted cross-entropy formula, averages the result across the batch and spatial dimensions, and returns a single loss value. I've added clipping of predictions to avoid issues when probabilities are 0 or 1 exactly. The `tf.reduce_mean` operation aggregates the per-pixel results into a single scalar loss value.

**Example 2: Dice Loss**

```python
import tensorflow as tf

def dice_loss(y_true, y_pred, smooth=1e-7):
    """
        Calculates the Dice loss.

        Args:
            y_true: The ground truth labels (batch_size, height, width, 1).
            y_pred: The predicted probabilities (batch_size, height, width, 1).
            smooth: Smoothing value to avoid division by zero.

        Returns:
            Dice loss.
    """

    y_true = tf.cast(y_true, tf.float32) # Convert y_true to float for numerical stability
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])
    dice = (2 * intersection + smooth) / (union + smooth)
    return 1 - tf.reduce_mean(dice)

# Example usage
y_true_example = tf.constant([[[[1.0]], [[0.0]]], [[[0.0]], [[1.0]]]]) # Example labels for a batch of two 2x1 images
y_pred_example = tf.constant([[[[0.9]], [[0.1]]], [[[0.2]], [[0.8]]]]) # Example predictions


loss_example = dice_loss(y_true_example, y_pred_example)
print(f"Dice Loss: {loss_example.numpy()}")

```

The `dice_loss` function computes the Dice coefficient between the prediction and ground truth. The smoothing constant prevents division by zero. It returns one minus the average Dice coefficient across the batch. I've made sure the operation is numerically stable.

**Example 3: Combined Weighted BCE and Dice Loss**

```python
import tensorflow as tf

def combined_loss(y_true, y_pred, w_positive=1.0, w_negative=1.0, dice_weight=0.5):
    """
      Calculates a combined loss of weighted binary cross-entropy and Dice loss.

      Args:
        y_true: The ground truth labels (batch_size, height, width, 1).
        y_pred: The predicted probabilities (batch_size, height, width, 1).
        w_positive: Weight for positive class in BCE loss.
        w_negative: Weight for negative class in BCE loss.
        dice_weight: Weight for Dice loss in the combined loss.

      Returns:
          Combined weighted BCE and Dice loss.
    """
    bce_loss = weighted_binary_crossentropy(y_true, y_pred, w_positive, w_negative)
    dice_loss_val = dice_loss(y_true, y_pred)
    return bce_loss * (1 - dice_weight) + dice_loss_val * dice_weight


# Example usage
y_true_example = tf.constant([[[[1.0]], [[0.0]]], [[[0.0]], [[1.0]]]]) # Example labels for a batch of two 2x1 images
y_pred_example = tf.constant([[[[0.9]], [[0.1]]], [[[0.2]], [[0.8]]]]) # Example predictions
positive_weight = 1.0
negative_weight = 0.2 # Lower weight for negative class
dice_weight = 0.7 # More emphasis on Dice loss


loss_example = combined_loss(y_true_example, y_pred_example, w_positive=positive_weight, w_negative=negative_weight, dice_weight=dice_weight)
print(f"Combined Loss: {loss_example.numpy()}")

```

The `combined_loss` function allows for balancing the contribution of both weighted BCE and Dice loss to a single total loss that is then used for backpropagation. A common choice is a value between 0.3 and 0.7 to balance the effect of both loss functions.

Choosing between these loss functions largely depends on the particular dataset characteristics. If the class imbalance is not severe, weighted binary cross-entropy often performs well. However, if there are instances of very small objects relative to the whole image, Dice loss might produce better segmentations. The combined loss function provides a flexible approach by incorporating benefits from both, often achieving a good trade-off. For further insight and advanced techniques, consulting literature on semantic segmentation and medical image analysis would be beneficial. These include journals, proceedings from computer vision and medical imaging conferences, and books on deep learning architectures for image processing.
