---
title: "How can multiple rectangles be detected using a simple CNN?"
date: "2025-01-30"
id: "how-can-multiple-rectangles-be-detected-using-a"
---
A core challenge in object detection, especially with convolutional neural networks (CNNs), lies not just in identifying one instance of an object, but in accurately locating and classifying multiple instances simultaneously. Consider a scenario involving a production line where circuit boards, each containing several integrated circuit chips (which we will approximate as rectangles for simplicity), need to be inspected for defects. Traditional, image-wide classification CNNs, trained to output a single class label, are insufficient for this task. I've personally encountered this while developing automated inspection systems, and this experience underscores the need for a different approach.

My primary experience reveals that direct application of a classification network, even one that is adept at recognizing rectangles, doesn't solve the multiple rectangle detection problem. This is because classification outputs a single label for the entire input image. It does not provide bounding box coordinates, nor the confidence that each bounding box truly contains an object of interest. Therefore, a detection approach that predicts not only class probabilities but also bounding box information (location, size) is required. The core modification necessary is the introduction of a mechanism to output multiple predictions per input image.

The process fundamentally involves adapting the typical classification-focused CNN architecture. The convolution layers responsible for feature extraction remain largely similar. However, we need to replace the final fully connected layer, which would normally produce class probabilities, with a structure that outputs, for each potential rectangle: 1) its class label (or if it's a rectangle, its objectness score), 2) its bounding box coordinates (typically as x, y, width, and height), and 3) confidence scores.

This output structure is achieved using several key architectural modifications. One common approach involves anchor boxes (or prior boxes). We create a set of predefined bounding box sizes and aspect ratios which are distributed across the image. These anchor boxes are essentially guesses of what bounding boxes may exist in the image. The network is then trained to regress the offsets to each of these anchor boxes, thereby predicting the final bounding boxes. The class label, or objectness, is predicted for each anchor box as well, so the system can discard anchor boxes that do not appear to contain a rectangle.

For example, if we define ten anchors, our prediction output might be of the form `[class_1, x_1, y_1, w_1, h_1, confidence_1, ..., class_10, x_10, y_10, w_10, h_10, confidence_10]`. If our task involves only detecting one class (rectangles), `class_i` can become simply a measure of confidence in the prediction of a rectangle, and we remove it when it is zero. We will discuss such an objectness score instead.

Let’s consider a simplified example using TensorFlow and Keras. I often use such frameworks because of their flexibility and wide community support.

```python
import tensorflow as tf
from tensorflow.keras import layers, models

def build_rectangle_detector(input_shape=(64, 64, 3), num_anchors=10):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.MaxPool2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPool2D((2, 2)),
        layers.Flatten(),
        # Prediction output: for each anchor, coordinates (x,y,w,h) and confidence
        layers.Dense(num_anchors * 5)  # 5 = 4 coordinates + 1 confidence score.
    ])
    return model

# Example Usage
model = build_rectangle_detector()
model.summary()
```

In this first example, a very basic CNN is constructed using a sequential model. Two convolutional layers are applied, followed by a max pooling layer. The output is flattened and passed to a dense layer. The crucial thing to notice is that in the `Dense` layer, we have `num_anchors * 5`. Here, for each anchor box, we predict `x`, `y`, `width`, `height`, and the objectness confidence, allowing the network to make multiple predictions per image. This model can then be trained to regress to ground-truth bounding boxes with a corresponding objectness confidence.

The primary limitation of this simplified model is that it does not explicitly define anchor boxes or any logic for their generation. The predictions from the final layer are considered as regressions to some anchor box based on the order of the prediction. In a more realistic scenario, we use a predefined set of anchor boxes with differing shapes and sizes, whose dimensions are then adjusted. We can extend the model slightly to handle this.

```python
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

def generate_anchors(image_shape, num_anchors, anchor_ratios=[0.5, 1, 2], anchor_scales=[1, 2]):
  """Creates a list of anchors for each cell in the image."""
  img_height, img_width, _ = image_shape
  feature_map_size_x = img_width//4
  feature_map_size_y = img_height//4

  anchors = []
  for y in range(feature_map_size_y):
    for x in range(feature_map_size_x):
        for scale in anchor_scales:
            for ratio in anchor_ratios:
              width = scale * np.sqrt(ratio)
              height = scale / np.sqrt(ratio)
              center_x = (x + 0.5) / feature_map_size_x
              center_y = (y + 0.5) / feature_map_size_y
              anchors.append([center_x,center_y,width,height])

  anchors = np.array(anchors,dtype=np.float32)
  anchors = np.reshape(anchors,(feature_map_size_y*feature_map_size_x*len(anchor_scales)*len(anchor_ratios),4))
  return anchors

def build_rectangle_detector_with_anchors(input_shape=(64, 64, 3), num_anchors=10):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.MaxPool2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPool2D((2, 2)),
        layers.Flatten(),
        layers.Dense(num_anchors * 5)
    ])
    return model

# Example Usage
image_shape = (64, 64, 3)
num_anchors = len(generate_anchors(image_shape, 10, anchor_scales=[1,2], anchor_ratios=[0.5,1,2])) # num_anchors needs to be set to the real number of anchors
model = build_rectangle_detector_with_anchors(input_shape=image_shape, num_anchors=num_anchors)
model.summary()
```

Here, we’ve added a `generate_anchors` method to pre-define anchor boxes. We use a simple logic for this example, but in practice, you’d want to implement more sophisticated methods to calculate the positions of the anchor boxes. The overall network architecture remains similar but is now coupled with a predefined set of anchors which can be passed into a loss function. The network is trained to adjust each of the anchor boxes based on what is learned from the data. However, this model still lacks a crucial component: it can produce many overlapping bounding boxes for each true object, therefore requiring a filtering step to prune the number of outputted bounding boxes to a sensible number.

The final example shows how we can add a layer to use non-maximum suppression. While this logic is often placed outside of the model in postprocessing, we can include it as a layer that can be computed by the graph. This allows for us to train the network with some loss function dependent on the output of the NMS layer.

```python
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
import numpy as np


def generate_anchors(image_shape, num_anchors, anchor_ratios=[0.5, 1, 2], anchor_scales=[1, 2]):
  """Creates a list of anchors for each cell in the image."""
  img_height, img_width, _ = image_shape
  feature_map_size_x = img_width//4
  feature_map_size_y = img_height//4

  anchors = []
  for y in range(feature_map_size_y):
    for x in range(feature_map_size_x):
        for scale in anchor_scales:
            for ratio in anchor_ratios:
              width = scale * np.sqrt(ratio)
              height = scale / np.sqrt(ratio)
              center_x = (x + 0.5) / feature_map_size_x
              center_y = (y + 0.5) / feature_map_size_y
              anchors.append([center_x,center_y,width,height])

  anchors = np.array(anchors,dtype=np.float32)
  anchors = np.reshape(anchors,(feature_map_size_y*feature_map_size_x*len(anchor_scales)*len(anchor_ratios),4))
  return anchors


def non_maximum_suppression(boxes, scores, max_boxes=10, iou_threshold=0.5):
    selected_indices = tf.image.non_max_suppression(boxes, scores, max_boxes, iou_threshold)
    return K.gather(boxes, selected_indices), K.gather(scores, selected_indices)


class NMSLayer(layers.Layer):
    def __init__(self, max_boxes=10, iou_threshold=0.5, **kwargs):
        super(NMSLayer, self).__init__(**kwargs)
        self.max_boxes = max_boxes
        self.iou_threshold = iou_threshold

    def call(self, inputs):
        boxes = inputs[:, :4]  # Extract box coordinates
        scores = inputs[:, 4]  # Extract objectness score
        nms_boxes, nms_scores = non_maximum_suppression(boxes, scores, self.max_boxes, self.iou_threshold)
        return tf.concat([nms_boxes, tf.expand_dims(nms_scores,-1)],axis=1) # Return boxes with the scores

def build_rectangle_detector_with_nms(input_shape=(64, 64, 3), num_anchors=10):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.MaxPool2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPool2D((2, 2)),
        layers.Flatten(),
        layers.Dense(num_anchors * 5),
        layers.Reshape((num_anchors,5)),
        NMSLayer(max_boxes=10, iou_threshold=0.5)
    ])
    return model


# Example usage
image_shape = (64, 64, 3)
num_anchors = len(generate_anchors(image_shape, 10, anchor_scales=[1,2], anchor_ratios=[0.5,1,2]))
model = build_rectangle_detector_with_nms(input_shape=image_shape, num_anchors=num_anchors)
model.summary()
```

Here, an `NMSLayer` has been defined which takes the raw box predictions and applies non maximum suppression on them. This layer helps to drastically reduce the number of outputted boxes. Note that the network also includes an intermediate reshaping to apply this layer.

It is important to emphasize that while this shows an example of training a CNN for object detection, it will need several crucial additions, including proper loss function implementation and dataset preparation, to work effectively.

For deeper exploration into object detection using CNNs, consider looking at the research papers on Fast R-CNN, Faster R-CNN, YOLO, and SSD. Textbooks on deep learning or computer vision can also provide more theoretical context. Online courses focusing on object detection using TensorFlow or PyTorch can provide a hands-on learning experience. Experimenting directly with these concepts is invaluable. This approach, while simplified, outlines the core modifications needed to tackle the challenge of detecting multiple rectangles using a CNN.
