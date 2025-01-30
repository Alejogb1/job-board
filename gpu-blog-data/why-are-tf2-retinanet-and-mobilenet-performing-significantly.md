---
title: "Why are TF2 RetinaNet and MobileNet performing significantly worse than YOLOv4?"
date: "2025-01-30"
id: "why-are-tf2-retinanet-and-mobilenet-performing-significantly"
---
The performance discrepancy between TF2 RetinaNet, MobileNet, and YOLOv4 in object detection stems primarily from fundamental differences in architecture, training strategies, and inherent design tradeoffs. My experience training and deploying these models in industrial settings has revealed specific reasons for YOLOv4's superior results in many scenarios, particularly concerning speed-accuracy balance.

**Understanding the Architectural Distinctions**

TF2 RetinaNet, built upon the Feature Pyramid Network (FPN) architecture and utilizing a ResNet backbone, prioritizes handling class imbalance inherent in object detection datasets. It achieves this through the focal loss function, which effectively down-weights the contribution of easily classified background examples. This approach, however, comes at a computational cost, affecting speed. While the FPN creates multiple feature maps at different scales, allowing the model to detect objects of varying sizes, the network's depth and complexity, particularly with ResNet backbones, impact inference time.

MobileNet, designed for resource-constrained environments, uses depthwise separable convolutions to reduce the number of parameters and computations. This efficiency is achieved by sacrificing model capacity compared to architectures like ResNet. Consequently, MobileNet’s accuracy is generally lower than larger, more complex models. When integrated with object detection frameworks, it becomes a balance between speed and performance. Though it can be used in a RetinaNet architecture, it will remain less performant compared to a stronger backbone.

YOLOv4, on the other hand, incorporates a multitude of architectural innovations. It employs a CSPDarknet53 backbone, which is computationally efficient and allows for deep feature extraction. It further combines it with SPP (Spatial Pyramid Pooling) and PAN (Path Aggregation Network) to handle multiscale features. Its anchor-based detection mechanism, coupled with a custom bounding box regression strategy, and carefully tuned loss functions, enables a different trade-off. The architecture itself has been deliberately optimised to deliver superior speed compared to the architecture of RetinaNet and MobileNet.

**Training Strategies and Impact on Performance**

The training procedure further contributes to the performance disparities. RetinaNet benefits significantly from a carefully implemented focal loss, which addresses class imbalance but needs proper hyperparameter tuning and convergence. Training RetinaNet requires a longer time and more computational power compared to YOLOv4.

MobileNet training focuses on efficient learning using limited resources. It is generally pretrained on ImageNet, making it quicker to converge. However, MobileNet-based object detection models struggle to attain the same performance metrics as more complex models due to their design for reduced computation and parameter count. Fine-tuning may help, but it is not typically enough to bridge the gap.

YOLOv4 employs a distinct training regimen which heavily uses mosaic augmentation, a complex image augmentation scheme, for better generalisation. Its loss function is a combination of several different losses that work well together. It also uses a mixture of data augmentation techniques to enhance its robustness to variations in input images. These training techniques, which require careful optimization, are specifically tailored to the architecture and achieve high performance.

**Code Examples and Commentary**

*Example 1: RetinaNet with ResNet50 Backbone (TensorFlow)*

```python
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow_addons.losses import SigmoidFocalCrossEntropy

def build_retinanet(num_classes, backbone='resnet50'):
  if backbone == 'resnet50':
    base_model = tf.keras.applications.ResNet50(include_top=False, input_shape=(None, None, 3))
    c3_output = base_model.get_layer('conv3_block4_out').output
    c4_output = base_model.get_layer('conv4_block6_out').output
    c5_output = base_model.get_layer('conv5_block3_out').output
  else:
    raise ValueError("Unsupported Backbone")


  # Feature Pyramid Network (FPN) construction.
  p5 = layers.Conv2D(256, 1, padding="same")(c5_output)
  m4 = layers.Conv2D(256, 1, padding="same")(c4_output)
  m4 = layers.add([m4, layers.UpSampling2D(size=(2, 2))(p5)])
  m3 = layers.Conv2D(256, 1, padding="same")(c3_output)
  m3 = layers.add([m3, layers.UpSampling2D(size=(2, 2))(m4)])

  p5 = layers.Conv2D(256, 3, padding="same")(p5)
  p4 = layers.Conv2D(256, 3, padding="same")(m4)
  p3 = layers.Conv2D(256, 3, padding="same")(m3)


  cls_outputs = []
  box_outputs = []
  for feature_map in [p3, p4, p5]:
      cls_conv = layers.Conv2D(256, 3, padding='same', activation="relu")(feature_map)
      cls_output = layers.Conv2D(num_classes*9, 3, padding='same', activation="sigmoid")(cls_conv)
      cls_outputs.append(cls_output)

      box_conv = layers.Conv2D(256, 3, padding='same', activation="relu")(feature_map)
      box_output = layers.Conv2D(4*9, 3, padding='same')(box_conv) #4 bounding box offsets and 9 anchors
      box_outputs.append(box_output)

  cls_outputs = tf.concat(cls_outputs, axis=1)
  box_outputs = tf.concat(box_outputs, axis=1)
  
  return Model(inputs=base_model.input, outputs=[cls_outputs,box_outputs])

# Example Loss Calculation
def retinanet_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
  cls_pred, box_pred = y_pred
  cls_true, box_true = y_true

  # cls loss using SigmoidFocalCrossEntropy (TensorFlow Addons)
  cls_loss = SigmoidFocalCrossEntropy(alpha=alpha,gamma=gamma)(cls_true, cls_pred)

  # box loss
  box_loss_mask = tf.expand_dims(cls_true, -1) #mask only for objects
  box_loss = tf.reduce_sum(tf.math.abs(box_true - box_pred), axis=-1)*box_loss_mask
  box_loss = tf.reduce_sum(box_loss)/tf.reduce_sum(box_loss_mask)

  return tf.reduce_mean(cls_loss) + box_loss
```
*Commentary:* This code snippet illustrates the core architectural components of a RetinaNet implemented in TensorFlow. The FPN construction, the separate classification and bounding box prediction branches, and the use of SigmoidFocalCrossEntropy are highlighted. This example demonstrates why RetinaNet can be computationally intensive with its larger backbone and deeper architecture compared to MobileNet, and, while effective at tackling class imbalance, often falls behind YOLOv4 in terms of speed.

*Example 2: MobileNetV2 with SSD-Lite framework (TensorFlow)*

```python
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow_addons.losses import SigmoidFocalCrossEntropy


def build_mobilenet_ssd(num_classes,input_shape=(300,300,3)):
    base_model = tf.keras.applications.MobileNetV2(include_top=False, input_shape=input_shape)
    output_1 = base_model.get_layer('block_13_expand_relu').output
    output_2 = base_model.get_layer('block_16_project').output

    # Extra Feature Layers
    x = layers.Conv2D(256, kernel_size=(1,1), padding="same", activation="relu")(output_2)
    x = layers.Conv2D(256, kernel_size=(3,3), padding="same", strides=(2,2), activation="relu")(x)
    extra_1 = x

    x = layers.Conv2D(128, kernel_size=(1,1), padding="same", activation="relu")(extra_1)
    x = layers.Conv2D(128, kernel_size=(3,3), padding="same", strides=(2,2), activation="relu")(x)
    extra_2 = x

    x = layers.Conv2D(128, kernel_size=(1,1), padding="same", activation="relu")(extra_2)
    x = layers.Conv2D(128, kernel_size=(3,3), padding="same", strides=(2,2), activation="relu")(x)
    extra_3 = x

    feature_maps = [output_1, output_2, extra_1, extra_2, extra_3]

    cls_outputs = []
    box_outputs = []
    for feature_map in feature_maps:
      cls_conv = layers.Conv2D(256, 3, padding='same', activation="relu")(feature_map)
      cls_output = layers.Conv2D(num_classes*6, 3, padding='same', activation="sigmoid")(cls_conv)
      cls_outputs.append(cls_output)

      box_conv = layers.Conv2D(256, 3, padding='same', activation="relu")(feature_map)
      box_output = layers.Conv2D(4*6, 3, padding='same')(box_conv) #4 bounding box offsets and 6 anchors
      box_outputs.append(box_output)
    cls_outputs = tf.concat(cls_outputs, axis=1)
    box_outputs = tf.concat(box_outputs, axis=1)
  
    return Model(inputs=base_model.input, outputs=[cls_outputs,box_outputs])

def mobilenet_ssd_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
  cls_pred, box_pred = y_pred
  cls_true, box_true = y_true

    # cls loss using SigmoidFocalCrossEntropy (TensorFlow Addons)
  cls_loss = SigmoidFocalCrossEntropy(alpha=alpha,gamma=gamma)(cls_true, cls_pred)

  # box loss
  box_loss_mask = tf.expand_dims(cls_true, -1) #mask only for objects
  box_loss = tf.reduce_sum(tf.math.abs(box_true - box_pred), axis=-1)*box_loss_mask
  box_loss = tf.reduce_sum(box_loss)/tf.reduce_sum(box_loss_mask)

  return tf.reduce_mean(cls_loss) + box_loss
```
*Commentary:* This code shows how MobileNetV2 is typically used in an object detection framework, with extra layers after the base model to generate multiple feature maps, similar to SSD. Note the use of depthwise separable convolutions inherent in the base model. While lightweight, the reduced capacity affects its performance and its accuracy in most scenarios will not match YOLOv4.

*Example 3: (Conceptual representation of YOLOv4 architecture)*
```python
import tensorflow as tf
from tensorflow.keras import layers, Model

def build_yolov4_head(features,num_classes):
  
  conv1 = layers.Conv2D(512,kernel_size=(3,3),padding="same",activation='relu')(features)
  conv2 = layers.Conv2D(1024,kernel_size=(3,3),padding="same",activation='relu')(conv1)
  conv3 = layers.Conv2D(512,kernel_size=(1,1),padding="same",activation='relu')(conv2)
  conv4 = layers.Conv2D(1024,kernel_size=(3,3),padding="same",activation='relu')(conv3)
  conv5 = layers.Conv2D(512,kernel_size=(1,1),padding="same",activation='relu')(conv4)

  cls_output= layers.Conv2D(num_classes*3,kernel_size=(1,1),padding="same", activation="sigmoid")(conv5)
  box_output = layers.Conv2D(4*3,kernel_size=(1,1),padding="same")(conv5)
  
  return cls_output, box_output
  

# this is a highly simplified version of the head, ignoring the CSP backbones and other parts
def build_yolov4(num_classes,input_shape=(608,608,3)):
    # the model uses a CSPDarknet backbone with SPP and PAN. these are omitted here for brevity
    # dummy input to simulate a feature map
    inputs = tf.keras.Input(shape=input_shape)
    
    #simplified backbone
    x=layers.Conv2D(16,kernel_size=(3,3),padding="same")(inputs)
    x=layers.Conv2D(32,kernel_size=(3,3),padding="same")(x)
    feature_map1 =layers.Conv2D(64,kernel_size=(3,3),padding="same")(x)
    x=layers.Conv2D(128,kernel_size=(3,3),padding="same")(feature_map1)
    feature_map2 = layers.Conv2D(256,kernel_size=(3,3),padding="same")(x)

    cls_output1, box_output1 = build_yolov4_head(feature_map2,num_classes)
    #downsample the feature_map2
    x = layers.Conv2D(128,kernel_size=(3,3),padding="same", strides=(2,2))(feature_map2)
    feature_map3 = layers.concatenate([x, feature_map1],axis=-1) #panning

    cls_output2, box_output2 = build_yolov4_head(feature_map3,num_classes)

    cls_outputs = tf.concat([cls_output1, cls_output2], axis=1)
    box_outputs = tf.concat([box_output1, box_output2], axis=1)
    return Model(inputs=inputs, outputs=[cls_outputs,box_outputs])

def yolo_loss(y_true, y_pred):
  cls_pred, box_pred = y_pred
  cls_true, box_true = y_true

  cls_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(cls_true, cls_pred)) #cross entropy
  box_loss_mask = tf.expand_dims(cls_true, -1) #mask only for objects
  box_loss = tf.reduce_sum(tf.math.abs(box_true - box_pred), axis=-1)*box_loss_mask
  box_loss = tf.reduce_sum(box_loss)/tf.reduce_sum(box_loss_mask)

  return cls_loss+box_loss
```
*Commentary:* This simplified code is conceptual and does not represent the entire architecture of YOLOv4, but showcases the head design and the way YOLOv4 uses multiple prediction layers. Crucially, this architecture allows for quicker inference speed compared to the FPN of RetinaNet or the efficiency-focused design of MobileNet. This allows YOLOv4 to deliver higher performance in a given setting compared to the former two models, as well as achieving superior performance for a given inference speed.

**Resource Recommendations**

For a deeper understanding of these architectures and training methodologies, I recommend exploring several resources, such as academic publications outlining the original RetinaNet, MobileNet, and YOLOv4 papers. Further insight can be obtained by consulting advanced deep learning books that discuss object detection architectures, loss functions, and common training techniques. Also, engaging with online machine learning communities can offer practical tips and troubleshooting guidance. Finally, examining public code repositories implementing these models can provide clarity on the implementation details and optimization techniques.
