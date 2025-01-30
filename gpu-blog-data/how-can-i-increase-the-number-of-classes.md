---
title: "How can I increase the number of classes in a TensorFlow custom object detection model using additional data?"
date: "2025-01-30"
id: "how-can-i-increase-the-number-of-classes"
---
A challenge frequently encountered when extending custom object detection models is incorporating new object classes not present during the initial training. Direct retraining of the entire model from scratch, while an option, becomes computationally prohibitive and inefficient as datasets grow. Instead, I've found a more effective approach involves leveraging transfer learning and fine-tuning specific layers within the existing model using the new class data. This method preserves the learned features of the original classes, enabling the model to quickly adapt to the new classes.

The core concept rests on the notion that a convolutional neural network (CNN), which forms the base of most object detection models, learns hierarchical features. The earlier layers in the network capture low-level features like edges and corners, which are generally applicable across various object categories. Subsequent layers extract more complex features, and the final few layers are highly specific to the classification task. Thus, to introduce new classes, the modifications primarily concern the later layers involved in class prediction, often referred to as the classification and bounding box regression heads.

To increase the number of classes, the following process should be considered. First, the new labeled dataset for these new classes must be properly formatted, which usually aligns with the format of the initial training data. The new data must contain images along with their corresponding bounding box annotations specifying the location and class of each object instance. Assuming we are utilizing the TensorFlow Object Detection API, this translates to creating TFRecords that adhere to the API's input data format requirements.

Second, the base model is loaded with its pre-trained weights. It is imperative to strategically choose which layers to fine-tune. Typically, the feature extraction portion of the network (i.e., the backbone) is kept frozen or trained with a very low learning rate to preserve previously learned knowledge. The layers responsible for class classification and bounding box regression are the ones targeted for significant adjustment. Specifically, the final classification layer must have its number of output nodes increased to match the total number of classes (old and new). The bounding box regression head, while not directly dependent on the number of classes, should also be fine-tuned since its weights are often closely coupled to the classification head.

Lastly, careful selection of hyperparameters, especially the learning rate, is critical during this fine-tuning stage. Overly aggressive adjustments to the classification and regression layers can disrupt the feature representations already learned by the base model. A much lower learning rate for the base layers versus the newer classifier layers allows for efficient adaptation to the new classes. The training then proceeds in the standard manner, and the model will progressively learn to associate input features with the added object classes.

Here are three examples outlining this process in a hypothetical scenario using the TensorFlow Object Detection API with a Faster R-CNN model. The examples will focus on modifying the model’s configuration file and code snippets that highlight the adaptation process. These examples illustrate concepts, and exact implementations may vary based on specific model configurations and frameworks.

**Example 1: Modification of the Configuration File**

This example shows how to update a hypothetical model configuration file (`faster_rcnn_config.pbtext`) to account for new classes. Assume there are initially 3 classes and we need to add 2 more.

```
model {
  faster_rcnn {
    ... # Omitted base network configuration
    num_classes: 5 # Increased from 3 to 5
    image_resizer {
      ...
    }
    feature_extractor {
        ... # Omitted backbone configuration
    }
    box_coder {
        faster_rcnn_box_coder {
        }
    }
    faster_rcnn_box_predictor {
      ...
      fc_hyperparams {
        ...
      }
      conv_hyperparams {
        ...
      }
    }
    faster_rcnn_post_processing {
      ...
    }
   first_stage_anchor_generator {
        ...
   }
   first_stage_box_predictor_arg_scope {
      ...
   }
   first_stage_post_processing {
     ...
   }
   roi_pooling_layer {
        ...
   }
   second_stage_box_predictor {
    ...
      use_dropout: false
      dropout_keep_prob: 1.0
      fully_connected_layers {
         num_output: 1024 # Example: Keep intermediate FC layers
        }
      fully_connected_layers {
         num_output: 1024 # Example: Keep intermediate FC layers
      }
      class_prediction_bias_initializer: {
        zeros_initializer {}
      }
      class_prediction_num_outputs: 5 # Reflects the updated total classes
    }
    ...
  }
}
train_config {
  ...
  batch_size: 24 # Example of other configuration
  ...
}
```

**Commentary:**

The key changes are in `num_classes` within the `faster_rcnn` block and `class_prediction_num_outputs` inside `second_stage_box_predictor`. These are modified to reflect the total number of object classes, old plus the newly added ones. The base model configuration remains unchanged, allowing the model to retain previously learned features. Note: The `train_config` section may need adjustments to training parameters.

**Example 2: Snippet illustrating the loading of a pre-trained checkpoint and freezing of backbone layers**

The following is a conceptual Python snippet demonstrating how to load a pre-trained model, freeze the backbone, and set a low learning rate for these layers. This would be integrated with the TensorFlow Object Detection API during training.

```python
import tensorflow as tf
from object_detection.builders import model_builder
from object_detection.protos import pipeline_pb2

pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
with tf.io.gfile.GFile('faster_rcnn_config.pbtext', 'rb') as f:
  config_str = f.read()
  text_format.Merge(config_str, pipeline_config)

model_config = pipeline_config.model
detection_model = model_builder.build(
    model_config=model_config, is_training=True)


ckpt_path = 'path/to/pretrained/model/checkpoint'  # Pre-trained model checkpoint path
checkpoint = tf.train.Checkpoint(detection_model=detection_model)
checkpoint.restore(ckpt_path).expect_partial()


trainable_variables = detection_model.trainable_variables

# Identifying trainable layers. In this example we filter based on name.
backbone_trainable = [var for var in trainable_variables if 'feature_extractor' in var.name ]
classification_trainable = [var for var in trainable_variables if 'second_stage_box_predictor' in var.name ]

# Example of configuring optimizers using a manual strategy.
optimizer_backbone = tf.keras.optimizers.Adam(learning_rate=0.00001) # Very low learning rate for backbone
optimizer_classifier = tf.keras.optimizers.Adam(learning_rate=0.001) # Moderate Learning Rate

# Manually apply the respective optimizers to the layers based on filtering.
optimizer_variables = {
    var: optimizer_backbone if var in backbone_trainable else optimizer_classifier  for var in trainable_variables
}

def train_step(images, labels):
    with tf.GradientTape() as tape:
        prediction_dict = detection_model(images, training=True)
        losses_dict = detection_model.loss(prediction_dict, labels)
        total_loss = losses_dict['Loss/total_loss']
        gradients = tape.gradient(total_loss, detection_model.trainable_variables)
    for var, grad in zip(detection_model.trainable_variables, gradients):
        if grad is not None:
          optimizer_variables[var].apply_gradients([(grad, var)])
```
**Commentary:**

This example loads a pre-trained model, then separates the trainable variables based on their names to allow for specific adjustments for the backbone layers, where we apply a very small learning rate. The `train_step` function performs gradient calculations and updates trainable variables based on the configured optimizers.

**Example 3: Data loading example.**

This shows how new training data might be loaded. It presumes the TFRecords are in the typical TFOD format.

```python
import tensorflow as tf
from object_detection.utils import dataset_util


def load_tf_records(record_files, config):
    dataset = tf.data.TFRecordDataset(record_files)

    def parser(serialized_example):
        features = {
           'image/encoded': tf.io.FixedLenFeature((), tf.string),
            'image/format': tf.io.FixedLenFeature((), tf.string, default_value='jpeg'),
            'image/height': tf.io.FixedLenFeature((), tf.int64),
            'image/width': tf.io.FixedLenFeature((), tf.int64),
            'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
            'image/object/class/label': tf.io.VarLenFeature(tf.int64)
        }
        parsed = tf.io.parse_single_example(serialized_example, features)

        image = tf.io.decode_jpeg(parsed['image/encoded'], channels=3)
        image_shape = tf.stack([parsed['image/height'], parsed['image/width'], 3])
        image = tf.reshape(image, image_shape)
        
        xmin = tf.sparse.to_dense(parsed['image/object/bbox/xmin'])
        ymin = tf.sparse.to_dense(parsed['image/object/bbox/ymin'])
        xmax = tf.sparse.to_dense(parsed['image/object/bbox/xmax'])
        ymax = tf.sparse.to_dense(parsed['image/object/bbox/ymax'])
        labels = tf.cast(tf.sparse.to_dense(parsed['image/object/class/label']), dtype=tf.int32)
        bboxes = tf.stack([ymin, xmin, ymax, xmax], axis=1)

        # We want to convert labels 1 -> 0 and 2 ->1 if our initial model had only 2 classes. 
        # This is important to do before merging the original dataset
        labels = tf.where(tf.greater(labels, 0), labels - 1, labels)
        
        return image, {"groundtruth_boxes": bboxes, "groundtruth_classes": labels}


    dataset = dataset.map(parser)
    dataset = dataset.batch(config.train_config.batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset
```

**Commentary:**

This function demonstrates how to load new data from TFRecords files, decode images, extract bounding box coordinates and object labels, and batch the data. Notably, if the class labels begin at a number greater than zero, these need to be shifted downwards (e.g., new classes with labels 3 and 4 must become 1 and 2). The new classes’ labels have to be contiguous with the original training classes. It also demonstrates handling sparse tensors often used in the TFOD API to represent bounding boxes and class labels. In a real situation, additional data augmentation would also be done during this step.

In summary, successfully increasing the number of classes in a pre-trained object detection model involves meticulous configuration adjustments, selective layer fine-tuning, and careful data preparation. By focusing the updates on the classification head and using a low learning rate for the base layers, one can efficiently extend existing models to new object types.

For further learning and practical implementations, several resources may be beneficial: the TensorFlow Object Detection API documentation, research publications focusing on transfer learning for object detection (such as related to Faster R-CNN, SSD, and EfficientDet architectures), and various online courses that cover the topic of fine-tuning neural networks. Investigating specific object detection frameworks other than the TensorFlow API, such as PyTorch, will further broaden one’s approach to these kinds of tasks.
