---
title: "How can I perform image segmentation using a custom dataset in TensorFlow 2.8?"
date: "2025-01-26"
id: "how-can-i-perform-image-segmentation-using-a-custom-dataset-in-tensorflow-28"
---

Image segmentation with custom datasets in TensorFlow 2.8 requires a structured approach, moving beyond standard image classification tasks. The key distinction lies in predicting a pixel-wise class label rather than a single label for the entire image. I've spent the last several months implementing a robust aerial image segmentation pipeline for urban planning, a process that illuminated several critical steps. This involves significant data preparation, custom data loading, model building, training, and evaluation.

**1. Data Preparation and Understanding**

My experience reveals that successful segmentation hinges on meticulously curated datasets. Unlike classification tasks where image labeling is straightforward, segmentation requires pixel-perfect annotations. For the urban planning project, we manually created binary masks representing building footprints within satellite imagery. The format we settled on was PNG masks, where a value of '1' denotes a pixel belonging to the target class (building) and '0' represents the background. TensorFlow can ingest these via `tf.io.decode_png`.

The crucial step before any modeling is to thoroughly understand the characteristics of your dataset. This means not just the number of images but also:

*   **Image Resolution:** Consistent input sizes reduce variance, although variations can be addressed through rescaling and cropping during the loading pipeline.
*   **Class Balance:** Highly imbalanced datasets will skew training towards the dominant class. Address this with techniques like class weighting or oversampling. We utilized class weighting to counteract the disproportionately higher amount of background pixels.
*   **Annotation Quality:** Noise and inaccuracies in annotations propagate errors through the training. Thoroughly examine and correct issues to avoid model overfitting to erroneous data. This was a costly but vital step that significantly impacted results.
*   **Data Augmentation Opportunities:** Identify transformations specific to your data. For aerial imagery, rotation, flips, and random crops were effective for increasing the diversity and robustness of the model.

**2. Custom Data Loading with TensorFlow Data API**

TensorFlow 2.8 provides a robust `tf.data` API for creating efficient input pipelines. A bespoke data loading process is critical when dealing with custom datasets. The following demonstrates how I approached loading image and mask pairs:

```python
import tensorflow as tf
import os

def load_image_mask(image_path, mask_path):
    image = tf.io.read_file(image_path)
    image = tf.io.decode_jpeg(image, channels=3) # or tf.io.decode_png if your images are PNGs
    image = tf.image.convert_image_dtype(image, tf.float32) # Normalize pixel values to [0,1]
    mask = tf.io.read_file(mask_path)
    mask = tf.io.decode_png(mask, channels=1) # Assuming single-channel PNG masks
    mask = tf.image.convert_image_dtype(mask, tf.float32) #Convert to float32
    return image, mask

def create_dataset(image_dir, mask_dir, batch_size, image_size):
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.jpg','.jpeg','.png'))] #Adjust the file extensions as needed
    mask_files = [os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.png')]

    # Ensure that images and masks align and have the same number
    assert len(image_files) == len(mask_files), "Image and mask counts do not match"
    image_files.sort()
    mask_files.sort()

    dataset = tf.data.Dataset.from_tensor_slices((image_files, mask_files))
    dataset = dataset.map(load_image_mask, num_parallel_calls=tf.data.AUTOTUNE)

    def resize_image_mask(image, mask):
        image = tf.image.resize(image, image_size)
        mask = tf.image.resize(mask, image_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR) # Nearest Neighbor for masks
        return image, mask
    dataset = dataset.map(resize_image_mask, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE).shuffle(buffer_size=100) # Use shuffle in the train data loader
    return dataset

image_dir = 'path/to/your/images'
mask_dir = 'path/to/your/masks'
batch_size = 32
image_size = (256,256) # adjust the size to your model needs
dataset = create_dataset(image_dir, mask_dir, batch_size, image_size)

# Example usage:
for image_batch, mask_batch in dataset.take(1):
    print(image_batch.shape, mask_batch.shape)
```

This code performs several actions: loading images and masks from file paths, ensuring both have compatible types (`float32`), then resizing them to the model's expected input. The most crucial step is using `tf.image.ResizeMethod.NEAREST_NEIGHBOR` when resizing masks. This prevents interpolation, preserving the binary nature of pixel labels during resizing. The `prefetch` command optimizes data loading speed, and `shuffle` ensures random order in each batch.

**3. Model Selection and Implementation**

For my segmentation task, I explored several common models. U-Net stood out as particularly effective. It's an encoder-decoder architecture that captures both high-level semantic information and low-level spatial details necessary for accurate segmentation. Here is a simplified version built using TensorFlow:

```python
import tensorflow as tf
from tensorflow.keras import layers

def conv2d_block(inputs, filters, kernel_size=3, padding='same'):
    x = layers.Conv2D(filters, kernel_size, padding=padding)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, kernel_size, padding=padding)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


def encoder_block(inputs, filters, pool_size=(2,2), padding='same'):
    x = conv2d_block(inputs, filters)
    p = layers.MaxPool2D(pool_size, padding=padding)(x)
    return x, p


def decoder_block(inputs, skip_features, filters, kernel_size=2, padding='same'):
    x = layers.Conv2DTranspose(filters, kernel_size, strides=2, padding=padding)(inputs)
    x = layers.concatenate([x, skip_features])
    x = conv2d_block(x, filters)
    return x


def build_unet(input_shape, num_classes=1):

    inputs = layers.Input(input_shape)
    # Encoder blocks
    s1, p1 = encoder_block(inputs, filters=64)
    s2, p2 = encoder_block(p1, filters=128)
    s3, p3 = encoder_block(p2, filters=256)
    s4, p4 = encoder_block(p3, filters=512)

    # Bottleneck
    b = conv2d_block(p4, filters=1024)

    # Decoder blocks
    d1 = decoder_block(b, s4, filters=512)
    d2 = decoder_block(d1, s3, filters=256)
    d3 = decoder_block(d2, s2, filters=128)
    d4 = decoder_block(d3, s1, filters=64)

    # Output layer
    outputs = layers.Conv2D(num_classes, 1, padding='same', activation='sigmoid')(d4) #sigmoid for binary

    model = tf.keras.Model(inputs, outputs)
    return model

input_shape = (256, 256, 3) # match to your image_size
num_classes = 1  # binary segmentation, if multi-class, change this value
model = build_unet(input_shape, num_classes)
model.summary()
```

This code presents a basic U-Net implementation. Each encoder layer downsamples the input using max pooling, while the decoder upsamples through transposed convolutions. The crucial 'skip' connections concatenate corresponding layers in the encoder and decoder to preserve spatial details lost during downsampling.  The final convolutional layer outputs a probability map, representing our segmentation mask. Here I used `sigmoid` activation for binary classification of building footprints (0 or 1 for each pixel). The use of `batch normalization` layers stabilizes training.

**4. Training and Evaluation**

The training process in TensorFlow 2.8 is standard, but some key considerations should be implemented:

```python
optimizer = tf.keras.optimizers.Adam()
loss_fn = tf.keras.losses.BinaryCrossentropy() #Change to CategoricalCrossentropy if multi-class
metrics = ['accuracy']
model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)
epochs = 10
history = model.fit(dataset, epochs=epochs)
```
This snippet defines the optimizer, loss function and metrics. I opted for binary cross-entropy given my binary mask outputs. When utilizing multi-class outputs categorical cross-entropy needs to be used and adjusted to number of classes in model building.

I found that accuracy isn't sufficient in highly imbalanced datasets. Metrics like the Intersection over Union (IoU), also known as the Jaccard index, and Dice Coefficient are far more informative for evaluation. TensorFlow doesn’t directly provide IoU, but it can be implemented as such.

```python
def iou_metric(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.5, tf.float32) # threshold the predicted probability
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    iou = intersection / (union + tf.keras.backend.epsilon())
    return iou
model.compile(optimizer=optimizer, loss=loss_fn, metrics=[iou_metric, 'accuracy'])
history = model.fit(dataset, epochs=epochs)
```

Here I added IoU to the list of metrics used during the training and evaluation. It is an indicator of model quality with respect to the given ground truth in the masks. The use of tensorboard is also a good option for visually tracking progress of these metrics during training.

**Resource Recommendations:**

*   **TensorFlow Documentation:** The official TensorFlow website provides comprehensive information on the `tf.data` API, model building, and training procedures.
*   **Relevant research papers:** I highly recommend exploring recent publications on image segmentation architectures, especially those focused on specific tasks aligned to your use case.
*   **Online tutorials:** There are various online resources dedicated to deep learning with TensorFlow that illustrate best practices. While many focus on classification tasks, you can adapt the concepts to segmentation. The key is to emphasize proper data loading, custom loss functions, and evaluation metrics that are suitable for segmentation, particularly metrics such as IoU and Dice coefficient.

Through this project, I learned that successful image segmentation with custom datasets demands a comprehensive understanding of data, a meticulously crafted data pipeline, and a keen eye for selecting and evaluating the model. The code samples here, while simplified, provide a foundation for implementing an end-to-end segmentation solution using TensorFlow 2.8.
