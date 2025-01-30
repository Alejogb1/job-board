---
title: "How can a CNN model be built in Google Earth Engine for multi-class image classification?"
date: "2025-01-30"
id: "how-can-a-cnn-model-be-built-in"
---
Building a convolutional neural network (CNN) within Google Earth Engine (GEE) for multi-class image classification requires a strategic combination of its data access and processing capabilities with TensorFlow’s model building tools. GEE, while primarily a geospatial analysis platform, enables the preparation of large-scale image datasets suitable for machine learning, while TensorFlow provides the necessary framework for CNN architecture definition, training, and evaluation. This process, though feasible, presents particular challenges in data handling and model deployment compared to a typical Python environment.

The core challenge lies in translating Earth Engine's image representations into a format compatible with TensorFlow. GEE images are fundamentally distributed data structures, stored as tiles across Google’s infrastructure. Conversely, TensorFlow expects tensors, typically residing in memory as NumPy arrays. The bridge between these two representations is crucial. This requires careful planning for data extraction, batch processing, and data type conversions within the Earth Engine environment.

The standard approach involves several distinct steps: first, select and preprocess the imagery; second, generate training and testing datasets; third, define the CNN model in TensorFlow; fourth, train the model using the extracted data; and finally, export the trained model for inference within Earth Engine. Let’s detail each stage.

**1. Data Selection and Preprocessing:**

GEE’s robust filtering and compositing functionalities must be leveraged. This involves selecting a dataset (e.g., Sentinel-2, Landsat) based on the application’s specific needs, applying cloud masking, and potentially reducing the temporal dimension via median compositing or time series analysis. The dataset should ideally contain the features required for separating your target classes (e.g., different land cover types).

Pre-processing is paramount. This includes steps like rescaling the bands to a normalized range (e.g., 0-1), and potentially applying transformations like vegetation indices, ensuring that the input data is within a suitable numerical range for neural network training. In my experience, not performing adequate preprocessing can lead to unstable training.

**2. Generating Training and Testing Datasets:**

This step is critical and often the most demanding from a computational perspective. Feature collections defining the regions of interest (ROIs) representing each class must be created. These ROIs should accurately and comprehensively represent the spatial variations within each class. These vector polygons are then used to sample the image data. I generally recommend a stratified sampling approach, meaning that within the ROIs, pixels are chosen randomly, and equal numbers are selected for each class.

The samples must be exported from GEE as TensorFlow datasets (TFRecords format). Each sample in the TFRecord should contain not just the pixel data but also the associated label corresponding to the class it belongs to. Care must be taken here to handle the data types, ensuring they align between GEE and TensorFlow (e.g., 32-bit floats). It's prudent to carefully control the number of exported samples; exporting too many can lead to storage issues or excessive processing time.

**3. Defining the CNN Model:**

Within a TensorFlow environment external to GEE (typically in a Jupyter notebook or similar), I construct the CNN architecture using Keras, TensorFlow's high-level API. The architecture depends on the classification task. Common architectures involve convolutional layers, pooling layers, and fully connected layers. The final layer will have a softmax activation to output the probability of each class. It is crucial to select the appropriate number of filters, kernel sizes, and activation functions for the problem at hand. Experimentation is usually necessary here to determine the optimal structure.

**4. Training the Model:**

The exported TFRecords datasets are loaded using the TensorFlow `tf.data` API for model training. I incorporate best practices such as data augmentation (e.g., rotations, flips) to increase the model's generalization capability. The model is trained using an optimization algorithm (e.g., Adam) and an appropriate loss function (e.g., categorical cross-entropy) for multi-class classification. During training, I always monitor metrics such as accuracy, precision, recall, and loss to assess the model’s performance. Hyperparameter tuning is often necessary to achieve satisfactory results.

**5. Exporting and Using the Model in GEE:**

Once trained, the TensorFlow model is exported as a SavedModel. This is a standard format for storing a TensorFlow model's architecture and weights, which can be loaded into a Google Earth Engine asset. In GEE, the exported model is wrapped in a `ee.Model` object and applied to GEE images for predictions. This process involves converting GEE image patches to tensors, performing inference, and then restructuring the resulting predictions back into a GEE image. This last step needs careful design to maintain the geographical reference.

Here are three illustrative code examples. Note that this code is conceptual; complete scripts would require specific data and configurations.

**Example 1: GEE Data Export to TFRecords:**

```python
# Google Earth Engine Snippet (Conceptual)

import ee
import ee.mapclient

ee.Initialize()

# 1. Feature Collection for classes
feature_collections = {
    'forest': ee.FeatureCollection(...),  # Add forest ROIs
    'urban': ee.FeatureCollection(...),  # Add urban ROIs
    'water': ee.FeatureCollection(...)    # Add water ROIs
}

# 2. Satellite Data
image = ee.ImageCollection("LANDSAT/LC08/C01/T1_SR").filterDate('2022-01-01', '2022-12-31').median()

# 3. Sample generation
def sample_image(feature, label):
    sample = image.sample(feature, region=feature.geometry(), scale=30, numPixels=128)
    return sample.set('label', label)

samples = []
for label, fc in feature_collections.items():
    samples.append(fc.map(lambda feature: sample_image(feature, label)))
samples = ee.FeatureCollection(samples).flatten()

# 4. Export to TFRecords
task = ee.batch.Export.table.toCloudStorage(
    collection=samples,
    description='training_data',
    bucket='my-gee-bucket', # Replace with your bucket
    fileNamePrefix='training_samples',
    fileFormat='TFRecord'
)
task.start()
```

**Example 2: TensorFlow Model Definition (Python - Keras):**

```python
# Python Code (TensorFlow)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_cnn_model(input_shape, num_classes):
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Assuming input_shape (e.g., 128x128x7 for Landsat) and num_classes (3 in this example) are defined
input_shape = (128, 128, 7)
num_classes = 3
cnn_model = create_cnn_model(input_shape, num_classes)

cnn_model.summary() # Print architecture details
# ... Further model training and evaluation
```

**Example 3: GEE Inference with Trained Model:**

```python
# Google Earth Engine Snippet (Conceptual)

import ee

ee.Initialize()

# 1. Load the SavedModel
model_asset_id = 'projects/my-project/assets/my_cnn_model'  # Replace
cnn_model = ee.Model.fromSavedModel(model_asset_id)

# 2. Input image
image = ee.ImageCollection("LANDSAT/LC08/C01/T1_SR").filterDate('2023-01-01', '2023-12-31').median()

# 3. Patch-wise classification
patches = image.reproject('EPSG:4326', scale=30).sampleRegions(
    collection=ee.FeatureCollection([ee.Feature(ee.Geometry.Rectangle(-180, -90, 180, 90))]),
    properties=['.geo'],
    scale=30,
    tileScale=16,
    geometries=True,
    )
predictions = cnn_model.predictImage(patches)

# 4. Convert predictions to image with class labels
classification = predictions.argmax().toInt()
# ... Displaying and analyzing the classification map
```

**Resource Recommendations:**

For a comprehensive understanding of machine learning and CNNs, I recommend researching university-level online courses, such as those offered by Stanford or MIT. For TensorFlow, the official documentation and tutorials from TensorFlow.org are invaluable. For GEE-specific applications, reviewing the official Earth Engine documentation and tutorials is necessary. Furthermore, publications in relevant journals focusing on remote sensing and machine learning, frequently contain the latest methodologies. Practical experience through experimentation is also a critical component of developing robust and accurate models. Lastly, studying existing examples and tutorials within the Earth Engine developer community can provide concrete examples. A strong theoretical foundation alongside practical experimentation ultimately leads to successful implementation.
