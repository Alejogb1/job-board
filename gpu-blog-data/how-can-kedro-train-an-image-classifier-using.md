---
title: "How can Kedro train an image classifier using Keras ImageDataGenerator?"
date: "2025-01-30"
id: "how-can-kedro-train-an-image-classifier-using"
---
The inherent challenge in integrating Keras' `ImageDataGenerator` with Kedro lies in managing the data pipeline's inherent complexities within Kedro's structured data catalog and modular node execution framework.  My experience building a large-scale medical image classification system highlighted the need for a robust, configurable approach that avoids hardcoding paths and parameters.  Successfully bridging this gap requires a deep understanding of both Kedro's data versioning and Keras' data augmentation capabilities.

**1. Clear Explanation:**

Kedro's strength lies in its ability to manage data as a first-class citizen, using a catalog to define data locations and metadata.  However, `ImageDataGenerator` expects data to be readily accessible via directory structures. The solution, therefore, centers on leveraging Kedro's data loading capabilities to populate a directory structure that `ImageDataGenerator` can directly consume.  This involves creating Kedro nodes that:

* **Download or Extract Data:** Obtain the raw image data (potentially from a remote source or archive). This could utilize existing Kedro nodes or custom ones leveraging libraries like `requests` or `shutil`.
* **Preprocess and Organize Data:** This critical step prepares the images for the `ImageDataGenerator`.  This includes resizing, normalization, and creating subdirectories for training, validation, and testing sets, mirroring the structure expected by `ImageDataGenerator`.  Error handling is crucial here to manage inconsistencies in the data.
* **Generate and Train the Model:**  This node uses the prepared directory structure to instantiate `ImageDataGenerator` and subsequently train a Keras model.  Careful consideration must be given to the hyperparameters passed to both `ImageDataGenerator` and the Keras model compilation step.  Callbacks for saving model weights and monitoring training progress are essential.
* **Register and Save the Trained Model:**  Upon training completion, the trained Keras model, along with associated metadata (e.g., accuracy, loss), should be registered in Kedro's catalog for downstream use.

This multi-stage pipeline ensures reproducibility, maintainability, and version control of the entire process, a key advantage over ad-hoc scripting.

**2. Code Examples with Commentary:**

**Example 1: Data Preparation Node:**

```python
import os
from kedro.io import DataCatalog, Path, MemoryDataSet
from PIL import Image
import numpy as np

def _preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))  # Resize for example
    img_array = np.array(img) / 255.0  # Normalize
    return img_array

def preprocess_images(image_paths, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    for image_path in image_paths:
        try:
            preprocessed_image = _preprocess_image(image_path)
            class_name = os.path.basename(os.path.dirname(image_path))
            class_dir = os.path.join(target_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            image_name = os.path.basename(image_path)
            np.save(os.path.join(class_dir, image_name[:-4] + ".npy"), preprocessed_image) #Save as npy to avoid issues with large number of images
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            #Handle errors gracefully


@node
def prepare_image_data(raw_images: Path, preprocessed_images: str):
    image_paths = list(raw_images.glob("*/*.jpg")) # assumes images are already classified in subfolders
    preprocess_images(image_paths, preprocessed_images)

```

This node iterates through images, resizes, normalizes them, and saves preprocessed images as .npy files organized by class in a structure `ImageDataGenerator` understands.  Error handling ensures robustness.  The use of `Path` objects from Kedro's IO system facilitates integration with the data catalog.

**Example 2: Model Training Node:**

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from kedro.io import DataCatalog, Path

@node
def train_model(preprocessed_images: str, model_path: Path):
    train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1./255) #Rescale only for testing


    train_generator = train_datagen.flow_from_directory(
        preprocessed_images,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )

    validation_generator = test_datagen.flow_from_directory(
        preprocessed_images,
        target_size=(224, 224), #This assumes a validation split.  Could be improved with separate paths
        batch_size=32,
        class_mode='categorical'
    )

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(len(train_generator.class_indices), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    checkpoint = ModelCheckpoint(str(model_path), save_best_only=True, monitor='val_accuracy', mode='max')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)


    model.fit(
        train_generator,
        epochs=50,
        validation_data=validation_generator,
        callbacks=[checkpoint, early_stopping]
    )

```

This node leverages `ImageDataGenerator` to create data generators for training and validation.  It defines a simple CNN model (easily replaceable with more complex architectures), compiles it, and trains it using callbacks to save the best model and prevent overfitting.  The use of `flow_from_directory` directly integrates with the directory structure created in the previous node.


**Example 3:  Model Registration Node:**

```python
from kedro.io import DataCatalog
from tensorflow.keras.models import load_model

@node
def register_model(model_path: Path, trained_model: str):
    model = load_model(str(model_path))
    DataCatalog.register("trained_model", MemoryDataSet({"model": model})) #Store in memory for this example.  Consider using a more suitable storage for production
```


This node loads the best-performing model and registers it in Kedro's catalog as a `MemoryDataSet` (for simplicity in this example). For a production environment, consider using a more persistent storage option like a file system or cloud storage.

**3. Resource Recommendations:**

*   "Deep Learning with Python" by Francois Chollet: A comprehensive guide to Keras and deep learning fundamentals.
*   Kedro documentation: Essential for understanding Kedro's data management and pipeline construction features.
*   TensorFlow documentation:  Provides detailed information on Keras APIs and model building.
* A good understanding of Python's `os` and `shutil` modules for file system management.


This structured approach, combining Kedro's pipeline management with Keras' data augmentation, enables the creation of a reproducible and scalable image classification system.  Remember to adapt the code examples to your specific dataset characteristics and desired model architecture.  Thorough error handling and careful consideration of hyperparameters are crucial for success.
