---
title: "How to train a CNN model using images organized in sub-folders (train, validation, test)?"
date: "2025-01-30"
id: "how-to-train-a-cnn-model-using-images"
---
The core challenge in training a Convolutional Neural Network (CNN) on image data organized into subfolders lies in efficiently loading and preprocessing these images while maintaining a clear separation between training, validation, and testing sets.  My experience building image classification models for medical diagnostics highlighted the critical need for robust data handling, particularly given the often imbalanced nature of medical image datasets.  Neglecting this aspect can lead to overfitting, inaccurate performance estimations, and ultimately, flawed model deployment.  This response details how I’ve addressed this challenge, using Python and popular libraries.

**1.  Clear Explanation:**

The process hinges on three key steps: data loading, data augmentation (optional but highly recommended), and model training.  Firstly,  we need to traverse the directory structure, loading images from each subfolder ('train', 'validation', 'test'). Each image is typically converted to a NumPy array, normalized, and potentially resized to a standard input size for the CNN.  Data augmentation techniques, such as random rotations, flips, and crops, artificially expand the training set, mitigating overfitting and improving generalization.  Finally, we use a suitable deep learning framework, such as TensorFlow/Keras or PyTorch, to define, compile, and train the CNN model using the preprocessed data.  Crucially, the validation set is used to monitor performance during training, preventing overfitting and guiding hyperparameter tuning.  The test set remains untouched until the final model evaluation.

**2. Code Examples with Commentary:**

**Example 1:  Using Keras' `ImageDataGenerator` (TensorFlow/Keras):**

```python
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
        'data/validation',
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
        'data/test',
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical',
        shuffle=False)


model = tf.keras.models.Sequential([
    # ... your CNN model architecture here ...
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_generator,
          steps_per_epoch=len(train_generator),
          epochs=10,
          validation_data=validation_generator,
          validation_steps=len(validation_generator))


loss, accuracy = model.evaluate(test_generator, steps=len(test_generator))
print('Test accuracy:', accuracy)

```

This example leverages Keras' `ImageDataGenerator` for efficient data loading and augmentation.  `flow_from_directory` automatically handles image loading and label encoding from subfolders.  Note the use of `categorical_crossentropy` loss, suitable for multi-class classification.  The `shuffle=False` argument for the test generator is crucial for consistent evaluation.  I've incorporated common augmentation techniques—adjust these based on your specific dataset and model.  Remember to replace the placeholder comment with your chosen CNN architecture.


**Example 2: Manual Data Loading with PyTorch:**

```python
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.ImageFolder('data/train', transform=transform)
validation_dataset = datasets.ImageFolder('data/validation', transform=transform)
test_dataset = datasets.ImageFolder('data/test', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ... define your PyTorch CNN model ...
model = YourCNNModel() # Replace with your model definition

# ... training loop using train_loader and validation_loader ...

# ... evaluation loop using test_loader ...
```

This PyTorch example provides more control over the data loading process.  `datasets.ImageFolder` conveniently loads images from directories.  `transforms.Compose` chains multiple image transformations, including resizing and normalization.  DataLoaders efficiently handle batching and shuffling.  This approach offers flexibility but requires more manual data handling compared to Keras' `ImageDataGenerator`. Remember to replace `YourCNNModel()` with your actual PyTorch model definition and write the training and evaluation loops.



**Example 3:  Handling Imbalanced Datasets (both Keras and PyTorch adaptable):**

If your datasets are imbalanced (e.g., significantly more images in one class than others), consider class weighting during training.  This is crucial for preventing the model from being biased towards the majority class.

```python
# Calculate class weights (example for Keras)
import numpy as np
from sklearn.utils import class_weight

train_labels = np.array(train_generator.classes)
class_weights = class_weight.compute_sample_weight('balanced', train_labels)

# Train the model with class weights
model.fit(train_generator,
          steps_per_epoch=len(train_generator),
          epochs=10,
          validation_data=validation_generator,
          validation_steps=len(validation_generator),
          class_weight=class_weights)

# For PyTorch, calculate class weights similarly, and use them in your training loop like this:
criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights))
```

This code snippet demonstrates how to compute class weights using scikit-learn's `compute_sample_weight`.  These weights are then passed to the `fit` method in Keras or integrated into the loss function in PyTorch to address class imbalance.


**3. Resource Recommendations:**

For further learning, I recommend exploring detailed tutorials on image classification using Keras and PyTorch.  Consult textbooks on deep learning and computer vision for a theoretical foundation.  Explore documentation for libraries like scikit-learn for data preprocessing and analysis techniques beyond those shown here.  The official documentation of TensorFlow and PyTorch provides comprehensive resources on model building, training, and evaluation.  Finally, examining research papers focusing on CNN architectures and training strategies will enhance your understanding and inform your choices.  Careful consideration of these resources, alongside rigorous experimentation, is essential for building effective CNN models.
