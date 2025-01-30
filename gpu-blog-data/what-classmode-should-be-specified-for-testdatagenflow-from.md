---
title: "What class_mode should be specified for `test_datagen.flow` from a directory?"
date: "2025-01-30"
id: "what-classmode-should-be-specified-for-testdatagenflow-from"
---
The `class_mode` parameter in `test_datagen.flow_from_directory` directly impacts how the generator yields data during testing.  Crucially, its selection depends entirely on the intended use of the generated data and the underlying model architecture.  In my experience optimizing image classification models for high-throughput industrial applications, overlooking this nuance has frequently led to incorrect evaluation metrics and suboptimal model deployment.  Incorrect `class_mode` specification can result in errors ranging from silent failures to profoundly inaccurate performance evaluations, undermining the reliability of the entire testing process.  Therefore, a careful selection based on a clear understanding of your model's requirements is paramount.

My work involved processing millions of images daily for automated defect detection in printed circuit boards. This necessitated rigorous testing procedures, and understanding the implications of `class_mode` proved essential for consistent and accurate evaluation of model performance. I've encountered all three common scenarios, and will detail each with appropriate code examples.

**1. `class_mode = None`:** This setting should be used when your model predicts a single value per image, and you're not interested in comparing against a ground truth class label.  This is common in applications like image regression where you predict a continuous variable, such as object size or a material's physical property based on its image.  The generator then yields only the image data and ignores any directory structure representing classes.

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
        'test_directory',
        target_size=(150, 150),
        batch_size=32,
        class_mode=None,  # Crucial: No class labels are used
        shuffle=False) # Essential for consistent evaluation order

# Example prediction loop
predictions = model.predict(test_generator, steps=len(test_generator), verbose=1)

# Predictions will be a NumPy array of shape (num_samples, prediction_dimension).
#  No class labels are associated with these predictions.
```

The `shuffle=False` flag is critical here.  Shuffling would disrupt the correspondence between predictions and the original image order, making any downstream analysis impossible.  Note that the `'test_directory'` structure is irrelevant in this scenario; it only serves to organize the image files. The model's output is directly interpreted as the predicted value, without any classification context.

**2. `class_mode = 'categorical'`:** This option is used for multi-class classification problems where each image belongs to exactly one class.  The generator yields image data along with one-hot encoded class labels.  This is the most common scenario in image classification tasks.  The directory structure must reflect the class labels, with each subdirectory representing a distinct class.

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical # For one-hot encoding if needed

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
        'test_directory',
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical',
        shuffle=False)

# Example evaluation using model.evaluate
loss, accuracy = model.evaluate(test_generator, steps=len(test_generator), verbose=1)

# The model should output probabilities for each class, enabling accuracy calculations.
```

In this case, the directory structure defines the class labels.  For instance, `test_directory/class_a/image1.jpg` implies that `image1.jpg` belongs to class 'a'.  The `shuffle=False` is again essential for reliable evaluation.  The model's output is then interpreted as class probabilities, which are used to compute metrics like accuracy or F1-score against the corresponding one-hot encoded labels.

**3. `class_mode = 'binary'`:**  This is employed when dealing with binary classification problems, where each image belongs to one of two classes.  Similar to 'categorical', the directory structure defines the classes, but the output is a single probability value per image, representing the probability of belonging to the positive class.

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
        'test_directory',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary',
        shuffle=False)

# Example evaluation using model.evaluate
loss, accuracy = model.evaluate(test_generator, steps=len(test_generator), verbose=1)

# The model outputs a single probability value for each image indicating the likelihood of belonging to the positive class.
```

Here, a binary classification model is assumed. The directory structure should contain two subdirectories representing the two classes.  The model's output is a single probability value, interpreted as the likelihood of the image belonging to the positive class (typically the first subdirectory alphabetically).  Again, `shuffle=False` maintains order for accurate evaluation.

**Resource Recommendations:**

The official TensorFlow documentation on `ImageDataGenerator`.  A comprehensive text on deep learning covering model evaluation and metrics. A practical guide on image classification using Keras.  Understanding these resources will significantly enhance your ability to effectively utilize the `ImageDataGenerator` and accurately evaluate model performance.  The key is to understand the relationship between your model's output, the `class_mode` setting, and the directory structure of your test data.  Failing to establish this connection will result in unreliable and potentially misleading results.  Thorough testing using the correct `class_mode` is crucial for deploying robust and dependable machine learning models.
