---
title: "How do I fix the AttributeError: 'Functional' object has no attribute 'predict_segmentation' when loading a Keras TensorFlow model?"
date: "2025-01-30"
id: "how-do-i-fix-the-attributeerror-functional-object"
---
The error "AttributeError: 'Functional' object has no attribute 'predict_segmentation'" arises typically when attempting to utilize a method, `predict_segmentation`, that does not exist on the `tf.keras.Model` object that has been loaded. This often occurs after saving and reloading models trained using custom architectures or those that rely on specific, higher-level abstraction methods. It's crucial to understand that saving a Keras model, especially one employing subclassed models or custom layers, might not preserve the original higher-level APIs you might have defined for it. The `predict_segmentation` method likely existed during training, possibly as a method you'd defined within your custom class, but was not serialized directly with the model upon saving using standard `model.save()` functions.

Specifically, when a Keras model is saved using `tf.keras.models.save_model` or its equivalent, what is primarily serialized is the computational graph – the connections between layers and their weights. Functionalities not explicitly part of the model’s architecture as recognized by Keras, like custom methods or specific training loops, are not automatically preserved. When you load the model back via `tf.keras.models.load_model`, you're instantiating an object of type `tf.keras.models.Model` or `tf.keras.models.Sequential`, which are general-purpose Keras model classes. If you haven't explicitly incorporated that method into the model using the appropriate Keras mechanisms, it will not be available after loading.

I encountered this issue frequently when experimenting with medical image segmentation models. I’d create a custom subclass of `tf.keras.Model`, add a `predict_segmentation` method to handle preprocessing, the actual prediction, and post-processing steps (like thresholding and connected component analysis), and then save the model. Upon loading it, the `predict_segmentation` method was nowhere to be found, leaving me with an `AttributeError`. This happened, for instance, when switching from a training environment to a separate prediction environment where the model object was not re-instantiated from scratch using the original class definition.

Here's how I’ve addressed the issue, broadly categorizing the solutions into three key approaches, each with a related code example to illustrate the concept.

**1. Re-define the Custom Model Class and Load Weights**

The most straightforward solution, especially if you have custom methods or layers, is to explicitly re-instantiate your custom model class using its original definition and load the saved weights into that instantiated object. This restores both the structure of the model and the custom logic encapsulated within the class.

```python
import tensorflow as tf

# Assume this was the original custom model class
class MySegmentationModel(tf.keras.Model):
    def __init__(self, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')
        self.conv2 = tf.keras.layers.Conv2D(num_classes, 1, activation='sigmoid', padding='same') # example output layer
        # additional layers here

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        return x

    def predict_segmentation(self, image): # Method with business logic
        # Preprocessing image
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        # Prediction
        prediction = self(tf.expand_dims(image, axis=0))
        # Post-processing
        segmentation = tf.squeeze(tf.cast(prediction > 0.5, tf.int32)) # Simple threshold
        return segmentation


# Assume model has been trained and saved using:
# model = MySegmentationModel(num_classes=4)
# ...
# model.save('my_saved_model')


# Correct usage: Loading a saved model using its custom class
# First recreate your custom model:
loaded_model = MySegmentationModel(num_classes=4) # must match the trained model definition
# Load the saved weights into it
loaded_model.load_weights('my_saved_model')

# Now the predict_segmentation method works
test_image = tf.random.normal(shape=(128, 128, 3))
segmentation_result = loaded_model.predict_segmentation(test_image)

print(f"Shape of segmentation prediction:{segmentation_result.shape}") # (128, 128) as expected
```

In this example, the `MySegmentationModel` class contains the `predict_segmentation` method. To use it after loading a model saved from this class, you must instantiate an object of `MySegmentationModel` and load the saved weights. The `load_weights` method is used because the model architecture has been re-established by instantiation. This preserves all original functionalities. This avoids re-compiling the entire model and retraining.

**2. Incorporate Prediction Logic as a Sub-model**

If you are averse to re-instantiating model classes directly, you could encapsulate the logic of `predict_segmentation` into a separate Keras `Model` (subclassing `tf.keras.Model` or a `tf.keras.Sequential` model) and use that for your prediction. The main model can then be a part of this new sub-model. This sub-model and its architecture will be fully serialized by keras.

```python
import tensorflow as tf

# Assuming your core model definition (as saved)
class SegmentationCoreModel(tf.keras.Model):
    def __init__(self, num_classes, **kwargs):
      super().__init__(**kwargs)
      self.conv1 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')
      self.conv2 = tf.keras.layers.Conv2D(num_classes, 1, activation='sigmoid', padding='same')


    def call(self, inputs):
      x = self.conv1(inputs)
      x = self.conv2(x)
      return x


# New wrapper model incorporating prediction functionality
class SegmentationPredictionModel(tf.keras.Model):
  def __init__(self, num_classes, **kwargs):
    super().__init__(**kwargs)
    self.core_model = SegmentationCoreModel(num_classes) # Instance of the underlying core model

  def call(self, inputs):
    return self.core_model(inputs)

  def predict_segmentation(self, image):
      # Preprocessing image
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)
      # Prediction
      prediction = self(tf.expand_dims(image, axis=0))
      # Post-processing
      segmentation = tf.squeeze(tf.cast(prediction > 0.5, tf.int32))
      return segmentation

# Assume core_model has been trained
core_model = SegmentationCoreModel(num_classes=4)
# core_model.save('my_saved_core_model')

# Load the core model and incorporate in the wrapper model.
loaded_core_model = tf.keras.models.load_model('my_saved_core_model')

# Create an instance of the wrapper model, assigning the loaded model
prediction_model = SegmentationPredictionModel(num_classes=4)
prediction_model.core_model = loaded_core_model # Replace core model with loaded core model
# Call predict segmentation as usual
test_image = tf.random.normal(shape=(128, 128, 3))
segmentation_result = prediction_model.predict_segmentation(test_image)

print(f"Shape of segmentation prediction:{segmentation_result.shape}") # (128, 128) as expected
```

Here, the core model’s architecture is kept separate from higher-level logic.  The `SegmentationPredictionModel` encapsulates both the `SegmentationCoreModel` and `predict_segmentation`. In this way, all necessary logic is part of the saved structure. This decouples the core architecture from the additional preprocessing and post-processing needed.

**3. Use a Function for Prediction, Separating Logic from the Model**

This approach involves defining the prediction-related functionalities (preprocessing, actual prediction, post-processing) as a standalone function. When making predictions, this function is called, passing the loaded model and the input image as arguments. This is useful if custom methods are not desired in model classes.

```python
import tensorflow as tf

# Assumed model (same as original save target)
class SegmentationCoreModel(tf.keras.Model):
  def __init__(self, num_classes, **kwargs):
    super().__init__(**kwargs)
    self.conv1 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')
    self.conv2 = tf.keras.layers.Conv2D(num_classes, 1, activation='sigmoid', padding='same')


  def call(self, inputs):
    x = self.conv1(inputs)
    x = self.conv2(x)
    return x

# Prediction logic as a standalone function
def predict_segmentation_function(model, image):
  # Preprocessing
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  # Prediction
  prediction = model(tf.expand_dims(image, axis=0))
  # Post-processing
  segmentation = tf.squeeze(tf.cast(prediction > 0.5, tf.int32))
  return segmentation

# Model has been trained and saved (as before)
model = SegmentationCoreModel(num_classes=4)
# model.save('my_saved_model')

# Load model
loaded_model = tf.keras.models.load_model('my_saved_model')


# Make a prediction by passing it to the function
test_image = tf.random.normal(shape=(128, 128, 3))
segmentation_result = predict_segmentation_function(loaded_model, test_image)

print(f"Shape of segmentation prediction:{segmentation_result.shape}") # (128, 128) as expected
```

This method decouples the model logic from the prediction pipeline entirely. The `predict_segmentation_function` is now a separate utility that uses the loaded model for inference. It also keeps model architectures clean and focused on the training step only.

**Recommendations:**

When dealing with custom models and the need to save/load them, I recommend a layered approach: first, fully understand what gets saved by `tf.keras.models.save_model`. Second, carefully think about where to best house additional logic: as part of the custom model class, inside a separate prediction model, or as a standalone function. For model research and experimentation, the first solution works well – re-instantiating the custom model and loading its weights directly. However, for production or deployment, wrapping the model or using a separate prediction function leads to greater modularity. Refer to the official TensorFlow documentation for model saving, loading and subclassing, as well as guides on modularizing Keras models. Additionally, consider exploring design patterns related to model deployment and inferencing, such as separating model training from model serving.
