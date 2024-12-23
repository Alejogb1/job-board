---
title: "How can Teachable Machine accuracy be improved?"
date: "2024-12-23"
id: "how-can-teachable-machine-accuracy-be-improved"
---

Alright, let's talk about fine-tuning Teachable Machine outputs. I’ve spent a good portion of my career working with similar model-building platforms, and while they provide a fantastically accessible entry point to machine learning, achieving truly high accuracy often demands more than just the default settings. I’ve seen firsthand how frustrating it can be when models plateau, and getting past that hump requires a more nuanced approach.

Fundamentally, improving Teachable Machine accuracy boils down to addressing core machine learning principles. It's not about magic tricks but about improving data quality, model architecture, and the training process itself. Let's break this down into actionable steps, drawing from my experiences with similar frameworks.

First, the bedrock: your *data*. This is where most improvements begin, and often where issues hide. The single biggest mistake I see is insufficient data variety. For example, imagine you're training a model to identify different types of fruit. If your training set only includes apples photographed in ideal lighting conditions, it will likely struggle with an apple shot from a different angle, under shadows, or even just a slightly different shade of red. In one project, we tried to build a classifier for detecting defects in manufactured parts using images, and initially, our model was terrible. We later realized that almost all the "defective" images had a very similar background or lighting condition compared to our "good" images, creating correlation, not causation. The model was picking up on background variations rather than defects. We had to actively vary the lighting, angle, and background in our training data. Data augmentation, such as rotations, flips, and color adjustments, is an easy yet potent strategy here. Teachable Machine offers a basic version of this but often you'll need to manually prepare datasets using image processing tools. Think of it this way: the more representative your training data is of the real world, the better your model will generalize.

Secondly, consider *feature extraction and the underlying architecture*. While Teachable Machine abstracts this, understanding its constraints is valuable. Typically, it employs a pre-trained convolutional neural network (CNN) as a base. These CNNs have been trained on massive datasets (like ImageNet) and are designed to recognize a wide variety of features in images. However, these general features may not be optimal for your specific task. In one project involving the classification of medical images, I discovered that while the overall model was 'working', performance was not ideal. Because the pre-trained model had focused heavily on general image features, we found that the model didn't leverage the unique details of the medical images as much as it could. To tackle this, we carefully fine-tuned only some of the later layers of the pre-trained network with our specialized dataset, allowing the model to focus on patterns and features specific to medical images without discarding pre-existing knowledge.

Finally, let's address the *training process*. Teachable Machine automates much of this, but understanding a few concepts can lead to improvements. Hyperparameter tuning is crucial. This means adjusting settings like learning rate and batch size. The default values in Teachable Machine might be good starting points, but they are rarely optimal. It's essential to test variations to see what works best for *your* data. Also, watch out for overfitting – when a model performs exceptionally well on training data but poorly on unseen data. Cross-validation is vital here. Split your data into training, validation and test sets to evaluate your model’s performance during training and assess generalisation ability. Teachable Machine often does this automatically to a degree, but for more in-depth evaluation, you'll need to manage these splits directly.

Let me illustrate these points with some code snippets, using Python with a few hypothetical functions. I’m intentionally keeping this at a high level to highlight the concepts rather than get bogged down with specific library calls.

**Snippet 1: Data Augmentation (using a hypothetical image library):**

```python
def augment_image(image, augmentation_type):
    """Applies various image augmentations.
        Args:
            image (Image): The image to augment
            augmentation_type (str): The augmentation operation.
        Returns:
            Image: The augmented image.
    """
    if augmentation_type == "rotate":
        return image.rotate(angle=random.randint(-30, 30))
    elif augmentation_type == "flip_horizontal":
        return image.flip_horizontal()
    elif augmentation_type == "adjust_brightness":
        return image.adjust_brightness(factor=random.uniform(0.7, 1.3))
    else:
        return image

def augment_dataset(images, augmentation_types, num_augmentations_per_image):
    """Augments a dataset of images.
        Args:
            images (list): A list of Image objects.
            augmentation_types (list): A list of augmentation operations.
            num_augmentations_per_image (int): Number of times to augment each image.
        Returns:
            list: A list of augmented Image objects.
    """
    augmented_images = []
    for image in images:
        for i in range(num_augmentations_per_image):
            augmented_type = random.choice(augmentation_types)
            augmented_images.append(augment_image(image, augmented_type))
    return augmented_images

# Hypothetical usage:
images = load_images_from_directory("original_images")
augmentation_types = ["rotate", "flip_horizontal", "adjust_brightness"]
augmented_images = augment_dataset(images, augmentation_types, 3)
save_images_to_directory(augmented_images, "augmented_images")
```

This snippet illustrates how data augmentation could be implemented. We randomly pick from a set of image transformation techniques such as rotation and brightness adjustment to introduce more variability into the training data. Remember that a good variety of augmented data makes your model more robust to real-world variations in input images.

**Snippet 2: Hyperparameter Tuning (conceptual demonstration):**

```python
def train_model(model, train_data, validation_data, learning_rate, batch_size, num_epochs):
    """Trains a model with given hyperparameters.
    Args:
      model: The model to train
      train_data: Training data
      validation_data: Validation data
      learning_rate: Learning rate parameter
      batch_size: Batch size parameter
      num_epochs: Number of training epochs
    Returns:
        model: The trained model
    """

    optimizer =  Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_data, epochs=num_epochs, batch_size=batch_size, validation_data=validation_data)
    return model


def hyperparameter_search(model, train_data, validation_data, learning_rates, batch_sizes, num_epochs):
    """Searches for the best hyperparameters.
    Args:
      model: The model to train
      train_data: Training data
      validation_data: Validation data
      learning_rates: List of learning rate values to test.
      batch_sizes: List of batch size values to test.
      num_epochs: Number of epochs to train
      """
    best_accuracy = 0
    best_params = {}

    for lr in learning_rates:
        for bs in batch_sizes:
            trained_model = train_model(model, train_data, validation_data, lr, bs, num_epochs)
            _, accuracy = trained_model.evaluate(validation_data, verbose=0)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = {"learning_rate": lr, "batch_size": bs}
    return best_params

# Hypothetical Usage:
model = load_pre_trained_model()
train_data = load_train_data()
validation_data = load_validation_data()
learning_rates = [0.001, 0.0001]
batch_sizes = [32, 64]
num_epochs = 20
best_hyperparameters = hyperparameter_search(model, train_data, validation_data, learning_rates, batch_sizes, num_epochs)
print(f"Best hyperparameters: {best_hyperparameters}")
trained_model = train_model(model, train_data, validation_data,
                            best_hyperparameters["learning_rate"],
                            best_hyperparameters["batch_size"],
                            num_epochs)
```

Here, we're systematically exploring different combinations of learning rates and batch sizes and evaluating the model performance after training with different combinations, on a held-out validation set. In a real-world scenario you may be using grid search or bayesian optimization for hyperparameter selection. This highlights that optimizing learning rate, batch size, and the number of epochs is not a 'one-size fits all'.

**Snippet 3: Fine-tuning the base model (conceptual demonstration):**

```python
def fine_tune_model(model, train_data, validation_data, num_layers_to_unfreeze, learning_rate, num_epochs):
    """Fine-tunes the last layers of a pre-trained model.
    Args:
        model: The model to fine-tune
        train_data: Training data
        validation_data: Validation data
        num_layers_to_unfreeze: Number of layers to unfreeze for training.
        learning_rate: Learning rate parameter.
        num_epochs: Number of epochs to train
    Returns:
       model: The fine-tuned model.
    """
    for layer in model.layers[:-num_layers_to_unfreeze]:
        layer.trainable = False #freeze the layers
    optimizer = Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_data, epochs=num_epochs, validation_data=validation_data)
    return model

# Hypothetical Usage:
model = load_pre_trained_model()
train_data = load_train_data()
validation_data = load_validation_data()
num_layers_to_unfreeze = 4 #example
learning_rate = 0.0001 #example
num_epochs = 10
fine_tuned_model = fine_tune_model(model, train_data, validation_data, num_layers_to_unfreeze, learning_rate, num_epochs)
```
This illustrates the concept of fine-tuning. By selectively unfreezing and training only some layers of the base pre-trained model, we avoid catastrophic forgetting, and adapt the model to the specific details of our dataset.

To further deepen your understanding, I recommend exploring resources like "Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville for a comprehensive theoretical background. For practical application, "Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow" by Aurélien Géron provides excellent guidance, with Python examples. Additionally, research papers on topics such as data augmentation techniques, fine-tuning of pre-trained CNNs (such as the VGG or ResNet family), and Bayesian optimisation will provide a more in-depth approach.

In conclusion, improving Teachable Machine accuracy isn’t about a single switch, but about thoughtfully applying core machine learning principles. Focus on data quality and diversity, explore the potential for fine-tuning, and meticulously adjust hyperparameters. It’s a process of iteration and refinement that almost always yields improved results. And don’t be afraid to experiment; it’s often the slightly unorthodox approach that unlocks the best performance.
