---
title: "How can TensorFlow Hub be used in Azure Machine Learning?"
date: "2025-01-26"
id: "how-can-tensorflow-hub-be-used-in-azure-machine-learning"
---

Integrating TensorFlow Hub within Azure Machine Learning significantly accelerates model development and deployment by providing readily available, pre-trained model components. My experience working on several image classification and natural language processing projects has shown that leveraging these pre-trained modules can dramatically reduce training times and improve model performance, particularly when dealing with limited data. TensorFlow Hub essentially acts as a repository of pre-trained models that can be integrated into new models via a standardized interface. Azure Machine Learning provides the compute resources and pipeline orchestration capabilities to facilitate efficient use of these resources.

Let's break down the integration process with concrete examples. At the core, this process involves selecting an appropriate module from TensorFlow Hub, loading it within a TensorFlow model definition, and then training this model using Azure Machine Learning’s capabilities. The advantage here is that, instead of starting from scratch, one can initialize the early layers of a network with a powerful pre-trained representation. This approach is incredibly effective in transfer learning, a core component of my past projects, which involves adapting a model trained on one task to a new related task.

First, we need to consider the module types offered by TensorFlow Hub. Primarily, these fall into a few categories: *feature vector modules* that output high-level feature representations for input data, *image classification modules* that are pre-trained on large image datasets and can be fine-tuned for new tasks, and *text embedding modules* that provide vector representations for text that encapsulate semantic meaning. The choice of module depends heavily on the problem being addressed.

Here’s a simple example illustrating how to use a feature vector module for image classification within the context of Azure Machine Learning:

```python
import tensorflow as tf
import tensorflow_hub as hub
from azureml.core import Workspace
from azureml.core.compute import ComputeTarget
from azureml.core import Experiment
from azureml.train.tensorflow import TensorFlow
import os

# Azure Machine Learning Configuration (replace with your own values)
subscription_id = "your_subscription_id"
resource_group = "your_resource_group"
workspace_name = "your_workspace_name"

ws = Workspace(subscription_id, resource_group, workspace_name)
cluster_name = "your_compute_cluster_name" # replace with your compute cluster name or create one
compute_target = ComputeTarget(workspace=ws, name=cluster_name)
# define data location
ds = ws.get_default_datastore()

# Define experiment
experiment = Experiment(workspace=ws, name="tfhub_image_classification")

# Define training script
train_script_dir = './training_script' #Directory containing the training script
os.makedirs(train_script_dir, exist_ok = True)

with open(os.path.join(train_script_dir, 'training_script.py'), 'w') as f:
    f.write("""
import tensorflow as tf
import tensorflow_hub as hub
import argparse
import os

def create_model(module_url, num_classes):
    # Input layer for images
    image_input = tf.keras.layers.Input(shape=(224, 224, 3))

    # Load TensorFlow Hub module
    hub_layer = hub.KerasLayer(module_url, trainable=False)  # keep weights as is

    # Apply the hub module
    hub_output = hub_layer(image_input)

    # Add custom classification layers
    x = tf.keras.layers.Dense(128, activation='relu')(hub_output)
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    # Build and return the model
    model = tf.keras.models.Model(inputs=image_input, outputs=output)
    return model

def load_dataset(data_dir):
    #Load data using tensorflow methods, placeholder for demonstration
    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split=0.2)
    train_data = image_generator.flow_from_directory(data_dir, target_size=(224, 224),
                                                  batch_size=32, subset='training', class_mode='categorical')
    val_data   = image_generator.flow_from_directory(data_dir, target_size=(224, 224),
                                                batch_size=32, subset='validation', class_mode='categorical')

    return train_data, val_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type = str, help = "path to input data folder")
    args = parser.parse_args()
    # Example of specifying a module from TF Hub
    module_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4"
    
    # Determine the number of classes
    # placeholder implementation, ideally this information should be passed
    train_data, val_data = load_dataset(args.data_dir)
    num_classes = train_data.num_classes

    # Build the model with the module
    model = create_model(module_url, num_classes)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model (using dummy datasets for demonstration)
    model.fit(train_data, validation_data=val_data, epochs=10)
    
    model.save('output/image_model')
""")

# Configure the TensorFlow estimator
estimator = TensorFlow(source_directory=train_script_dir,
                       script_params = {'--data_dir': ds.path('image_data')},
                       entry_script='training_script.py',
                       compute_target=compute_target,
                       framework_version='2.11',
                       use_gpu=True,
                       pip_packages=['tensorflow-hub'])

# Submit the experiment run
run = experiment.submit(estimator)

run.wait_for_completion(show_output=True)
```

This code snippet demonstrates several critical steps. Firstly, within the `create_model` function, a `hub.KerasLayer` instance is created using the specified `module_url`, loading the pre-trained module. The `trainable=False` argument ensures that the module’s weights are frozen during training. This preserves the knowledge embedded in the pre-trained network. Subsequently, custom dense layers are added to adapt the module’s output to the target task. Note the placeholder load\_dataset and arguments parsing.  In real case scenario, loading from Datastore, and more robust argument passing would be required.  This entire process runs in an Azure ML environment.

Next, let's illustrate how text embedding modules can be used. Consider a sentiment analysis task:

```python
import tensorflow as tf
import tensorflow_hub as hub
from azureml.core import Workspace
from azureml.core.compute import ComputeTarget
from azureml.core import Experiment
from azureml.train.tensorflow import TensorFlow
import os

# Azure Machine Learning Configuration (replace with your own values)
subscription_id = "your_subscription_id"
resource_group = "your_resource_group"
workspace_name = "your_workspace_name"

ws = Workspace(subscription_id, resource_group, workspace_name)
cluster_name = "your_compute_cluster_name" # replace with your compute cluster name or create one
compute_target = ComputeTarget(workspace=ws, name=cluster_name)
# define data location
ds = ws.get_default_datastore()

# Define experiment
experiment = Experiment(workspace=ws, name="tfhub_text_classification")

# Define training script
train_script_dir = './training_script' #Directory containing the training script
os.makedirs(train_script_dir, exist_ok = True)

with open(os.path.join(train_script_dir, 'training_script.py'), 'w') as f:
    f.write("""
import tensorflow as tf
import tensorflow_hub as hub
import argparse
import os
import numpy as np

def create_model(module_url, num_classes):
    # Input layer for text
    text_input = tf.keras.layers.Input(shape=(1,), dtype=tf.string)

    # Load TensorFlow Hub module
    hub_layer = hub.KerasLayer(module_url, trainable=False)

    # Apply the hub module
    hub_output = hub_layer(text_input)
    
    # Add custom classification layers
    x = tf.keras.layers.Dense(64, activation='relu')(hub_output)
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    # Build and return the model
    model = tf.keras.models.Model(inputs=text_input, outputs=output)
    return model

def load_dataset():
    #Dummy implementation of loading the dataset.
    #Typically, this would be read from a CSV or other storage location
    texts = ["this is great", "not good at all", "ok", "amazing!", "terrible"]
    labels = np.array([1, 0, 1, 1, 0]) # 1 represents positive, 0 represents negative

    dataset = tf.data.Dataset.from_tensor_slices((texts, labels)).batch(2)

    return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
     # Example of specifying a module from TF Hub
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    
    # Determine the number of classes
    num_classes = 2 # Placeholder
    dataset = load_dataset()

    # Build the model with the module
    model = create_model(module_url, num_classes)

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model (using dummy datasets for demonstration)
    model.fit(dataset, epochs=5)

    model.save('output/text_model')
""")

# Configure the TensorFlow estimator
estimator = TensorFlow(source_directory=train_script_dir,
                       entry_script='training_script.py',
                       compute_target=compute_target,
                       framework_version='2.11',
                       use_gpu=True,
                       pip_packages=['tensorflow-hub'])

# Submit the experiment run
run = experiment.submit(estimator)

run.wait_for_completion(show_output=True)
```

This script utilizes the Universal Sentence Encoder module. This module transforms text into vector embeddings that capture sentence-level semantic information. The input layer is configured to accept string type and the output of this hub layer will be used for a dense classification layer. Again, a dummy dataset is used, but the Azure pipeline setup remains consistent with previous example.

Finally, consider a scenario where we would like to fine-tune a pre-trained model instead of keeping it frozen:

```python
import tensorflow as tf
import tensorflow_hub as hub
from azureml.core import Workspace
from azureml.core.compute import ComputeTarget
from azureml.core import Experiment
from azureml.train.tensorflow import TensorFlow
import os

# Azure Machine Learning Configuration (replace with your own values)
subscription_id = "your_subscription_id"
resource_group = "your_resource_group"
workspace_name = "your_workspace_name"

ws = Workspace(subscription_id, resource_group, workspace_name)
cluster_name = "your_compute_cluster_name" # replace with your compute cluster name or create one
compute_target = ComputeTarget(workspace=ws, name=cluster_name)
# define data location
ds = ws.get_default_datastore()

# Define experiment
experiment = Experiment(workspace=ws, name="tfhub_image_fine_tune")

# Define training script
train_script_dir = './training_script' #Directory containing the training script
os.makedirs(train_script_dir, exist_ok = True)

with open(os.path.join(train_script_dir, 'training_script.py'), 'w') as f:
    f.write("""
import tensorflow as tf
import tensorflow_hub as hub
import argparse
import os

def create_model(module_url, num_classes):
    # Input layer for images
    image_input = tf.keras.layers.Input(shape=(224, 224, 3))

    # Load TensorFlow Hub module
    hub_layer = hub.KerasLayer(module_url, trainable=True)  # Make weights trainable

    # Apply the hub module
    hub_output = hub_layer(image_input)

    # Add custom classification layers
    x = tf.keras.layers.Dense(128, activation='relu')(hub_output)
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    # Build and return the model
    model = tf.keras.models.Model(inputs=image_input, outputs=output)
    return model

def load_dataset(data_dir):
    #Load data using tensorflow methods, placeholder for demonstration
    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split=0.2)
    train_data = image_generator.flow_from_directory(data_dir, target_size=(224, 224),
                                                  batch_size=32, subset='training', class_mode='categorical')
    val_data   = image_generator.flow_from_directory(data_dir, target_size=(224, 224),
                                                batch_size=32, subset='validation', class_mode='categorical')

    return train_data, val_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type = str, help = "path to input data folder")
    args = parser.parse_args()
    # Example of specifying a module from TF Hub
    module_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4"
    
     # Determine the number of classes
    train_data, val_data = load_dataset(args.data_dir)
    num_classes = train_data.num_classes

    # Build the model with the module
    model = create_model(module_url, num_classes)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model (using dummy datasets for demonstration)
    model.fit(train_data, validation_data=val_data, epochs=10)
    
    model.save('output/image_model')
""")

# Configure the TensorFlow estimator
estimator = TensorFlow(source_directory=train_script_dir,
                       script_params = {'--data_dir': ds.path('image_data')},
                       entry_script='training_script.py',
                       compute_target=compute_target,
                       framework_version='2.11',
                       use_gpu=True,
                       pip_packages=['tensorflow-hub'])

# Submit the experiment run
run = experiment.submit(estimator)

run.wait_for_completion(show_output=True)
```

The key difference here lies in the `trainable=True` argument during the instantiation of the `hub.KerasLayer`. This enables fine-tuning the pre-trained module alongside the new custom layers. This requires more training data compared to simply utilizing a frozen layer, but can result in optimal model performance in many cases.

For further information regarding Tensorflow Hub I recommend consulting the official TensorFlow documentation, which includes a detailed overview of modules available and specific use cases. Additionally, the Azure Machine Learning documentation provides thorough coverage of integrating external libraries within its training environments. Finally, various tutorials and examples of use cases can be found by simply searching on common search engines.
