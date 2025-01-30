---
title: "How can I save and use a Detectron2 model to predict on a single image?"
date: "2025-01-30"
id: "how-can-i-save-and-use-a-detectron2"
---
Detectron2, a powerful object detection library from Facebook AI Research, utilizes a configuration-driven approach for model definition and training. This implies that saving and reusing a trained model for single-image predictions involves not just the model weights, but also the entire configuration associated with it. Therefore, the correct methodology revolves around exporting the configuration along with the trained model weights and then loading it back to instantiate the model for inference.

Specifically, the process can be dissected into two primary stages: saving the trained model with its configuration, and loading the saved model for inference on a single image. I've frequently used this in my projects involving automated visual inspection and robotics, where fast and efficient object recognition is paramount. Over the past few years I’ve found it’s less efficient to retrain every time, so saving and loading became standard procedure.

**Saving the Trained Model with Configuration**

During Detectron2 training, the model’s weights are typically stored periodically, often at the end of each training epoch, along with key metrics. However, the core challenge is to capture the precise model architecture and data preprocessing steps – the configuration itself. Detectron2 facilitates this by encapsulating the complete configuration into a Python dictionary and saving this alongside the model weights. This configuration information is crucial as it dictates how input images are processed, which layers are utilized, and what parameters are employed during inference.

The training loop in Detectron2 typically involves an `Trainer` object. After training is complete, the following code snippet would be utilized to save the trained model and its configuration:

```python
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
import os

# Assuming 'cfg' object is already defined through the Detectron2 configuration
# and training is completed
# Example: cfg = get_cfg()
#          cfg.merge_from_file(your_config_file)
#          cfg.MODEL.WEIGHTS = your_pretrained_weight_file
#          trainer = DefaultTrainer(cfg)
#          trainer.resume_or_load(resume=False)
#          trainer.train() # Assuming training finished

def save_model_with_config(cfg, model_path, config_path):
  """Saves the trained model and its configuration.

  Args:
      cfg: The Detectron2 configuration object.
      model_path: The path where model weights will be saved.
      config_path: The path where configuration will be saved.
  """

  # Saving model weights
  torch.save(trainer.model.state_dict(), model_path)

  # Saving configuration
  with open(config_path, 'w') as f:
    f.write(cfg.dump())

# Setting the paths
model_save_path = "trained_model.pth"
config_save_path = "trained_config.yaml"

# Calling the function to save
save_model_with_config(cfg, model_save_path, config_save_path)

```

*Explanation:*

1.  `torch.save(trainer.model.state_dict(), model_path)`: This line saves only the learned parameters (weights and biases) of the model into the specified `model_path`. This is a dictionary of tensors, making it compact and efficient to load later.
2.  `cfg.dump()`:  This method converts the entire Detectron2 configuration object (`cfg`) into a YAML formatted string which allows readability and is easily parsable. This encompasses all settings related to the network’s architecture, dataset, and training procedure.
3. The configuration is saved as a YAML file at `config_path`. This keeps the necessary information for correct loading.

**Loading the Model and Performing Inference**

To perform inference on a single image, one has to first reconstruct the model architecture using the previously saved configuration.  The saved model weights can then be loaded into this reconstructed architecture. Finally, the single image is prepared for input into the model. I’ve used this frequently when running models within serverless functions where efficiency is paramount. The following code illustrates the process:

```python
import torch
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.data.detection_utils import read_image
from detectron2.structures import Instances
import cv2
import numpy as np

def load_model_for_inference(config_path, model_path, device="cpu"):
  """Loads the trained model and prepares it for inference.

  Args:
      config_path: The path where the saved config is stored.
      model_path: The path where the saved weights are stored.
      device: The device to run the model on (cpu or cuda).
  Returns:
      The built and loaded model ready for inference.
  """

  # Load the saved configuration
  cfg = get_cfg()
  cfg.merge_from_file(config_path)
  cfg.MODEL.WEIGHTS = model_path
  cfg.MODEL.DEVICE = device #setting device for inference
  cfg.freeze()


  # Build the model from the configuration
  model = build_model(cfg)
  model.eval() # Setting model to evaluation mode
  return model


def predict_single_image(model, image_path, device = "cpu"):
    """ Runs inference on a single image

    Args:
      model: the model loaded for inference
      image_path: path of the image to run prediction
      device: device on which to perform the calculation
    Returns:
        Detected Instances object
    """

    # Read the image
    img = read_image(image_path, format="BGR")

    # Image format for inference
    image_tensor = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))

    if device == "cuda":
      image_tensor = image_tensor.cuda()
    with torch.no_grad(): # no gradients computed for inference
        # Passing the image to the model
        inputs = {"image": image_tensor, "height": img.shape[0], "width": img.shape[1]}
        outputs = model([inputs])[0]

        # Output is a Detectron2 instance object
        return outputs
# Example usage
inference_model = load_model_for_inference(config_save_path, model_save_path, device="cuda")
# Example of loading an image
image_file = "example_image.jpg"

# Load and process the image
predictions = predict_single_image(inference_model, image_file, device = "cuda")
print(predictions)
```

*Explanation:*

1.  `cfg = get_cfg(); cfg.merge_from_file(config_path)`:  Here, the saved configuration from the YAML file (`config_path`) is loaded, re-establishing the complete model configuration using the Detectron2 configuration manager. We set the `MODEL.WEIGHTS` path as the previously saved model weights file.  `cfg.freeze()` makes the loaded configuration immutable to avoid accidental changes during inference
2.  `model = build_model(cfg)`: With the configuration loaded, the `build_model` function initializes the model architecture identical to that used during training.  The model is set to evaluation mode with `model.eval()` for disabling gradient computation.
3.  `image_tensor = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))`: This code transforms the input image to the correct format for the model. It converts the NumPy array representing the image into a tensor, explicitly ensuring the floating-point data type (float32) is used for the pixel data and transposes the channels first so the model receives a channels-first format.
4.  `with torch.no_grad()`: Prevents backpropagation and gradient calculation during inference.
5.  The raw output of the model is a Python dictionary of tensors, specifically a list of Instances which contains bounding boxes, masks, scores and classes. These are now ready for post-processing.

**Visualization**

While not strictly part of the initial question, the output can be visualized. Since the output `predictions` contains `pred_boxes`, `pred_classes` and `scores`, one could use this information with the `Visualizer` provided in `detectron2.utils.visualizer`. The following code snippet shows an example:

```python
from detectron2.utils.visualizer import Visualizer
import cv2
import numpy as np

def visualize_prediction(image_path, predictions, metadata):
  """Visualizes the prediction results over an image

  Args:
    image_path: Path of the image on which the prediction was made
    predictions: Detectron2 Instances object containing predictions
    metadata: detectron2 Metadata object. Can be obtained using DatasetCatalog and MetadataCatalog
  Returns:
    visualized image as a np array.
  """
  image = cv2.imread(image_path)
  visualizer = Visualizer(image[:, :, ::-1], metadata=metadata, scale=0.5) # Convert BGR to RGB

  vis_output = visualizer.draw_instance_predictions(predictions)

  # Return as a numpy array of the image
  return vis_output.get_image()[:, :, ::-1] # Convert back to BGR


# Example of getting the metadata for the dataset (in our case, COCO)
from detectron2.data import DatasetCatalog, MetadataCatalog

dataset_name = "coco_2017_val"
metadata = MetadataCatalog.get(dataset_name)


# Run the prediction
vis_image = visualize_prediction(image_file, predictions, metadata)
# Display or Save
cv2.imwrite("result_image.jpg", vis_image)
```
*Explanation:*
1. `metadata = MetadataCatalog.get(dataset_name)`: This line retrieves metadata relevant to the dataset the model was trained on, essential for proper label rendering on the visualized image.
2.  The `Visualizer` object takes an input image and its metadata. It draws bounding boxes, mask, labels and scores using this information on the image.
3.  The resulting image is returned as a Numpy array and is saved as a `result_image.jpg`.

**Resource Recommendations**

To further your understanding, consult the official Detectron2 documentation available on the project's website. Additionally, review the tutorial notebooks provided within the Detectron2 repository. These notebooks demonstrate various aspects of the library, including training, configuration management, and inference, which is essential when building end-to-end solutions. I would also recommend exploring open source repositories on platforms like GitHub, as this provides practical insight into how the library can be used in a real-world context, something I’ve found invaluable.
