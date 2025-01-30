---
title: "How can semantic segmentation be implemented using Detectron2?"
date: "2025-01-30"
id: "how-can-semantic-segmentation-be-implemented-using-detectron2"
---
Semantic segmentation, the task of assigning a class label to every pixel in an image, is a crucial component in numerous computer vision applications, from autonomous driving to medical image analysis. Over the past several years I've found that the Detectron2 framework, developed by Facebook AI Research, provides a robust and flexible environment for implementing various instance and semantic segmentation models. My experience with projects ranging from satellite image analysis to robotic manipulation has proven its effectiveness.

Detectron2 builds upon PyTorch and offers pre-trained models, training utilities, and a modular design that significantly streamlines the implementation process. The key to leveraging Detectron2 for semantic segmentation lies in understanding its configuration system, the data loading pipeline, and the model architecture.

The core of Detectron2 revolves around configuration files, typically in YAML format. These files define everything, from the model architecture and optimizer settings to data loading parameters. For semantic segmentation, the configuration defines which pre-trained backbone to utilize (e.g., ResNet, ResNeXt), which segmentation head to employ (e.g., FPN, DeepLab), and the number of classes in the segmentation task. When beginning a new project, I usually start with an existing configuration and modify it to suit the problem at hand.

Data loading in Detectron2 requires creating a custom Dataset class that inherits from the `torch.utils.data.Dataset` class. The key component is the `__getitem__` method, which is called during training. This method reads an image and its corresponding segmentation mask, applies data augmentation techniques, and converts them into tensors suitable for the Detectron2 model. The mask must be in a specific format: a single-channel image where each pixel value corresponds to the class label. Detectron2 handles these specific conversions internally, as long as the data is prepared.

The models themselves are defined through configuration parameters, but are implemented using PyTorch's modular design. During training, the model is passed input images and masks, which it compares to the generated predictions during the forward pass, calculating loss and backpropagating to update weights. Detectron2 abstracts most of the boilerplate PyTorch code, allowing the focus to be on data preparation and model refinement.

Here are three examples that demonstrate different aspects of implementing semantic segmentation using Detectron2:

**Example 1: Basic Semantic Segmentation Training with a Pre-trained Model**

This first example demonstrates the minimal code required to train a semantic segmentation model, utilizing a standard configuration. I would often use a configuration very close to this when quickly validating the data and process.

```python
import detectron2
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json
from detectron2.engine import DefaultTrainer
import os
import numpy as np


def register_custom_dataset():
    # Assuming images and masks are in subdirectories 'images' and 'masks',
    # with masks being single-channel label images.
    DatasetCatalog.register("my_dataset_train", lambda: load_coco_json("path/to/my/annotation_train.json", "path/to/images/train"))
    MetadataCatalog.get("my_dataset_train").set(
        thing_classes=["background", "class_a", "class_b"]
    )  # Modify to your classes
    DatasetCatalog.register("my_dataset_val", lambda: load_coco_json("path/to/my/annotation_val.json", "path/to/images/val"))
    MetadataCatalog.get("my_dataset_val").set(
        thing_classes=["background", "class_a", "class_b"]
    )  # Modify to your classes
    
register_custom_dataset()


cfg = get_cfg()
cfg.merge_from_file(
    "detectron2/configs/Cityscapes/mask_rcnn_R_50_FPN.yaml"  # Pick a suitable config, usually a semantic config.
)
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ("my_dataset_val",)
cfg.MODEL.WEIGHTS = "detectron2://ImageNetPretrained/MSRA/R-50.pkl"  # Load pre-trained weights.
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # Update based on the number of classes in your dataset.
cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 3  # Update based on the number of classes in your dataset.
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.MAX_ITER = 5000
cfg.TEST.EVAL_PERIOD = 1000


os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
```

This code loads a pre-trained Mask R-CNN configuration (although this is typically used for instance segmentation, the semantic segmentation head can be accessed). It then specifies the custom training and validation datasets, loads weights, defines the classes, sets batch size and iterations, initializes the trainer object, and starts training. Note the use of `MetadataCatalog` to assign a human-readable class name to a class ID. This is useful for visualization.  The key lines that control the number of classes are `cfg.MODEL.ROI_HEADS.NUM_CLASSES` and `cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES`. These must match the number of semantic classes to predict. I use the `DefaultTrainer` class, which implements training, validation, logging, and saving.

**Example 2: Using a Custom Semantic Segmentation Head**

This example demonstrates how to customize the model by replacing the default segmentation head with a custom one. This is more complex and usually not necessary unless you are trying out novel model architectures.

```python
import detectron2
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json
from detectron2.modeling import build_model
from detectron2.layers import ShapeSpec, Conv2d,  get_norm
from detectron2.engine import DefaultTrainer
import torch
import torch.nn as nn
import os


def register_custom_dataset():
    # Assuming images and masks are in subdirectories 'images' and 'masks',
    # with masks being single-channel label images.
    DatasetCatalog.register("my_dataset_train", lambda: load_coco_json("path/to/my/annotation_train.json", "path/to/images/train"))
    MetadataCatalog.get("my_dataset_train").set(
        thing_classes=["background", "class_a", "class_b"]
    )  # Modify to your classes
    DatasetCatalog.register("my_dataset_val", lambda: load_coco_json("path/to/my/annotation_val.json", "path/to/images/val"))
    MetadataCatalog.get("my_dataset_val").set(
        thing_classes=["background", "class_a", "class_b"]
    )  # Modify to your classes
    
register_custom_dataset()


class CustomSemanticHead(nn.Module):
    def __init__(self, cfg, input_shape: ShapeSpec):
        super().__init__()
        num_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        norm = cfg.MODEL.SEM_SEG_HEAD.NORM
        self.conv1 = Conv2d(
            input_shape.channels,
            256,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not norm,
            norm=get_norm(norm, 256),
            activation=nn.ReLU()
        )
        self.conv2 = Conv2d(
            256,
            num_classes,
            kernel_size=1,
            stride=1,
            padding=0
        )


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


cfg = get_cfg()
cfg.merge_from_file(
    "detectron2/configs/Cityscapes/mask_rcnn_R_50_FPN.yaml"
)
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ("my_dataset_val",)
cfg.MODEL.WEIGHTS = "detectron2://ImageNetPretrained/MSRA/R-50.pkl"
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 3
cfg.MODEL.SEM_SEG_HEAD.NAME = "CustomSemanticHead"
cfg.MODEL.SEM_SEG_HEAD.NORM = "GN"
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.MAX_ITER = 5000
cfg.TEST.EVAL_PERIOD = 1000


def build_custom_model(cfg):
    model = build_model(cfg)
    model.sem_seg_head = CustomSemanticHead(cfg, model.backbone.output_shape())
    return model


os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.model = build_custom_model(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
```

This code defines `CustomSemanticHead`, a simple head composed of two convolution layers. The first convolutional layer reduces the channels and applies ReLU activation. The second convolutional layer produces the final channel output, which is equal to the number of classes.  We then override the training loop and pass our custom model to the `trainer`. This allows for much more flexible architecture implementation.  Note how we use the `ShapeSpec` to extract channel information for our custom header.

**Example 3: Data Augmentation**

Finally, here's an example focused on data augmentation, which is critical for improving model robustness and generalisation.  Data augmentation should always be considered in semantic segmentation tasks.

```python
import detectron2
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils
from detectron2.engine import DefaultTrainer
import os
import numpy as np
import copy
import random
import torch
import cv2

def register_custom_dataset():
    # Assuming images and masks are in subdirectories 'images' and 'masks',
    # with masks being single-channel label images.
    DatasetCatalog.register("my_dataset_train", lambda: load_coco_json("path/to/my/annotation_train.json", "path/to/images/train"))
    MetadataCatalog.get("my_dataset_train").set(
        thing_classes=["background", "class_a", "class_b"]
    )  # Modify to your classes
    DatasetCatalog.register("my_dataset_val", lambda: load_coco_json("path/to/my/annotation_val.json", "path/to/images/val"))
    MetadataCatalog.get("my_dataset_val").set(
        thing_classes=["background", "class_a", "class_b"]
    )  # Modify to your classes
    
register_custom_dataset()


class CustomMapper:
    def __init__(self, cfg, is_train=True):
        self.is_train = is_train
        self.transforms = []
        if self.is_train:
            self.transforms = [
                T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
                T.RandomRotation(angle=[-10, 10], expand=False),
                T.RandomBrightness(0.8, 1.2),
            ]

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format="BGR")
        mask = cv2.imread(dataset_dict['mask_file_name'], cv2.IMREAD_GRAYSCALE)
        aug = T.AugmentationList(self.transforms)
        image, transforms = T.apply_transform_gens(aug, image)
        
        image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        
        mask = T.apply_transform_gens(transforms, mask)
        
        mask = torch.as_tensor(np.ascontiguousarray(mask))
        
        instances = utils.annotations_to_instances(
            dataset_dict["annotations"],
            image.shape[1:],
        )
        
        return {
            "image": image,
            "instances": instances,
            "sem_seg": mask
            }


cfg = get_cfg()
cfg.merge_from_file(
    "detectron2/configs/Cityscapes/mask_rcnn_R_50_FPN.yaml"
)
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ("my_dataset_val",)
cfg.MODEL.WEIGHTS = "detectron2://ImageNetPretrained/MSRA/R-50.pkl"
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 3
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.MAX_ITER = 5000
cfg.TEST.EVAL_PERIOD = 1000

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.data_loader = trainer.build_train_loader(cfg, mapper=CustomMapper(cfg, is_train=True))
trainer.resume_or_load(resume=False)
trainer.train()
```

This code shows how to incorporate custom data augmentation by creating a new `CustomMapper` class. Here I implement random horizontal flips, random rotations, and random brightness adjustments. The key is to ensure that the data augmentation is applied consistently across the image and its corresponding mask. I then override the default data loader with our custom one.  This provides a very flexible way to inject data augmentations into the training pipeline.  The `T.apply_transform_gens` is the main interface that maps the image transforms to the annotations and the semantic mask. The `utils.read_image` function allows the code to handle images in a convenient format.

In summary, implementing semantic segmentation with Detectron2 primarily involves understanding the configuration system, defining a dataset, and potentially customizing the model architecture. The framework simplifies the complexities of training deep learning models, allowing for experimentation and rapid prototyping.

To further deepen your understanding, I recommend reviewing the Detectron2 documentation and exploring example notebooks. Additionally, studying the configurations for different model architectures such as DeepLabv3+ or U-Net within the Detectron2 repository can provide a valuable insight. Looking into published research papers on semantic segmentation and reviewing the mathematical derivations of the loss functions used in these papers is very helpful, as is researching best practices around the choice of metrics in a segmentation problem. Finally, understanding the PyTorch fundamentals is helpful for diving into custom implementations.
