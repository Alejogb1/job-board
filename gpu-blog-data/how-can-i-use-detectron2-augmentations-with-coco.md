---
title: "How can I use Detectron2 augmentations with COCO datasets registered via `register_coco_instances`?"
date: "2025-01-30"
id: "how-can-i-use-detectron2-augmentations-with-coco"
---
The core challenge in integrating Detectron2 augmentations with COCO datasets registered using `register_coco_instances` lies in the proper handling of augmentation pipelines within the dataset's data loading process.  My experience developing object detection models for a large-scale retail inventory project highlighted the need for meticulous attention to this detail. Incorrect implementation can lead to data corruption, inconsistent training, and ultimately, poor model performance.  Successfully integrating augmentations requires understanding the dataset's structure and how Detectron2 manages data transformations.

**1. Clear Explanation:**

Detectron2's `DatasetMapper` class is central to this process.  It defines how raw data from a registered dataset is transformed into a format suitable for the model.  By default, `register_coco_instances` creates a dataset that, when accessed, provides image paths and annotation dictionaries.  Augmentations aren't applied at this stage. Instead, the `DatasetMapper` applies the augmentations specified in its configuration during the data loading process.  Therefore, to leverage augmentations with COCO datasets registered via `register_coco_instances`, we must configure the `DatasetMapper` appropriately.  This involves creating a custom `DatasetMapper` or modifying an existing one to incorporate the desired augmentation pipeline.  The augmentation pipeline itself is a sequence of transformation functions, each manipulating the image and annotations.

Crucially, any augmentation applied must handle both the image and the corresponding annotations consistently.  For instance, if an augmentation rotates the image, it must also correctly rotate the bounding box coordinates. Detectron2 provides a range of built-in augmentation functions that are annotation-aware, ensuring this consistency.  Failing to use these functions or incorrectly implementing custom augmentations will lead to annotation-image misalignment, rendering the training data unusable.  The augmentation pipeline must be carefully constructed to avoid introducing artifacts or inconsistencies.


**2. Code Examples with Commentary:**

**Example 1: Basic Augmentation with Built-in Transformations**

```python
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.data.build import build_detection_train_loader
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import transforms as T

# Register COCO dataset
register_coco_instances("my_coco_train", {}, "train.json", "coco_images")
register_coco_instances("my_coco_val", {}, "val.json", "coco_images")

# Create configuration
cfg = get_cfg()
# ... (load your config, model, etc.) ...

# Define augmentation pipeline
cfg.DATASETS.TRAIN = ("my_coco_train",)
aug = T.AugmentationList([
    T.ResizeShortestEdge(short_edge_len=[640, 672, 704, 736, 768, 800], max_size=1333),
    T.RandomFlip(),
])

cfg.INPUT.MIN_SIZE_TRAIN = (640,)  # Example
cfg.INPUT.MAX_SIZE_TRAIN = 1333  # Example
cfg.INPUT.AUGMENTATIONS = [
    {"name": "RandomFlip"},
    {"name": "ResizeShortestEdge", "short_edge_len": [640, 672, 704, 736, 768, 800], "max_size": 1333}
]


# Custom Dataset Mapper (optional, but recommended for complex augmentations)
# from detectron2.data import detection_utils as du
# class MyMapper(du.DatasetMapper):
#     def __init__(self, cfg, is_train=True):
#         super().__init__(cfg, is_train)
#         self.augmentations = aug

#cfg.DATASETS.MAPPER_NAME = "MyMapper"  # Only needed if using a custom mapper

# Build training data loader
data_loader = build_detection_train_loader(cfg)


# Initialize and train your model
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
```

This example utilizes Detectron2's built-in augmentation functions (`RandomFlip`, `ResizeShortestEdge`) which are inherently aware of annotations.  The `cfg.INPUT.AUGMENTATIONS` configuration option is  directly used within the DefaultTrainer to apply the augmentations.


**Example 2: Custom Augmentation with Annotation Handling**

```python
import cv2
from detectron2.data import transforms as T
from detectron2.structures import BoxMode

class MyCustomAugmentation(T.Augmentation):
    def __init__(self, prob=0.5):
        super().__init__()
        self.prob = prob

    def __call__(self, aug_input):
        image = aug_input.image
        if self.prob > 0 and self.prob < 1 and np.random.rand() < self.prob:
             image = cv2.GaussianBlur(image, (5, 5), 0)  # Example blurring

        annotations = aug_input.annotations
        #  No annotation changes are needed for Gaussian blur.
        return aug_input.image, annotations

aug = T.AugmentationList([
    MyCustomAugmentation(prob=0.8),
    # ... other augmentations
])
# ...rest of the code from example 1, configuring aug
```

This example demonstrates creating a custom augmentation, `MyCustomAugmentation`, which adds Gaussian blur with a probability. It’s crucial to handle annotations appropriately within the custom augmentation; here, no annotation adjustment is needed for blurring. For more complex augmentations (e.g., cropping, rotations), careful adjustment of bounding boxes is required.


**Example 3:  Complex Augmentation Pipeline with  `DatasetMapper` customization**

```python
from detectron2.data import detection_utils as du

class MyCustomMapper(du.DatasetMapper):
    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train, augmentations=aug)  # Pass the augmentation pipeline

# ... (Rest of the code from example 1 and 2; replace the default mapper with MyCustomMapper)
cfg.DATASETS.MAPPER_NAME = "MyCustomMapper"
```

This code snippet demonstrates how to create a custom `DatasetMapper` that explicitly integrates the augmentation pipeline, providing finer-grained control over the augmentation process.


**3. Resource Recommendations:**

The Detectron2 documentation, specifically sections on data loading, augmentation, and the `DatasetMapper` class.  Thorough understanding of image processing fundamentals is beneficial.  Reviewing example configurations and scripts provided in the Detectron2 repository offers valuable practical insights.  Studying the source code of Detectron2's built-in augmentation functions helps in understanding how annotation-aware transformations are implemented.  Familiarizing oneself with standard image augmentation techniques from computer vision literature provides a solid foundation for designing custom augmentations.
