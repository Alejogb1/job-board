---
title: "How to fine-tune object detection models using Detectron2?"
date: "2024-12-16"
id: "how-to-fine-tune-object-detection-models-using-detectron2"
---

Alright, let’s tackle this one. Fine-tuning object detection models, especially using something like Detectron2, can feel like navigating a complex ecosystem. I remember back in 2019 when I was working on a project to automatically analyze satellite imagery for urban sprawl detection. We started with a pre-trained Mask R-CNN model from Detectron2, but quickly realized its generic object classes were wildly insufficient for our specific needs. We needed it to recognize things like 'construction sites,' 'new residential buildings,' and 'undeveloped land' – not your standard 'person,' 'car,' or 'dog.' So, that's where the fine-tuning journey began, and I’ve picked up a few tricks along the way.

Essentially, fine-tuning in the context of object detection involves taking a pre-trained model (trained on a massive dataset like COCO) and adapting it to a specific task using your own, often smaller, dataset. The core idea is to leverage the feature extraction capabilities the model has already learned, rather than starting from scratch, which would require enormous amounts of data and computational resources.

One critical aspect, and where I've seen a lot of people stumble, is getting the data preparation *correct*. Detectron2 expects data in a specific format, typically a json file with annotations that describe the bounding boxes and labels of the objects in your images. If this stage is not handled correctly, the model will be fed garbage, and no amount of fine-tuning magic will help. Here's a snippet in Python that shows how I would approach the data formatting step when we need to use a custom data:

```python
import json
import os

def create_detectron2_json(image_dir, annotation_dir, output_json):
    """Converts custom data to Detectron2-compatible JSON format."""

    dataset = []
    image_id = 0

    for filename in os.listdir(annotation_dir):
        if filename.endswith(".txt"):
            image_filename = filename.replace(".txt", ".jpg") # Assuming .jpg images
            image_path = os.path.join(image_dir, image_filename)

            if not os.path.exists(image_path):
                print(f"Warning: Image file {image_filename} missing for annotation {filename}")
                continue


            annotations = []
            with open(os.path.join(annotation_dir, filename), "r") as f:
                for line in f:
                    parts = line.strip().split(" ")
                    class_id = int(parts[0])
                    x1, y1, x2, y2 = map(float, parts[1:])
                    bbox = [x1, y1, x2, y2] # Detectron2 expects [x1, y1, x2, y2] format
                    annotation = {
                        "bbox": bbox,
                        "bbox_mode": 0, # BoxMode.XYXY_ABS (absolute coordinates)
                        "category_id": class_id, # Mapping your classes to 0-n
                        "iscrowd": 0
                    }
                    annotations.append(annotation)


            image_info = {
                "file_name": image_path,
                "id": image_id,
                "height": 1000, # replace with the height of images if you know
                "width": 1000,  # replace with the width of images if you know
                "annotations": annotations
            }
            dataset.append(image_info)
            image_id += 1

    with open(output_json, "w") as outfile:
         json.dump(dataset, outfile)


# Example Usage (assuming images are in 'images/' and annotations in 'labels/'):
image_directory = "images"
annotation_directory = "labels"
output_file = "custom_dataset.json"
create_detectron2_json(image_directory, annotation_directory, output_file)

```

This snippet goes through each annotation file, parses the bounding box information and class labels, and compiles them into a Detectron2 compatible JSON file along with the corresponding image file path. I've always found that doing this correctly upfront saves an immense amount of debugging further down the line. The critical part here is the `bbox_mode`: `0` corresponds to `BoxMode.XYXY_ABS` in Detectron2. This step requires careful attention to your dataset’s coordinate format.

Now, after getting the data in the right format, comes the actual fine-tuning process. Detectron2 provides a flexible `Trainer` class that takes care of most of the heavy lifting. However, you’ll need to configure it with the correct settings, especially when dealing with a small, custom dataset. This is a common issue because smaller datasets can lead to overfitting. Regularization strategies are your friend here. I learned this the hard way, encountering significant overfitting during my satellite imagery project, with the model doing incredibly well on the training set and horribly on any new satellite images. We applied different data augmentation strategies such as rotation, scaling, and flipping of the training images to artificially expand the available data, preventing such issues.

Here's a code example demonstrating basic fine-tuning with Detectron2, highlighting key parameters that I found crucial:

```python
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances

# 1. Register your custom dataset
register_coco_instances("my_custom_dataset", {}, "custom_dataset.json", "")

# 2. Configure Detectron2 for Fine-tuning
cfg = get_cfg()
cfg.merge_from_file("detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Base model
cfg.DATASETS.TRAIN = ("my_custom_dataset",)
cfg.DATASETS.TEST = () # No need for a test set during fine-tuning, I usually use a separate dataset later for evaluation.
cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_50_FPN_3x/137849456/model_final_45727c.pkl" # Pre-trained weights
cfg.SOLVER.IMS_PER_BATCH = 2 # Adjust based on your available GPU memory.
cfg.SOLVER.BASE_LR = 0.00025 # Very important parameter: Use a small LR for fine-tuning!
cfg.SOLVER.MAX_ITER = 2000  # Number of iterations, can increase if the loss not going down.
cfg.SOLVER.STEPS = [] # No learning rate decay initially, you can add learning rate decay later.
cfg.SOLVER.WARMUP_ITERS = 500 # Warmup can be helpful
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3 # Number of classes in your custom data
cfg.OUTPUT_DIR = "output_fine_tuned" # Directory to save the models.

# 3. Create and Train the Model
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False) # Change to True if you resume training
trainer.train()
```

Here, I am using the Faster R-CNN model as the base model, but you can easily swap that for other models provided by Detectron2. Key things to note are the learning rate (`SOLVER.BASE_LR`), number of classes, and the dataset registrations. These parameters are where most of the tuning happens to get good performance. I usually start with a very low learning rate (e.g., 0.00025 as shown), because the pre-trained model's weights are already pretty good. I've noticed that cranking up the LR might lead to forgetting the feature representation that the network has already learnt. This can be further improved by monitoring the training loss and validating it in a separate test dataset.

Finally, once the fine-tuning is done, it's critical to evaluate the model properly. You shouldn’t just eyeball the results. I’ve learned that having a good holdout test set is essential to gauge the actual performance. Here's how you'd evaluate the fine-tuned model:

```python
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

# 1. Register a test dataset (if needed)

register_coco_instances("my_custom_test", {}, "custom_test_dataset.json", "")
cfg.DATASETS.TEST = ("my_custom_test",)

# 2. Load the model weights
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Test score threshold. I adjust this based on the desired balance between precision and recall.

# 3. Create an Evaluator and Perform Inference
evaluator = COCOEvaluator("my_custom_test", output_dir="./evaluation_results")
val_loader = build_detection_test_loader(cfg, "my_custom_test")
results = inference_on_dataset(trainer.model, val_loader, evaluator)

print(results)
```

This code first registers a separate test dataset and then loads the model weights from the output directory produced during training. Then, it runs inference on the test set and calculates metrics like Average Precision (AP) which can give a good sense of how well the model is performing.

For further study on this, I highly recommend diving into the original Detectron2 documentation; the paper by He, Gkioxari, Dollár, and Girshick, "Mask R-CNN," published in 2017, provides crucial insights into the underlying architecture. For a more general deep learning understanding, *Deep Learning* by Goodfellow, Bengio, and Courville is an excellent resource. Also, the research papers on transfer learning, like “How transferable are features in deep neural networks” by Yosinski et al., are beneficial for deeper understanding into the mechanics of fine-tuning.

In closing, fine-tuning isn’t a magic bullet; it requires patience, experimentation, and a solid understanding of the dataset, model parameters, and the evaluation metrics. But with a structured approach, it’s entirely possible to get a good object detection model to perform precisely as you need. I hope my insights from my own practical experience have helped, and happy coding!
