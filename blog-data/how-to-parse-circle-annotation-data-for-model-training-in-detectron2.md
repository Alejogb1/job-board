---
title: "How to parse circle annotation data for model training in detectron2?"
date: "2024-12-14"
id: "how-to-parse-circle-annotation-data-for-model-training-in-detectron2"
---

alright, so you're looking to feed circle annotations into detectron2 for model training, huh? i've been down this road before, and it's definitely got some quirks. it's not like just slapping bounding boxes around objects, circles require a bit more finesse.

let's break this down from a practical angle. when detectron2 talks about annotations, it usually expects json-formatted data, typically adhering to the coco format. coco, if you haven’t bumped into it, primarily focuses on polygons, boxes and keypoints, and it doesn't directly have circle objects natively, so we have to do a bit of extra work here. i remember one time i was working on an industrial inspection project; we were using circles to identify the presence of particular features on circuit boards. i initially thought i could just hack in an approach without transforming the circle data to polygons but detectron2 just wasn't having it – it's pretty picky.

so, here's the game plan. we’ll take those circle annotations and convert them into a polygon representation which detectron2 will then consume without complaint. essentially, we’ll create a polygon that approximates a circle very closely. this approximation is done by generating many vertices along the circumference of the circle. more points result in a polygon that looks more and more like a circle.

typically, you would have a data structure holding the center x, center y, and the radius of each circle. something like this:

```python
circle_annotations = [
    {"cx": 100, "cy": 150, "radius": 30, "category_id": 1},
    {"cx": 250, "cy": 300, "radius": 50, "category_id": 2},
    {"cx": 400, "cy": 100, "radius": 20, "category_id": 1}
    # and so on
]
```

the `category_id` tells detectron2 what class it is, like if these circles are capacitors, resistors, etc.

the first step is writing a function to convert each of these circles into a polygon. here's how that would look in python:

```python
import numpy as np
import math

def circle_to_polygon(cx, cy, radius, num_points=36):
  """converts circle to a polygon."""
  points = []
  for i in range(num_points):
    angle = 2 * math.pi * i / num_points
    x = cx + radius * math.cos(angle)
    y = cy + radius * math.sin(angle)
    points.append([x, y])
  return np.array(points).flatten().tolist()
```

what this function does is quite simple. it computes a number of points on a circle at even angles. the more `num_points` you select, the smoother the approximation. typically 36 points is sufficient. it then returns a flattened list of x, y coordinates which detectron2 will accept.

next, we have to actually construct our coco-style json data format, i.e. we need to arrange everything in the specific format that detectron2 expects, here is an example:

```python
def create_coco_annotation(circle_annotations, image_id):
    """formats the circle annotation to a coco structure"""
    annotations = []
    annotation_id = 1
    for circle in circle_annotations:
        polygon = circle_to_polygon(circle["cx"], circle["cy"], circle["radius"])
        annotation = {
            "segmentation": [polygon],
            "area": math.pi * circle["radius"] ** 2,  # area of the circle
            "iscrowd": 0,
            "image_id": image_id,
            "bbox": [
                circle["cx"] - circle["radius"],
                circle["cy"] - circle["radius"],
                2 * circle["radius"],
                2 * circle["radius"],
            ],
            "category_id": circle["category_id"],
            "id": annotation_id,
        }
        annotations.append(annotation)
        annotation_id += 1
    return annotations
```

this `create_coco_annotation` method iterates through your `circle_annotations`, transforms each circle to a polygon, and wraps that, and a few other items, into a dictionary following coco structure. it also generates bounding boxes based on the circle’s radius and center, just so we stay compatible with coco spec. the area field is also something coco expects, which we can just calculate easily from radius. we also include an `id` because coco needs that too. `iscrowd` is 0 for individual instances.

finally, here’s how we’d tie all of this into detecting2. typically you need a python dictionary in the following structure : `{"images": [...], "annotations":[...]}` so here is a method to do that:

```python
def format_all_data(all_circle_annotations, image_paths):
    """builds the dictionary format that detectron2 expects"""
    dataset_dicts = []
    for idx, image_path in enumerate(image_paths):
        image_data = {
                'file_name': image_path,
                'height': 600, #put here the actual height of the image
                'width': 800, #put here the actual width of the image
                'id': idx
        }
        annotations = create_coco_annotation(all_circle_annotations[idx], idx)
        dataset_dicts.append({**image_data, 'annotations': annotations})
    return dataset_dicts
```

this method iterates over all your circle annotations which is a list of list of circle annotations and matches them with images. note that i'm just using placeholder height and width. you need to actually extract these values for each individual image path you have. the method returns a list of dictionaries in the format that detectron2 expects.

now, let’s speak about how to register this custom data into detectron2 and use it. we need to register our dataset in the `DatasetCatalog`.

```python
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data import build_detection_train_loader
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg

# load the dataset and the paths here
circle_data = [
   [{"cx": 100, "cy": 150, "radius": 30, "category_id": 0}, {"cx": 250, "cy": 300, "radius": 50, "category_id": 1}], #annotations for image 0
    [{"cx": 100, "cy": 150, "radius": 30, "category_id": 0}, {"cx": 250, "cy": 300, "radius": 50, "category_id": 1}], #annotations for image 1
   [{"cx": 100, "cy": 150, "radius": 30, "category_id": 0}, {"cx": 250, "cy": 300, "radius": 50, "category_id": 1}] #annotations for image 2
]
image_paths = ['image1.png','image2.png','image3.png'] #replace with actual image paths

dataset_dicts = format_all_data(circle_data, image_paths)

def get_circle_dataset():
    """registers the dataset and returns its structure"""
    return dataset_dicts

DatasetCatalog.register("circle_dataset", get_circle_dataset)
MetadataCatalog.get("circle_dataset").set(thing_classes=["class1", "class2"]) #your class names

cfg = get_cfg()
cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_50_FPN_3x/137849456/model_final_457216.pkl"
cfg.DATASETS.TRAIN = ("circle_dataset",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 1000
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
```

i’ve taken the liberty to add the training code at the bottom. the `get_circle_dataset` is the callback function that is called by detectron2. there is also the `metadata` object, which holds a series of metadata that we set, for example, the class names. this has to match your category ids, in our example class1 is for category_id 0 and class 2 for category id 1. this code has a minimal training implementation, but you should add more stuff, like tensorboard logs, saving checkpoints, etcetera...

when dealing with custom data, it's essential to thoroughly validate your annotations. sometimes i've found that a silly mistake like an incorrect radius value can totally mess up the training, so it's a good idea to display a random sample of images with their polygon annotations drawn on them to catch issues early. it's like when you try to assemble furniture without the instructions and end up with an extra screw, it's usually an oversight somewhere.

for extra details on annotations structures and how coco dataset works, you should definitely refer to the original coco paper ["microsoft coco: common objects in context"]. another book that i found to be very useful is ["computer vision: algorithms and applications"] by richard szeliski. also, if you need some theoretical background in general vision and also some intuition on deep learning i'd advise reading ["deep learning"] by goodfellow et al.

remember to install all the necessary dependencies like detectron2, numpy, pytorch, etc. if you haven't already, or things will go really south. also adjust the paths for images and model weights as needed. that's about it. if you have more problems you can post it here, i'm sure someone will help you.
