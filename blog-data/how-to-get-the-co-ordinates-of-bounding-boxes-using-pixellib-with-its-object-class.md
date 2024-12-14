---
title: "How to get the co-ordinates of bounding boxes using pixellib with its object class?"
date: "2024-12-14"
id: "how-to-get-the-co-ordinates-of-bounding-boxes-using-pixellib-with-its-object-class"
---

alright, so you're looking to extract bounding box coordinates using pixellib, specifically with its object detection capabilities, huh? i've been down that road, many, many times. i remember when i first got into computer vision, i was practically throwing code at the wall, hoping something would stick. getting those bounding boxes was always a fiddly part. but hey, we've all been there. let me share some of what i've learned and hopefully save you some headache.

first things first, pixellib, for those who might be less familiar, uses mask r-cnn under the hood. it's a powerful architecture for both object detection and segmentation. the good thing for us here is that it provides convenient methods for accessing the detection results, including bounding boxes.

the general approach boils down to these steps: you load your image, perform object detection, and then iterate through the results to extract the coordinates. now, let's dive into some code examples because that's probably what you're here for anyway.

assume you've got pixellib installed. if not, `pip install pixellib` is your friend. and you've got your image, let's call it 'my_image.jpg', we'll make some examples with this in mind.

here's a basic example where we just grab and print the bounding boxes, the simplest way i can think about showing you this:

```python
import pixellib
from pixellib.instance import instance_segmentation

#load the instance segmentation model
segment_image = instance_segmentation()
segment_image.load_model("mask_rcnn_coco.h5") #or your own trained model

#detect objects
detections = segment_image.segmentImage(image_path="my_image.jpg", show_bboxes = True)

#extract the bounding boxes
if detections is not None: # check if detection is successful
    for box in detections[1]['rois']:
        x1, y1, x2, y2 = box
        print(f"bounding box: x1:{x1}, y1:{y1}, x2:{x2}, y2:{y2}")

else:
    print("No object found in the image")


```

let me break this down:
*   we import the necessary pixellib libraries.
*   we load an instance segmentation model. i'm using `mask_rcnn_coco.h5`, which is pre-trained on the coco dataset and good for general purpose detection. you may need to download it first if you don't have it locally; that's one of the things i usually forget about when i'm setting things up in new systems. it happens to the best of us. you could also use a custom-trained model.
*   we run segmentation on our image `my_image.jpg`. the `show_bboxes=true` just shows the image with boxes on it but doesn't actually extract the data itself.
*   the result is stored as a tuple, detections[0] are masks, detections[1] has all the rest, we check if the detections is not empty, then iterate over the bounding boxes in `detections[1]['rois']`. rois is short for 'regions of interest' by the way.
*   finally, we unpack the bounding box coordinates and print them. these coordinates are `x1, y1, x2, y2`, representing the top-left and bottom-right corners, respectively.

now, printing to the console is nice, but in real-world applications, you'd probably need to store or process these coordinates. here's an enhanced version that does that and also adds the class id and score associated with the detection:

```python
import pixellib
from pixellib.instance import instance_segmentation
import json # we'll use json for a better output example

#load the instance segmentation model
segment_image = instance_segmentation()
segment_image.load_model("mask_rcnn_coco.h5")

#detect objects
detections = segment_image.segmentImage(image_path="my_image.jpg", show_bboxes = True)

#extract bounding boxes and other information
bounding_boxes_data = []
if detections is not None:
  for i, box in enumerate(detections[1]['rois']):
      x1, y1, x2, y2 = box
      class_id = detections[1]['class_ids'][i]
      score = detections[1]['scores'][i]
      bounding_boxes_data.append({
          'box': [x1, y1, x2, y2],
          'class_id': int(class_id),
          'score': float(score)
      })

  # save the bounding boxes into a file called bboxes.json
  with open("bboxes.json", "w") as f:
    json.dump(bounding_boxes_data, f, indent=4)

else:
  print("No object found in the image")
```

here's what's new:
*   we create an empty list `bounding_boxes_data` to hold all the results.
*   for each bounding box, we also get the `class_id` and the `score`.
*   we append all of this data into `bounding_boxes_data`, now each bounding box is an object.
*   finally, we save the data into a `bboxes.json` file so we can use it later. the `indent` parameter in the `json.dump()` function makes the json file human readable.

this is more practical, you can now easily manipulate the bounding box information and classes in your other scripts or pass the data to other services or models.

the `class_ids` correspond to the classes the model was trained on. for the `mask_rcnn_coco.h5` model, you can find a list of these in the coco dataset documentation. there are 80 categories ranging from person, car, bird, etc. that's something important to keep in mind when using these pre-trained models. what if you want to check what classes exist in the dataset you are using? i know this is not the original question but i think that is important for the sake of clarity and for you to learn about this too:

```python
import pixellib
from pixellib.instance import instance_segmentation

#load the instance segmentation model
segment_image = instance_segmentation()
segment_image.load_model("mask_rcnn_coco.h5")

#get class names for the model
class_names = segment_image.load_model_classes()

#print the classes
for i, class_name in enumerate(class_names):
    print(f"{i} : {class_name}")
```

this is a simple and practical addition to help you get to know your classes in the model you are using. you can call this when setting things up, i do this from time to time. i've lost a few hours trying to figure out which class i had to check. you will do it too if you work long enough with these things. and that makes me think about the time i spent an entire day trying to make a model work, turns out i forgot to install one dependency of the library that i was using, that was embarrassing, haha.

i'd also recommend that if you're new to all this, reading through some papers about mask r-cnn might help you a lot, it's not just about using the tool but also about understanding how it works under the hood, which is a very good skill to have. the original mask r-cnn paper is a great place to start. for more in depth understanding, you could look into the book "computer vision: algorithms and applications" by richard szeliski, a classic in the field.

and finally, remember to always check the documentation of the libraries you are using, that's the best source of information, that's how i learned to do all this stuff, by checking the documentation and trying things out. also don't be afraid to ask questions in stackoverflow or the github repository of the project you are using. there is a community of people willing to help, i know i was on the other side when i started and people helped me a lot, it's nice to be able to share the knowledge with others. good luck with your computer vision adventures!
