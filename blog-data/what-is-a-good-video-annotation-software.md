---
title: "What is a good Video annotation software?"
date: "2024-12-15"
id: "what-is-a-good-video-annotation-software"
---

alright, so you're looking for a solid video annotation tool, huh? i've been down that rabbit hole more times than i care to remember. i've spent countless hours wrestling with subpar software, so i can definitely relate. it's one of those things that sounds easy, but it quickly devolves into a messy affair if you don't pick the appropriate tools.

i've worked on a bunch of projects that relied heavily on video annotation. one that sticks out was for a computer vision research project i was on back in university. we were training models to identify specific objects in urban environments - think traffic lights, pedestrians, cars, and that sort of thing. the sheer volume of video data was mind-boggling. we tried a bunch of tools and some of them failed spectacularly. i mean, crashes, inconsistent labeling, import/export nightmares, you name it. we were essentially data annotation archaeologists after each labelling session. it was a true mess.

from those past experiences, i’ve really come to appreciate what makes a video annotation tool ‘good’. a good tool should be efficient, versatile and importantly it has to play well with the other tools in my workflow. there's not much worse than doing hours of work and then finding out that the output format of the video annotator isn't compatible with your actual project.

so, let’s talk about what to look for. first off, you're going to want something that supports a variety of annotation types. rectangular bounding boxes are a must, obviously. but also, things like polygons for irregular shapes, keypoints for pose estimation, and potentially even segmentation masks if your needs are complex, are critical. some projects need very specific kinds of annotations, so having a tool that is versatile is ideal. it also can save a lot of time of having to change tools in the middle of the project.

the interface is also critical. it needs to be intuitive and easy to use. the learning curve should be minimal, so you are not spending all of your time figuring out the tool instead of actually annotating the data. navigation should be smooth, allowing you to jump between frames quickly and easily. it must support both frame-by-frame annotations and interpolated annotations, which are a lifesaver if you need to track objects across frames. i hate those situations where every single frame has to be annotated independently. it's a time sink, plain and simple.

of course, exporting is very important. it has to be able to spit out the annotations in a useful format like json, csv, xml and ideally support standards like coco or yolo. compatibility here is just not an option, it's a requirement for everything to move as it should. the ability to import and export annotations is also great if you need to do batch corrections, or if you want to switch to a different tool in the future.

now, specifically regarding the actual programs, there are a few options. the first one that comes to mind is *vatic*. it's an open-source tool developed at caltech, it is definitely one of the most mature annotation tools and it has been around for a long time. if you ever have to work with videos, you should check it out. the setup could be slightly convoluted, since it requires a specific server infrastructure. the interface is not very modern but it is extremely powerful. this one can do everything you'd normally need, including boxes, polygons and keypoints.

next, you've got *labelme*. it's simpler than *vatic* and also open-source. it's written in python, making it easy to modify or extend. it focuses more on images but it has video annotation capability. the interface is much more pleasant and easier to use than vatic, although it is not as feature rich. if you don’t need all the fancy stuff and you want to set up something quickly, *labelme* is a really good choice. if you want to customize or extend the functionality using python, this is also a good place to start.

another alternative that you may want to consider is *cvat*. this is a more recent option, it's also open source and it’s also gaining popularity. it is feature rich and has a nice modern user interface. it also supports collaborative annotation, which can be very useful for large projects. it's also worth noting that it has a rest api, so you can integrate it with other tools if needed.

here's a little snippet from *labelme* that illustrates how you can convert the annotations to a json file:

```python
import json
from labelme import utils

def labelme_to_json(labelme_json_path, output_json_path):
    with open(labelme_json_path, 'r') as f:
        data = json.load(f)

    image_data = utils.img_b64_to_arr(data['imageData'])
    
    shapes = data['shapes']
    output_data = {
        "image_path":data['imagePath'],
        "height":image_data.shape[0],
        "width":image_data.shape[1],
        "annotations":[]
    }
    
    for shape in shapes:
        if shape['shape_type'] == 'rectangle':
           x1,y1 = shape['points'][0]
           x2,y2 = shape['points'][1]
           annotation = {
                "type":"rectangle",
               "x1":x1,
               "y1":y1,
               "x2":x2,
               "y2":y2,
                "label":shape['label']
            }
           output_data['annotations'].append(annotation)
        elif shape['shape_type'] == 'polygon':
            points = shape['points']
            annotation = {
               "type":"polygon",
                "points":points,
                "label":shape['label']
           }
            output_data['annotations'].append(annotation)
        
    with open(output_json_path, 'w') as outfile:
        json.dump(output_data, outfile,indent=4)

# example usage:
labelme_to_json('example.json','output.json')
```

this script extracts basic annotation information for polygons and rectangles from a labelme json format and saves it into a simplified format. it's a good starting point if you need to process labelme data programmatically.

another snippet of code, this one in python also, is a function to load the *cvat* xml export.

```python
import xml.etree.ElementTree as ET
import json

def parse_cvat_xml(xml_file_path, output_json_path):
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    
    output_data = {"images": []}

    for image_element in root.findall('image'):
        image_name = image_element.get('name')
        image_width = int(image_element.get('width'))
        image_height = int(image_element.get('height'))
        
        image_data = {
            'name': image_name,
            'width': image_width,
            'height': image_height,
            'annotations': []
        }

        for box_element in image_element.findall('box'):
            label = box_element.get('label')
            xtl = float(box_element.get('xtl'))
            ytl = float(box_element.get('ytl'))
            xbr = float(box_element.get('xbr'))
            ybr = float(box_element.get('ybr'))
            
            annotation = {
                'type': 'rectangle',
                'x1': xtl,
                'y1': ytl,
                'x2': xbr,
                'y2': ybr,
                'label': label
            }
            image_data['annotations'].append(annotation)
        output_data['images'].append(image_data)
            
    with open(output_json_path, 'w') as f:
       json.dump(output_data,f, indent=4)

# example usage:
parse_cvat_xml('example.xml', 'cvat_output.json')
```

this script is a basic xml parser for a cvat exported xml file. it outputs a simplified json with the bounding boxes and the associated metadata from the annotation. these two code snippets, just show how flexible the output of the annotation tools can be.

finally, here's a quick example of how to work with the `vatic` json output:

```python
import json
import os

def load_vatic_json(json_file_path):
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    frames = {}
    
    for obj in data:
        if 'frame' in obj:
            frame_number = obj['frame']
            if frame_number not in frames:
                frames[frame_number] = {
                    "frame":frame_number,
                    "annotations": []
                    }
            
            bbox = obj['box']
            x1,y1,x2,y2 = bbox
            
            annotation_type = 'rectangle'
            
            if 'polygon' in obj:
                 annotation_type = 'polygon'
                 points = obj['polygon']
                 frames[frame_number]['annotations'].append({
                     "type":annotation_type,
                     "points":points,
                     "label":obj['label']
                 })
            else:
                  frames[frame_number]['annotations'].append({
                     "type":annotation_type,
                     "x1":x1,
                     "y1":y1,
                     "x2":x2,
                     "y2":y2,
                     "label":obj['label']
                 })
    return frames
# example usage:
frames = load_vatic_json('vatic_example.json')
for key in frames:
    print (f"frame: {key}, annotations: {frames[key]['annotations']}")

```

this snippet parses a vatic json output, organizing the annotations by frame number. that helps when doing video processing.

as for resources, for a more theoretical approach to video annotation, i recommend checking out the papers on the *imageNet* challenge, specifically those that deal with object detection and segmentation, there is a lot of information about techniques used. if you want to get into the weeds of specific algorithms, then you should read the *computer vision algorithms and applications* by richard szeliski. for learning practical computer vision from the bottom up, check out *computer vision: models, learning, and inference* by simon prince. they are comprehensive books that will give you the theory and background. it's not all about annotation tools, it is also about understanding how the models are built and how they process this data, and knowing the format is critical.

that being said, when selecting an annotation tool you also have to consider the project scope, it’s no use to select a tool that is too powerful for your needs, or on the other hand too basic if you want to do something more specific.

in short, pick a tool that suits your needs, is easy to use, outputs data in a format you can use, and provides you with flexibility. a good choice for one project might be horrible for another. in my experience, the right tool can save countless hours and headaches. so choose wisely. i remember on a project we were annotating videos with bounding boxes and i ended up annotating so many frames, that i saw boxes when i closed my eyes, it wasn't pretty, so yes, take some time to select the tool that is more adequate for you.
