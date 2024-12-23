---
title: "How do I use val.py to calculate mAP on a custom testing dataset of YOLOv5?"
date: "2024-12-23"
id: "how-do-i-use-valpy-to-calculate-map-on-a-custom-testing-dataset-of-yolov5"
---

Let's dive into this, shall we? I remember a project back in '21 involving a highly specific object detection task for an autonomous inspection system; we had to roll our own validation and, well, *val.py* in YOLOv5 was our go-to. Getting it to play nicely with custom data can be a bit nuanced, so let me walk you through the process, focusing on clarity and pragmatism.

Essentially, *val.py*, as you likely know, is the script within the YOLOv5 repository designed for evaluating a trained model. It's primarily built for COCO dataset validation, but its architecture is flexible enough to handle custom datasets. The key lies in properly configuring the script to understand the structure and format of *your* annotations and images. We're not modifying the core validation logic itself—rather, we're providing the necessary context.

The crux of the issue often boils down to the `--data` flag within *val.py*. Instead of pointing to a default COCO dataset configuration file, you need to direct it to a yaml file that describes *your* dataset. This yaml file needs to specify:

1.  **The location of your training, validation (and optionally, test) image directories.** This is crucial; *val.py* will be attempting to load image paths from these locations.
2.  **The paths to your annotation files.** YOLOv5 expects these annotations in a specific format (one text file per image, each line defining a bounding box), usually under the "labels" directory.
3.  **The number of classes, and optionally, a mapping of indices to class names.**

Now, let's illustrate with an example. Suppose your dataset is structured like this:

```
my_dataset/
├── images/
│   ├── train/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── val/
│   │   ├── image3.jpg
│   │   ├── image4.jpg
│   │   └── ...
│   └── test/   <- (optional, if used for final evaluation)
│       ├── image5.jpg
│       ├── image6.jpg
│       └── ...
└── labels/
    ├── train/
    │   ├── image1.txt
    │   ├── image2.txt
    │   └── ...
    ├── val/
    │   ├── image3.txt
    │   ├── image4.txt
    │   └── ...
    └── test/    <- (optional, if used for final evaluation)
        ├── image5.txt
        ├── image6.txt
        └── ...
```

Your `data.yaml` file, which you'd pass via the `--data` flag, could look like this:

```yaml
train: my_dataset/images/train  # train images
val: my_dataset/images/val   # val images
test: my_dataset/images/test  # test images (optional)

nc: 3 # number of classes.

names: ['class_a', 'class_b', 'class_c'] #optional.
```

**Code Snippet 1: Creating a data.yaml file**

Here's a python script I've used in the past to generate this yaml file programmatically (useful if the paths or class names are dynamic):

```python
import yaml

def create_data_yaml(train_path, val_path, test_path, num_classes, class_names, output_path="data.yaml"):
    data = {
        'train': train_path,
        'val': val_path,
        'test': test_path,
        'nc': num_classes,
        'names': class_names
    }

    with open(output_path, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)

if __name__ == "__main__":
    train_path = "my_dataset/images/train"
    val_path = "my_dataset/images/val"
    test_path = "my_dataset/images/test"
    num_classes = 3
    class_names = ["class_a", "class_b", "class_c"]
    create_data_yaml(train_path, val_path, test_path, num_classes, class_names)
```

After creating `data.yaml` (and ensuring the labels are correctly formatted according to the YOLOv5 specification—normalized bounding box coordinates and class index), you can then execute *val.py*.

The basic command, assuming your trained model is located at `runs/train/exp/weights/best.pt`, would be:

```bash
python val.py --data data.yaml --weights runs/train/exp/weights/best.pt --task test --batch-size 16
```

The `--task test` flag will use the `test` dataset defined in your `data.yaml`. If omitted or changed to `val`, it uses the `val` dataset. The batch-size will depend on the resources you have and the size of images.

The script will then compute mAP, precision, recall, and other relevant metrics on *your* dataset.

Now, consider a scenario where your annotations are in a different format, for instance, a CSV file. While it's not the standard for YOLOv5, you can pre-process the CSV file into the YOLOv5 expected text format before running *val.py*. This was a common issue we encountered when ingesting annotations from legacy systems.

**Code Snippet 2: CSV Annotation Preprocessing to YOLOv5 Format**

Here’s an example of converting a hypothetical CSV annotation file to YOLOv5 format:

```python
import csv
import os

def convert_csv_to_yolov5(csv_path, output_dir):
    with open(csv_path, 'r') as file:
        reader = csv.reader(file)
        next(reader) # Skip header
        for row in reader:
            image_name = row[0] #assuming the image name is in the first column
            class_id = int(row[1])  #assuming the class id is in the second column
            x_min = float(row[2]) # bounding box values
            y_min = float(row[3])
            x_max = float(row[4])
            y_max = float(row[5])

            # Normalizing to YOLOv5
            image_width = int(row[6])  #assuming image width is in the seventh column
            image_height = int(row[7])#assuming image height is in the eighth column

            x_center = (x_min + x_max) / 2 / image_width
            y_center = (y_min + y_max) / 2 / image_height
            width = (x_max - x_min) / image_width
            height = (y_max - y_min) / image_height

            output_file_name = os.path.splitext(image_name)[0] + ".txt"
            output_file_path = os.path.join(output_dir, output_file_name)

            with open(output_file_path, 'w') as outfile:
                outfile.write(f"{class_id} {x_center} {y_center} {width} {height}\n")


if __name__ == "__main__":
     csv_path = 'annotations.csv'  # Path to your CSV file
     output_dir = 'my_dataset/labels/test' # Path to where you want the resulting .txt annotation files
     if not os.path.exists(output_dir):
            os.makedirs(output_dir)
     convert_csv_to_yolov5(csv_path, output_dir)
```

Finally, let's tackle the situation where your images might have different resolutions than those used in training. *val.py* will resize the images, according to the model input dimension, before evaluation. To influence this process, you can specify the `--img` flag followed by the desired input dimension. For example, using  `--img 640`, will evaluate the model on resized images of `640 x 640`.

**Code Snippet 3: Running val.py with custom input resolution**

Here's the command for that scenario:

```bash
python val.py --data data.yaml --weights runs/train/exp/weights/best.pt --task test --img 640 --batch-size 16
```

I've learned through hard experience that carefully reviewing the output of `val.py`, especially when dealing with custom data, is essential. Be sure to check the printed metrics and also understand the details of how mAP is calculated (typically, a precision-recall curve is involved)

For a deeper dive into the mathematics of mean Average Precision (mAP), and understanding the evaluation metrics in object detection, I’d suggest resources like “Computer Vision: Algorithms and Applications” by Richard Szeliski and papers like "The Pascal Visual Object Classes (VOC) challenge" and the COCO paper "Microsoft COCO: Common Objects in Context". These resources offer a solid theoretical foundation. Also, browsing through the YOLOv5 documentation and community forum on GitHub can be quite helpful.

In my experience, focusing on the fundamentals of data format, file organization, and pre-processing steps goes a long way. Hope this gives you a clearer path forward with your mAP calculation in YOLOv5. Good luck.
