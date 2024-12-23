---
title: "How do I get mAP for a custom test set with val.py in YOLOv5?"
date: "2024-12-23"
id: "how-do-i-get-map-for-a-custom-test-set-with-valpy-in-yolov5"
---

Okay, let's tackle this. I've certainly been down this road more than once—modifying YOLOv5's `val.py` to evaluate custom datasets, especially when you need to go beyond the default validation split. It's a common enough scenario, and while the basic mechanics are there, tweaking it requires a clear understanding of the process. So, let’s break it down and I'll share some of what I've learned through experience.

First off, the core issue you're facing is that `val.py` by default looks to the data configuration file (like `data/coco128.yaml` or similar) to define the path to your validation data. We want to override that, pointing it instead to our bespoke test dataset, while keeping the model and other parameters the same. This involves a bit of direct intervention in the script's logic. It's not a terribly difficult fix, but understanding the rationale behind the modification is key. The default setup is efficient for standard cases, but real-world projects invariably require flexibility like this.

Let’s consider that the typical project structure assumes your images and annotations live in folders named based on the dataset split. For example:

```
dataset/
  images/
    train/
      image1.jpg
      image2.jpg
      ...
    val/
      image3.jpg
      image4.jpg
      ...
    test/
      image5.jpg
      image6.jpg
      ...
  labels/
    train/
      image1.txt
      image2.txt
      ...
    val/
      image3.txt
      image4.txt
      ...
    test/
      image5.txt
      image6.txt
      ...
```

The `val.py` script uses the `dataset.yaml` file to locate the `val` split. We will, in essence, be bypassing this behavior.

Here’s the first critical modification we will make. I will show this in a Python snippet. Instead of relying on the parsed arguments defined by argparse, we’ll supply a hardcoded dataset path. This means we won't be changing `val.py`'s original functionality but will be adding a specialized branch for custom testing.

```python
# Modified val.py - First Snippet

import argparse
import os
import yaml # to read .yaml configuration files
from pathlib import Path

import torch
from models.common import DetectMultiBackend
from utils.dataloaders import create_dataloader
from utils.general import (LOGGER, check_dataset, check_img_size,
                           check_requirements, colorstr, increment_path,
                           non_max_suppression, print_args, scale_coords,
                           strip_optimizer, xyxy2xywh)
from utils.metrics import ConfusionMatrix, ap_per_class, box_iou
from utils.plots import output_to_target, plot_images, plot_val_study
from utils.torch_utils import select_device, time_sync

def run_validation(weights, data_path, img_size, batch_size, device, single_cls, model):
    # Load the model and its parameters if the model is None
    if model is None:
         model = DetectMultiBackend(weights, device=device, dnn=False, data=data_path, fp16=False)
         model.half()
    stride, names = model.stride, model.names
    imgsz = check_img_size(img_size, s=stride)  # check image size
    dataset = create_dataloader(
        Path(data_path).parent / 'images',
        path = Path(data_path).parent / 'labels',
        imgsz,
        batch_size,
        stride,
        single_cls=single_cls,
        pad=0.5,
        rect=True,
        prefix=colorstr('val: ')
        )[0]
    # Validate and get the results, we need the model for the forward pass
    results = validate(model=model, data_loader=dataset, device=device)
    return results # return the results object for later use


def validate(model, data_loader, device):
    # original code for validating
    seen = 0
    confusion_matrix = ConfusionMatrix(nc=len(model.names))
    s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Instances', 'P', 'R', 'mAP50', 'mAP50-95')
    dt = [0, 0, 0]
    model.eval()
    jdict = []
    for batch_i, (im, targets, paths, shapes) in enumerate(data_loader):
        t1 = time_sync()
        im = im.to(device, non_blocking=True)
        targets = targets.to(device)
        im = im.half() if model.fp16 else im.float()
        im /= 255
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        with torch.no_grad():
            preds = model(im)
        t3 = time_sync()
        dt[1] += t3 - t2
        # NMS
        preds = non_max_suppression(preds, 0.001, 0.6, labels=[], multi_label=True)

        # Evaluation
        for si, pred in enumerate(preds):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tbox = xyxy2xywh(labels[:, 0:4]) if nl else torch.zeros((0, 4))
            labelsn = torch.cat((labels[:, 4:], tbox), dim=1).cpu()

            if len(pred):
                # Predictions
                predn = pred.clone()
                scale_coords(im[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])
                predn[:, :4] = xyxy2xywh(predn[:, :4])
                # Output to target format
                output = output_to_target(predn)
                jdict.extend(output) # Add to jdict for coco mAP
                # Metrics
                confusion_matrix.process_batch(predn[:, 5].int(), labelsn[:, 0].int())
            seen += 1
            # mAP is done in this loop
    # Metrics
    stats = confusion_matrix.get_matrix()
    # get ap per class from ground truth and predictions
    ap, ap_class = ap_per_class(*stats, plot=False, save_dir=None, names=model.names)
    # The return from ap_per_class gives a mAP object
    ap50, ap = ap[:,0],ap[:,1]

    LOGGER.info(s)
    LOGGER.info(('%20s' + '%11.3g' * 6) % ( 'all', seen, stats[1].sum(), *ap.mean(0), *ap50.mean(0) ))

    return { 'mAP': ap.mean(0).item(), 'mAP50': ap50.mean(0).item()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='weights path') # path to the weights file
    parser.add_argument('--data_path', type=str, default='./dataset/test/', help='path to test folder (images + labels)') # this is the root of the data
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size h,w') # image size
    parser.add_argument('--batch-size', type=int, default=32, help='batch size') # batch size
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu') # the device for inference
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')  # treat as single-class dataset
    args = parser.parse_args()

    # --- New Logic Here ---
    # Pass the values to the running validation method
    device = select_device(args.device)
    results = run_validation(args.weights, args.data_path, args.imgsz, args.batch_size, device, args.single_cls, None)

    print(f"mAP on custom dataset: {results['mAP']:.3f}")
    print(f"mAP50 on custom dataset: {results['mAP50']:.3f}")

```

In this first modification, I've streamlined the main script to directly handle custom data. Notice the `run_validation` method which takes the test data path and processes it. The validation function which contains all the logic for evaluating the model, calculating the mAP scores and returning the result is in a separate method. All command line arguments are still valid; however, `val.py` will not load dataset configurations from a file. Instead, it will take the passed in `data_path`, which should be the path to the test set.

This snippet makes no changes to the original `val.py`, but it adds the functionality to run a test on a custom dataset.

Next, let’s look at how to customize the script to work with a custom yaml file. It would be good practice to load the custom test set using `yaml` rather than hardcode the `data_path`.

```python
# Modified val.py - Second Snippet

import argparse
import os
import yaml # to read .yaml configuration files
from pathlib import Path

import torch
from models.common import DetectMultiBackend
from utils.dataloaders import create_dataloader
from utils.general import (LOGGER, check_dataset, check_img_size,
                           check_requirements, colorstr, increment_path,
                           non_max_suppression, print_args, scale_coords,
                           strip_optimizer, xyxy2xywh)
from utils.metrics import ConfusionMatrix, ap_per_class, box_iou
from utils.plots import output_to_target, plot_images, plot_val_study
from utils.torch_utils import select_device, time_sync

def run_validation(weights, data_path, img_size, batch_size, device, single_cls, model):
        # Load the model and its parameters if the model is None
        if model is None:
           model = DetectMultiBackend(weights, device=device, dnn=False, data=data_path, fp16=False)
           model.half()
        stride, names = model.stride, model.names
        imgsz = check_img_size(img_size, s=stride)  # check image size

        with open(data_path, 'r') as f: # Open custom .yaml file
            data = yaml.safe_load(f)
        test_path = data['test']
        dataset = create_dataloader(
            Path(test_path).parent / 'images',
            path = Path(test_path).parent / 'labels',
            imgsz,
            batch_size,
            stride,
            single_cls=single_cls,
            pad=0.5,
            rect=True,
            prefix=colorstr('val: ')
            )[0]
        # Validate and get the results, we need the model for the forward pass
        results = validate(model=model, data_loader=dataset, device=device)
        return results # return the results object for later use


def validate(model, data_loader, device):
    # original code for validating
    seen = 0
    confusion_matrix = ConfusionMatrix(nc=len(model.names))
    s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Instances', 'P', 'R', 'mAP50', 'mAP50-95')
    dt = [0, 0, 0]
    model.eval()
    jdict = []
    for batch_i, (im, targets, paths, shapes) in enumerate(data_loader):
        t1 = time_sync()
        im = im.to(device, non_blocking=True)
        targets = targets.to(device)
        im = im.half() if model.fp16 else im.float()
        im /= 255
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        with torch.no_grad():
            preds = model(im)
        t3 = time_sync()
        dt[1] += t3 - t2
        # NMS
        preds = non_max_suppression(preds, 0.001, 0.6, labels=[], multi_label=True)

        # Evaluation
        for si, pred in enumerate(preds):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tbox = xyxy2xywh(labels[:, 0:4]) if nl else torch.zeros((0, 4))
            labelsn = torch.cat((labels[:, 4:], tbox), dim=1).cpu()

            if len(pred):
                # Predictions
                predn = pred.clone()
                scale_coords(im[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])
                predn[:, :4] = xyxy2xywh(predn[:, :4])
                # Output to target format
                output = output_to_target(predn)
                jdict.extend(output) # Add to jdict for coco mAP
                # Metrics
                confusion_matrix.process_batch(predn[:, 5].int(), labelsn[:, 0].int())
            seen += 1
            # mAP is done in this loop
    # Metrics
    stats = confusion_matrix.get_matrix()
    # get ap per class from ground truth and predictions
    ap, ap_class = ap_per_class(*stats, plot=False, save_dir=None, names=model.names)
    # The return from ap_per_class gives a mAP object
    ap50, ap = ap[:,0],ap[:,1]

    LOGGER.info(s)
    LOGGER.info(('%20s' + '%11.3g' * 6) % ( 'all', seen, stats[1].sum(), *ap.mean(0), *ap50.mean(0) ))

    return { 'mAP': ap.mean(0).item(), 'mAP50': ap50.mean(0).item()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='weights path') # path to the weights file
    parser.add_argument('--data_path', type=str, default='./dataset.yaml', help='path to dataset.yaml') # this is the root of the data
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size h,w') # image size
    parser.add_argument('--batch-size', type=int, default=32, help='batch size') # batch size
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu') # the device for inference
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')  # treat as single-class dataset
    args = parser.parse_args()

    # --- New Logic Here ---
    # Pass the values to the running validation method
    device = select_device(args.device)
    results = run_validation(args.weights, args.data_path, args.imgsz, args.batch_size, device, args.single_cls, None)

    print(f"mAP on custom dataset: {results['mAP']:.3f}")
    print(f"mAP50 on custom dataset: {results['mAP50']:.3f}")
```

In this second example, I've modified the `run_validation` method to open the `data_path` (which we are now passing the dataset yaml file) and look for the `test` tag, which is used to locate the correct testing data. The rest of the script is the same, except that we now pass the dataset file to `run_validation` rather than the test data path. In your `dataset.yaml` file you would add:
```yaml
test: "./dataset/test/"
```

Now, let’s say you don't want to use folders, instead, you have a `.txt` file that contains a list of your custom dataset files. This third snippet will show you how to modify the code to work with this situation.

```python
# Modified val.py - Third Snippet
import argparse
import os
import yaml
from pathlib import Path

import torch
from models.common import DetectMultiBackend
from utils.dataloaders import create_dataloader
from utils.general import (LOGGER, check_dataset, check_img_size,
                           check_requirements, colorstr, increment_path,
                           non_max_suppression, print_args, scale_coords,
                           strip_optimizer, xyxy2xywh)
from utils.metrics import ConfusionMatrix, ap_per_class, box_iou
from utils.plots import output_to_target, plot_images, plot_val_study
from utils.torch_utils import select_device, time_sync
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import glob

class CustomDataset(Dataset):
    def __init__(self, image_list, label_dir, img_size, transform=None):
        self.image_list = image_list
        self.label_dir = label_dir
        self.img_size = img_size
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        label_path = os.path.join(self.label_dir, os.path.basename(img_path).replace(Path(img_path).suffix, '.txt'))

        img = Image.open(img_path).convert('RGB')
        img = img.resize((self.img_size,self.img_size), Image.Resampling.LANCZOS)
        img = np.asarray(img).transpose((2,0,1))
        img = torch.from_numpy(img)

        if os.path.exists(label_path):
            labels = np.loadtxt(label_path).reshape(-1,5)
            # Convert xywh normalized to xyxy scaled to the size of the image
            labels[:, 1:5] = self.xywh2xyxy(torch.tensor(labels[:, 1:5]), self.img_size)
            labels = torch.from_numpy(labels)
        else:
            labels = torch.zeros(size=(0,5)) # Return empty tensors if no labels exist
        
        return img.float()/255, labels, str(img_path), [self.img_size,self.img_size]


    def xywh2xyxy(self, x, size):
        y = x.clone()
        y[:,0] = x[:,0] * size - x[:,2] * size / 2
        y[:,1] = x[:,1] * size - x[:,3] * size / 2
        y[:,2] = x[:,0] * size + x[:,2] * size / 2
        y[:,3] = x[:,1] * size + x[:,3] * size / 2
        return y
    
def run_validation(weights, data_path, img_size, batch_size, device, single_cls, model):
        # Load the model and its parameters if the model is None
        if model is None:
           model = DetectMultiBackend(weights, device=device, dnn=False, data=data_path, fp16=False)
           model.half()
        stride, names = model.stride, model.names
        imgsz = check_img_size(img_size, s=stride)  # check image size

        with open(data_path, 'r') as f: # open image list file
            image_list = f.readlines()
            image_list = [x.strip() for x in image_list]
        label_dir = Path(image_list[0]).parent.parent / "labels"

        dataset = CustomDataset(image_list, label_dir, imgsz)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        # Validate and get the results, we need the model for the forward pass
        results = validate(model=model, data_loader=dataloader, device=device)
        return results # return the results object for later use


def validate(model, data_loader, device):
    # original code for validating
    seen = 0
    confusion_matrix = ConfusionMatrix(nc=len(model.names))
    s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Instances', 'P', 'R', 'mAP50', 'mAP50-95')
    dt = [0, 0, 0]
    model.eval()
    jdict = []
    for batch_i, (im, targets, paths, shapes) in enumerate(data_loader):
        t1 = time_sync()
        im = im.to(device, non_blocking=True)
        targets = targets.to(device)
        im = im.half() if model.fp16 else im.float()
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        with torch.no_grad():
            preds = model(im)
        t3 = time_sync()
        dt[1] += t3 - t2
        # NMS
        preds = non_max_suppression(preds, 0.001, 0.6, labels=[], multi_label=True)

        # Evaluation
        for si, pred in enumerate(preds):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tbox = xyxy2xywh(labels[:, 0:4]) if nl else torch.zeros((0, 4))
            labelsn = torch.cat((labels[:, 4:], tbox), dim=1).cpu()

            if len(pred):
                # Predictions
                predn = pred.clone()
                scale_coords(im[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])
                predn[:, :4] = xyxy2xywh(predn[:, :4])
                # Output to target format
                output = output_to_target(predn)
                jdict.extend(output) # Add to jdict for coco mAP
                # Metrics
                confusion_matrix.process_batch(predn[:, 5].int(), labelsn[:, 0].int())
            seen += 1
            # mAP is done in this loop
    # Metrics
    stats = confusion_matrix.get_matrix()
    # get ap per class from ground truth and predictions
    ap, ap_class = ap_per_class(*stats, plot=False, save_dir=None, names=model.names)
    # The return from ap_per_class gives a mAP object
    ap50, ap = ap[:,0],ap[:,1]

    LOGGER.info(s)
    LOGGER.info(('%20s' + '%11.3g' * 6) % ( 'all', seen, stats[1].sum(), *ap.mean(0), *ap50.mean(0) ))

    return { 'mAP': ap.mean(0).item(), 'mAP50': ap50.mean(0).item()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='weights path') # path to the weights file
    parser.add_argument('--data_path', type=str, default='./image_list.txt', help='path to test images') # this is the root of the data
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size h,w') # image size
    parser.add_argument('--batch-size', type=int, default=32, help='batch size') # batch size
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu') # the device for inference
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')  # treat as single-class dataset
    args = parser.parse_args()

    # --- New Logic Here ---
    # Pass the values to the running validation method
    device = select_device(args.device)
    results = run_validation(args.weights, args.data_path, args.imgsz, args.batch_size, device, args.single_cls, None)

    print(f"mAP on custom dataset: {results['mAP']:.3f}")
    print(f"mAP50 on custom dataset: {results['mAP50']:.3f}")
```

In this third example, I've added a custom dataloader that allows reading the image paths from a list. The constructor of the dataloader will then read the corresponding label files (if they exist). All this is done while avoiding changes to the core `val.py` code, as that would be cumbersome. The important part of this is the `__getitem__` method of `CustomDataset`, where I read each image from the list of images and labels from the corresponding label file, as well as implement the correct size transformations. Notice the `run_validation` method is modified to open a text file and use that file to initialize the dataloader and pass it to the `validate` function. In your `image_list.txt` file you would add:

```text
./dataset/test/images/image1.jpg
./dataset/test/images/image2.jpg
./dataset/test/images/image3.jpg
...
```

To further your understanding of the underlying mechanics of object detection metrics, I highly recommend reading the original papers on the PASCAL VOC challenge (Everingham et al., 2010) and the COCO dataset (Lin et al., 2014) for context on mAP calculation, and specifically papers by Everingham, M., Van Gool, L., Williams, C. K., & Zisserman, A. on VOC challenge evaluation. Additionally, the PyTorch documentation on data loading (and it's excellent tutorials) and neural network evaluation is quite valuable to understand the code logic.

In conclusion, by making these targeted changes to `val.py`, you can effectively evaluate your custom datasets without needing to deeply alter the original script. The key is to understand how the dataloaders work within YOLOv5 and to modify that path to get the desired outcome.
