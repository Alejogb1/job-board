---
title: "How to create custom data loaders in fastai v2?"
date: "2025-01-30"
id: "how-to-create-custom-data-loaders-in-fastai"
---
The shift from fastai v1 to v2 significantly altered how data loading is handled, moving away from reliance on pre-built `ImageDataBunch` objects and towards a more modular and flexible system. I encountered this firsthand when migrating a complex medical image analysis project that heavily depended on customized data inputs. The core of the change lies in utilizing the `DataBlock` API in conjunction with custom functions and transforms. Understanding this new paradigm allows for precise control over every stage of the data pipeline.

The `DataBlock` class is essentially a configuration engine for data loading. It defines the relationship between the raw data source, the input types (like images, text, or tabular data), and the desired output types (labels).  The process involves specifying several key parameters: `blocks`, `get_items`, `get_y`, `splitter`, and `transforms`. These parameters determine how fastai interprets your dataset.

Firstly, `blocks` defines the input and output type using the `TransformBlock` classes. For example, if dealing with images and segmentation masks, one could use `(ImageBlock, MaskBlock)`. These `TransformBlocks` define the base types and how to load them. It is crucial to select the correct `TransformBlock` for each type of data involved, as this influences how transformations are applied and how data is batched.

Next, `get_items` determines how fastai will gather data file paths or identifiers. This function should return a list of whatever represents a single data example, i.e., filenames, image identifiers, row indices. It typically involves walking a directory structure, reading an index file, or generating a list based on available data. For my project involving volumetric medical images (3D), I found that creating a custom `get_items` that recursively found .nii.gz files was vital.

The `get_y` function, by contrast, defines how to extract labels (or targets) from the items returned by `get_items`. This could mean parsing a filename, reading label data from a separate file, or generating labels based on pre-defined rules. For my segmentation masks, my `get_y` function opened the corresponding mask file and returned it as a `PIL.Image`. It’s often the case that more complex custom code is required here than in `get_items`.

The `splitter` parameter governs how the data is divided into training, validation, and testing sets. Fastai provides built-in splitters like `RandomSplitter` and `IndexSplitter`, but custom splitting logic can also be implemented through a custom function or a `callable`. This was essential for handling our patient-specific data to ensure no patient data leaked between training and validation.

Finally, `transforms` applies all data augmentations and necessary pre-processing steps.  This parameter accepts lists of functions or class-based transforms. The order of these transforms is extremely important; resizing needs to happen before other spatial transformations like random rotations to ensure correct output. I typically place all necessary resizing operations early in this process.

Here are three code examples that illustrate the process with increasing complexity.

**Example 1: Basic Image Classification**

This example assumes a simple image dataset where images are directly under a directory and labels are in subdirectories.

```python
from fastai.data.block import DataBlock
from fastai.vision.data import ImageBlock, CategoryBlock
from pathlib import Path

def get_image_files_and_labels(data_dir):
    """Returns a list of (image path, category label) tuples."""
    image_files = []
    for cat_dir in (data_dir).iterdir():
        if cat_dir.is_dir():
            for file in cat_dir.glob("*.jpg"):
                image_files.append((file, cat_dir.name))
    return image_files


def get_item_func(image_path_and_label):
    return image_path_and_label[0]

def get_label_func(image_path_and_label):
    return image_path_and_label[1]

def train_val_split(items):
  """Splits the items into training and validation."""
  num_train = int(len(items) * 0.8)
  return items[:num_train], items[num_train:]

data_dir = Path("./data/images") # Replace with the path to the image directory

all_items = get_image_files_and_labels(data_dir)

data_block = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_item_func,
    get_y=get_label_func,
    splitter=train_val_split,
    )

dls = data_block.dataloaders(all_items)
```

This code uses the `get_items` and `get_y` functions to access the image path and the associated labels directly from a list of path-label tuples. A custom splitting function `train_val_split` is defined to ensure the data is split deterministically, which is a better practice for experiments.  The `dataloaders` method is then called on the data block object to generate fastai's data loader objects.

**Example 2: Image Segmentation with custom mask loading**

This example deals with a dataset containing images and corresponding segmentation masks. It demonstrates more intricate logic in the `get_y` function to handle custom mask loading.

```python
from fastai.data.block import DataBlock
from fastai.vision.data import ImageBlock, MaskBlock
from fastai.vision.augment import aug_transforms
from pathlib import Path
from PIL import Image
import numpy as np

def get_image_mask_pairs(data_dir):
    image_mask_pairs = []
    for file in (data_dir / "images").glob("*.png"):
        mask_file = (data_dir / "masks" / file.name)
        if mask_file.exists():
          image_mask_pairs.append((file, mask_file))
    return image_mask_pairs

def get_image_from_pair(image_mask_pair):
  return image_mask_pair[0]

def get_mask_from_pair(image_mask_pair):
  mask_path = image_mask_pair[1]
  mask = np.array(Image.open(mask_path))
  return mask

def train_val_split(items):
  """Splits the items into training and validation."""
  num_train = int(len(items) * 0.8)
  return items[:num_train], items[num_train:]

data_dir = Path("./data/segmentation")

all_items = get_image_mask_pairs(data_dir)

data_block = DataBlock(
    blocks=(ImageBlock, MaskBlock),
    get_items=get_image_from_pair,
    get_y=get_mask_from_pair,
    splitter=train_val_split,
    transforms=[*aug_transforms(),]
)
dls = data_block.dataloaders(all_items, bs=32)

```

In this instance, `get_mask_from_pair` opens the mask file as a `PIL.Image` and then converts it to a numpy array, which can then be processed as a segmentation mask by fastai. The augmentations are performed by the `aug_transforms` function which applies standard fastai augmentations. This demonstrates custom loading and the use of transforms.

**Example 3: Volumetric Medical Image Loading**

This final example delves into a more specialized use case: 3D medical images in `.nii.gz` format. It shows how to handle this multi-dimensional data efficiently.

```python
import SimpleITK as sitk
from fastai.data.block import DataBlock
from fastai.vision.data import ImageBlock, MaskBlock
from pathlib import Path
import numpy as np
from fastai.data.transforms import IntToFloatTensor
import torch

def get_nii_pairs(data_dir):
    """Recursively finds all .nii.gz files in the directory."""
    nii_pairs = []
    for file in data_dir.rglob("*.nii.gz"):
        mask_file = Path(str(file).replace("images", "masks"))
        if mask_file.exists():
          nii_pairs.append((file, mask_file))
    return nii_pairs

def get_image(image_mask_pair):
  img_file = image_mask_pair[0]
  img = sitk.ReadImage(str(img_file))
  img_array = sitk.GetArrayFromImage(img)
  return np.expand_dims(img_array.astype(np.float32), 0)  # Add channel dimension

def get_mask(image_mask_pair):
  mask_file = image_mask_pair[1]
  mask = sitk.ReadImage(str(mask_file))
  mask_array = sitk.GetArrayFromImage(mask)
  return np.expand_dims(mask_array.astype(np.int64), 0) # Add channel dimension

def train_val_split(items):
  """Splits the items into training and validation."""
  num_train = int(len(items) * 0.8)
  return items[:num_train], items[num_train:]


data_dir = Path("./data/medical")

all_items = get_nii_pairs(data_dir)

def squeeze_mask(mask):
  return mask.squeeze(0)

data_block = DataBlock(
  blocks=(ImageBlock, MaskBlock(codes=[0, 1])),  # MaskBlock can specify codes
  get_items=get_image,
  get_y=get_mask,
  splitter=train_val_split,
  item_tfms=[IntToFloatTensor(), squeeze_mask]
)
dls = data_block.dataloaders(all_items, bs=1, device=torch.device("cuda:0"))
```

This code uses the SimpleITK library to read `.nii.gz` files. Note that a channel dimension is added before the conversion to numpy, as fastai expects this even for greyscale images. Also, `IntToFloatTensor` is added to `item_tfms` since the image is read as a numpy array and is not processed by the image handling functions that fastai provides. In addition, the segmentation masks are processed and a `squeeze_mask` transform is defined to remove the extra dimension from the masks. This final example highlights the flexibility of the `DataBlock` and how one can incorporate existing tools. The data loader is set to use a specific GPU device, a detail important for systems with multiple GPUs.

For further learning, I would recommend focusing on the fastai documentation, specifically the `DataBlock` API and the source code of the `TransformBlock` classes.  Additionally, experimenting with custom transforms in the `item_tfms` and `batch_tfms` parameters, especially in complex settings like segmentation with heterogeneous label classes, can provide a much more detailed understanding of the inner workings of the data pipeline. Studying examples from the fastai forums and community projects can show real-world uses of custom data loaders. The fastai library's examples provide valuable insights that are difficult to gain from just looking at the documentation alone.
