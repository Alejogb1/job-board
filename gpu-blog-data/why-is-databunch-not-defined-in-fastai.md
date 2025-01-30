---
title: "Why is `DataBunch` not defined in Fastai?"
date: "2025-01-30"
id: "why-is-databunch-not-defined-in-fastai"
---
The absence of `DataBunch` as a standalone class in recent versions of Fastai arises from a significant architectural shift towards a more modular and flexible data handling system. Specifically, the responsibilities formerly attributed to `DataBunch` have been redistributed across several core components, notably `DataLoaders`. This restructuring provides granular control over data processing and pipeline construction, moving away from a monolithic approach.

In earlier iterations of Fastai, `DataBunch` served as a central container, encapsulating training, validation, and optionally test datasets, along with associated data loaders. It handled batching, shuffling, and any required transformations. The design, while initially convenient, proved somewhat limiting when confronted with diverse data modalities and complex pre-processing requirements. The monolithic nature made it challenging to inject custom behavior without modifying existing class structures. This is the primary reason why you won't find `DataBunch` defined in current Fastai.

The current approach, centered around `DataLoaders`, offers several advantages. Firstly, it enables greater composability. The various aspects of data handling are now implemented in separate, reusable classes, allowing for easier extension and modification. For instance, data augmentation is handled independently from data loading, allowing users to mix and match various techniques seamlessly. Secondly, the new architecture enhances flexibility. Users can now specify custom data loading logic using `Dataset` and `DataLoader` classes provided by PyTorch directly, further customized by callbacks within Fastai. This enables support for diverse data formats and more complex scenarios, including custom data pipelines. Lastly, the transition improves code clarity and maintainability; the more explicit relationships between individual components are easier to follow, especially when debugging more intricate setups.

The fundamental building blocks for the current Fastai data pipeline are `Datasets` and `DataLoaders`, echoing the PyTorch framework. A `Dataset` is any object that returns a single item from a dataset given an index, typically paired with its ground truth label, while a `DataLoader` encapsulates an iterator that yields batches of these items, with all necessary transformations applied. Fastai provides several utility functions for constructing common datasets and dataloaders, including those for image classification, language modeling, and tabular data.

To illustrate how this structure is utilized, let's consider three scenarios: image classification, text classification, and tabular regression, showing how we might create a dataloader for each.

**Example 1: Image Classification**

Assume we have an image dataset organized in a directory structure where subdirectories represent different classes. Here's how we would build a dataloader for it:

```python
from fastai.vision.all import *

def create_image_dataloaders(path, size=224, batch_size=64, valid_pct=0.2):
    """Creates DataLoaders for image classification."""
    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=RandomSplitter(valid_pct=valid_pct, seed=42),
        get_y=parent_label,
        item_tfms=Resize(size),
        batch_tfms=aug_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05)
    )
    return dblock.dataloaders(path, bs=batch_size)

# Assume a 'images' directory with subfolders of class labels
path_to_images = Path('./images')
image_dls = create_image_dataloaders(path_to_images)

# Usage
# image_dls.show_batch()
# next(iter(image_dls.train)).shape
```

Here, we use `DataBlock` as a declarative way to specify the entire data pipeline. `get_items` identifies all image files, `splitter` creates a training and validation set, `get_y` extracts labels from the parent directory names, `item_tfms` resizes images, and `batch_tfms` applies augmentations. The final `dataloaders` method creates a `DataLoaders` instance containing both training and validation loaders from this `DataBlock` setup. Notice no explicit `DataBunch` creation is required.

**Example 2: Text Classification**

Now, let's look at creating a dataloader for a text classification task. Assume we have a CSV file with text and corresponding labels:

```python
from fastai.text.all import *
import pandas as pd

def create_text_dataloaders(csv_path, text_col, label_col, batch_size=64, valid_pct=0.2):
    """Creates DataLoaders for text classification from a CSV."""
    df = pd.read_csv(csv_path)
    dblock = DataBlock(
        blocks=(TextBlock.from_df(text_col, seq_len=72), CategoryBlock),
        get_x=ColReader(text_col),
        get_y=ColReader(label_col),
        splitter=RandomSplitter(valid_pct=valid_pct, seed=42),
    )
    return dblock.dataloaders(df, bs=batch_size)


# Assume data.csv has text column named 'text' and label column named 'label'
csv_path = './data.csv'
text_dls = create_text_dataloaders(csv_path, text_col='text', label_col='label')

# Usage
# text_dls.show_batch()
# next(iter(text_dls.train))
```

In this example, `TextBlock.from_df` handles text tokenization and numericalization within the `DataBlock`. `ColReader` extracts text and labels from specified columns. Again, no manual `DataBunch` instantiation is needed; the `dataloaders` method generates the required dataloaders directly.

**Example 3: Tabular Regression**

Finally, consider a tabular regression problem, where we have a CSV with numerical and categorical features.

```python
from fastai.tabular.all import *
import pandas as pd

def create_tabular_dataloaders(csv_path, dep_var, cat_names, cont_names, batch_size=64, valid_pct=0.2):
    """Creates DataLoaders for tabular regression from a CSV."""
    df = pd.read_csv(csv_path)
    dblock = DataBlock(
        blocks=(TabularBlock(cat_names=cat_names, cont_names=cont_names), RegressionBlock),
        get_x=ColSplitter(),
        get_y=ColReader(dep_var),
        splitter=RandomSplitter(valid_pct=valid_pct, seed=42),
    )
    return dblock.dataloaders(df, bs=batch_size)

# Assume data_table.csv has feature columns 'feature1','feature2'
#   categorical 'category1' and a dependent variable 'target'
csv_path_table = './data_table.csv'
cat_features = ['category1']
cont_features = ['feature1', 'feature2']
dependent_var = 'target'

table_dls = create_tabular_dataloaders(csv_path_table,
    dep_var=dependent_var, cat_names=cat_features, cont_names=cont_features
)

# Usage
# table_dls.show_batch()
# next(iter(table_dls.train))
```

Here, `TabularBlock` automatically handles encoding of categorical and normalization of continuous features. `ColSplitter` ensures proper separation between feature columns and dependent variable. The `dataloaders` method, as before, returns our training and validation dataloaders ready for model training without explicit `DataBunch` construction.

The shift to `DataLoaders` marks a maturation of Fastai's data processing pipeline. This change provides a more flexible, powerful, and composable approach to data handling. Users are expected to create instances of `DataLoaders` through the `DataBlock` API or directly using PyTorch’s own `Dataset` and `DataLoader` classes, customized using Fastai callbacks.

For more information regarding these concepts, I would recommend consulting the official Fastai documentation, which includes extensive guides on data loading and processing. In particular, focus on the sections covering `DataBlock`, `DataLoaders`, and individual block types for various data modalities. Explore example notebooks on various tasks, like image classification, text classification, and tabular data analysis, which demonstrate how these components interact in a real context. These resources should clarify how `DataLoaders` have superseded `DataBunch` in the current Fastai architecture. Finally, experimenting with building custom `DataBlock` configurations will solidify your understanding of the new system.
