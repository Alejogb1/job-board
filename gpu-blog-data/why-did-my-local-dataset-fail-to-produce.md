---
title: "Why did my local dataset fail to produce examples using the tfds DIV2K official script?"
date: "2025-01-30"
id: "why-did-my-local-dataset-fail-to-produce"
---
The TensorFlow Datasets (tfds) DIV2K official script, when used with a local dataset, often fails due to mismatches in expected file structure or insufficient metadata, particularly concerning the required HR/LR image pairings. I've personally encountered this issue on several occasions while building custom pipelines for super-resolution research. The problem usually stems from the fact that the script is designed to parse the specific directory structure and naming conventions expected for the standard DIV2K dataset downloaded from a public source. When attempting to adapt it to locally stored, user-organized data, these assumptions can break down.

The primary source of failure is the reliance of the tfds DIV2K script on the presence of specific subdirectories labeled 'HR' and 'LR' containing corresponding high-resolution and low-resolution images, respectively. Further, these images are expected to adhere to a specific naming pattern for proper pairing.  For example, '0001.png' in the 'HR' folder corresponds to '0001x2.png', '0001x3.png', '0001x4.png' within the 'LR' folder (representing different downscaling factors). Deviations from this structure, even slight variations in filename extensions or the absence of specific scaling factor folders, will cause the tfds dataset generator to fail. The script uses string manipulation and pattern matching internally to create the necessary pairing information which becomes impossible with mismatched data structures. The issue isn't fundamentally about the data itself, but about its structural presentation to the `tfds.load` function.

Let me elaborate through some code examples, illustrating common problems and their solutions. Assume you have a local dataset following what you believe to be the right format, but `tfds.load` throws errors.

**Example 1: Missing Subdirectories**

Suppose the file structure looks like this:

```
my_div2k/
    images/
        0001.png  (HR image)
        0001x2.png (LR image, x2 downsampled)
        0001x3.png (LR image, x3 downsampled)
        0001x4.png (LR image, x4 downsampled)
        0002.png
        0002x2.png
        ...
```

The standard `tfds` script will fail here because it expects the data to be organised into `HR` and `LR` folders. The following snippet demonstrates how to define the dataset using the `builder` property to address this problem, reorganizing the path definitions dynamically.

```python
import tensorflow_datasets as tfds
import tensorflow as tf
import os

class MyDIV2KConfig(tfds.core.BuilderConfig):
    """Configuration for local DIV2K-like dataset."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class MyDIV2K(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for my custom DIV2K style data."""

    VERSION = tfds.core.Version("1.0.0")
    BUILDER_CONFIGS = [
      MyDIV2KConfig(
          name="default",
          description="My custom DIV2K-like dataset",
      ),
    ]

    def _info(self):
      return tfds.core.DatasetInfo(
        builder=self,
        description="My custom DIV2K-like dataset.",
        features=tfds.features.FeaturesDict({
          "hr": tfds.features.Image(),
          "lr": tfds.features.Image(),
          "lr_x2": tfds.features.Image(),
          "lr_x3": tfds.features.Image(),
          "lr_x4": tfds.features.Image(),
        }),
      )

    def _split_generators(self, dl_manager):
      return {
          "train": self._generate_examples(dl_manager),
      }

    def _generate_examples(self, dl_manager):
        images_path = os.path.join(dl_manager.manual_dir, 'images')
        for filename in os.listdir(images_path):
            if filename.endswith('.png') and not ('x2' in filename or 'x3' in filename or 'x4' in filename): #only process HR files
                hr_path = os.path.join(images_path, filename)
                base_name = os.path.splitext(filename)[0]
                lr_path_x2 = os.path.join(images_path, f'{base_name}x2.png')
                lr_path_x3 = os.path.join(images_path, f'{base_name}x3.png')
                lr_path_x4 = os.path.join(images_path, f'{base_name}x4.png')
                if os.path.exists(lr_path_x2) and os.path.exists(lr_path_x3) and os.path.exists(lr_path_x4):
                  yield base_name, {
                    "hr": hr_path,
                    "lr": lr_path_x2,
                    "lr_x2":lr_path_x2,
                    "lr_x3":lr_path_x3,
                    "lr_x4":lr_path_x4,
                   }


if __name__ == '__main__':
  manual_dir = 'my_div2k' #path where your data is located
  builder = MyDIV2K(data_dir='tmp') #this is where the downloaded dataset would be stored
  dataset = builder.as_dataset(split='train', manual_dir=manual_dir)

  for example in dataset.take(1):
        print(example['hr'].shape)
        print(example['lr'].shape)
```

In this example, we define a custom dataset builder `MyDIV2K` that inherits from `tfds.core.GeneratorBasedBuilder`. I have implemented the `_generate_examples` method. This method first constructs the directory of the dataset, then iterates through the HR images. Based on the HR image filename, it constructs the corresponding low resolution paths. This logic replaces the internal logic of the DIV2K script, which would have assumed a different structure. The `manual_dir` argument during loading allows us to point `tfds` to our custom data.

**Example 2: Incorrect Filename Extension**

Consider another common scenario, where your files might have JPEG extensions instead of PNGs:

```
my_div2k/
    HR/
        0001.jpg
        0002.jpg
        ...
    LR/
        0001x2.jpg
        0001x3.jpg
        0001x4.jpg
        0002x2.jpg
        ...

```

The standard script relies on the `.png` extension during file path construction. Here's a solution using string replacement during processing within a custom generator (again using a builder):

```python
import tensorflow_datasets as tfds
import tensorflow as tf
import os

class MyDIV2KConfigJPEG(tfds.core.BuilderConfig):
    """Configuration for local DIV2K-like dataset with jpeg extensions."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class MyDIV2KJPEG(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for my custom DIV2K style data using JPEGs."""

    VERSION = tfds.core.Version("1.0.0")
    BUILDER_CONFIGS = [
      MyDIV2KConfigJPEG(
          name="default",
          description="My custom DIV2K-like dataset using JPEGs",
      ),
    ]

    def _info(self):
      return tfds.core.DatasetInfo(
        builder=self,
        description="My custom DIV2K-like dataset using JPEGs.",
        features=tfds.features.FeaturesDict({
          "hr": tfds.features.Image(),
          "lr": tfds.features.Image(),
          "lr_x2": tfds.features.Image(),
          "lr_x3": tfds.features.Image(),
          "lr_x4": tfds.features.Image(),
        }),
      )

    def _split_generators(self, dl_manager):
      return {
          "train": self._generate_examples(dl_manager),
      }

    def _generate_examples(self, dl_manager):
      hr_dir = os.path.join(dl_manager.manual_dir, 'HR')
      lr_dir = os.path.join(dl_manager.manual_dir, 'LR')

      for filename in os.listdir(hr_dir):
          if filename.endswith('.jpg'):
              base_name = os.path.splitext(filename)[0]
              hr_path = os.path.join(hr_dir, filename)
              lr_path_x2 = os.path.join(lr_dir, f'{base_name}x2.jpg')
              lr_path_x3 = os.path.join(lr_dir, f'{base_name}x3.jpg')
              lr_path_x4 = os.path.join(lr_dir, f'{base_name}x4.jpg')

              if os.path.exists(lr_path_x2) and os.path.exists(lr_path_x3) and os.path.exists(lr_path_x4):
                  yield base_name, {
                      "hr": hr_path,
                      "lr": lr_path_x2,
                      "lr_x2":lr_path_x2,
                      "lr_x3":lr_path_x3,
                      "lr_x4":lr_path_x4,
                  }

if __name__ == '__main__':
  manual_dir = 'my_div2k'
  builder = MyDIV2KJPEG(data_dir='tmp')
  dataset = builder.as_dataset(split='train', manual_dir=manual_dir)
  for example in dataset.take(1):
        print(example['hr'].shape)
        print(example['lr'].shape)
```

Here, `_generate_examples` now explicitly looks for `.jpg` files and constructs the paths using the correct extension. The core approach is to define a custom dataset class using `tfds` which handles the specific needs of the given data format, instead of trying to force fit the data into the default `DIV2K` expectations.

**Example 3:  Missing scaling factors**

In a more sparse scenario, you might only have x2 versions of the LR image. The default script would be expecting all scaling factors.

```
my_div2k/
    HR/
        0001.png
        0002.png
        ...
    LR/
        0001x2.png
        0002x2.png
        ...
```

The modified `_generate_examples` method would need to account for potentially missing entries. Here's a modified version of the prior `MyDIV2KJPEG` generator to address this situation:

```python
import tensorflow_datasets as tfds
import tensorflow as tf
import os

class MyDIV2KConfigMissingFactors(tfds.core.BuilderConfig):
    """Configuration for local DIV2K-like dataset with missing LR factors."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class MyDIV2KMissingFactors(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for my custom DIV2K style data with missing scale factors."""

    VERSION = tfds.core.Version("1.0.0")
    BUILDER_CONFIGS = [
      MyDIV2KConfigMissingFactors(
          name="default",
          description="My custom DIV2K-like dataset with missing scale factors",
      ),
    ]

    def _info(self):
      return tfds.core.DatasetInfo(
        builder=self,
        description="My custom DIV2K-like dataset with missing scale factors.",
        features=tfds.features.FeaturesDict({
          "hr": tfds.features.Image(),
          "lr_x2": tfds.features.Image(),
          "lr_x3": tfds.features.Image(),
          "lr_x4": tfds.features.Image(),
        }),
      )

    def _split_generators(self, dl_manager):
      return {
          "train": self._generate_examples(dl_manager),
      }

    def _generate_examples(self, dl_manager):
      hr_dir = os.path.join(dl_manager.manual_dir, 'HR')
      lr_dir = os.path.join(dl_manager.manual_dir, 'LR')
      for filename in os.listdir(hr_dir):
          if filename.endswith('.png'):
              base_name = os.path.splitext(filename)[0]
              hr_path = os.path.join(hr_dir, filename)
              lr_path_x2 = os.path.join(lr_dir, f'{base_name}x2.png')
              lr_path_x3 = os.path.join(lr_dir, f'{base_name}x3.png') #may not exist
              lr_path_x4 = os.path.join(lr_dir, f'{base_name}x4.png') #may not exist

              example = {"hr": hr_path, "lr_x2": lr_path_x2, "lr_x3":None, "lr_x4":None}
              if os.path.exists(lr_path_x2):
                  example["lr_x2"] = lr_path_x2
              if os.path.exists(lr_path_x3):
                  example["lr_x3"]= lr_path_x3
              if os.path.exists(lr_path_x4):
                   example["lr_x4"]=lr_path_x4
              yield base_name, example


if __name__ == '__main__':
  manual_dir = 'my_div2k'
  builder = MyDIV2KMissingFactors(data_dir='tmp')
  dataset = builder.as_dataset(split='train', manual_dir=manual_dir)
  for example in dataset.take(1):
        print(example)
        print(example['hr'].shape)
        if example["lr_x3"]!=None:
           print(example["lr_x3"].shape)

```
This version handles the missing files by storing none in the dictionary. Importantly, the feature spec in the _info method would need to reflect the data, namely having `None` be a valid element. If all scaling factors are necessary for the work you intend to complete, you would need to ensure that your data reflects those scaling factors.

To summarize, the primary issue causing the DIV2K script to fail with local datasets arises from discrepancies between the script's expected structure and the actual data arrangement. The solution involves defining custom `tfds.core.GeneratorBasedBuilder` classes, which allow for flexible handling of your specific directory structures and naming conventions. These classes require the proper implementation of `_info` and `_generate_examples` to define the dataset's features and to read data accordingly. The examples provided above present different ways of resolving data mismatches through modifications in the `_generate_examples` function as needed. In most cases, the default DIV2K script is not suitable for custom datasets.

For further study, I'd recommend closely reviewing the TensorFlow Datasets documentation for the `tfds.core.GeneratorBasedBuilder` class. Understanding how to leverage the `dl_manager` argument within `_generate_examples` is critical. Additionally, exploring the source code of the default DIV2K dataset builder within TensorFlow Datasets (search for `div2k.py` in the tfds repository) is valuable. Finally, carefully reviewing your own data directory structure and meticulously tracking what is expected of the data by the model will often highlight the root issue causing failures during data loading.
