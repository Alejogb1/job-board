---
title: "How are images saved with overlaid detections using FiftyOne?"
date: "2025-01-30"
id: "how-are-images-saved-with-overlaid-detections-using"
---
The crucial aspect of saving images with overlaid detections in FiftyOne revolves around the dynamic interplay between its `Dataset` object, its flexible plotting capabilities, and the process of rendering these plots to static image files. I've routinely encountered situations where inspecting individual detections within a large image dataset became unwieldy, necessitating a method to visualize and preserve the results directly onto the images themselves.

The primary challenge lies not in the detection process itself – FiftyOne handles integrating detection models proficiently – but in persistently applying and storing those detections as visual overlays. Simply put, FiftyOne’s `Dataset` holds the underlying data and metadata, including the bounding box detections, but those detections aren’t inherently embedded in the image pixels. To achieve this, we need to leverage FiftyOne's plotting tools and subsequently save the rendered plots as image files.

The core procedure involves the following steps: First, we load or create a `Dataset` populated with image paths and detection data (which are often stored in a specific format like COCO or custom JSON). Second, we invoke the appropriate plotting function, typically using the `plot_detections()` method from a sample object within the `Dataset`. This method generates a plot of an image with overlaid detections. Third, instead of displaying this plot interactively, we instruct the plotting function to save the rendered output to a file. Finally, we iterate through the samples of the `Dataset`, applying the plotting and saving procedure to each one, resulting in a dataset of images with baked-in detections.

Crucially, `plot_detections()` doesn’t modify the original image file; instead, it creates a visual representation on top of the existing data. The saved output is a completely new image file containing the overlaid information. This is vital because it ensures the integrity of the source images within the `Dataset`.

Here are three code examples demonstrating the process, accompanied by commentaries explaining their individual purpose and scope:

**Example 1: Basic Detection Overlay and Save**

```python
import fiftyone as fo
import fiftyone.zoo as foz
import os

# Create a dummy dataset (replace with your actual dataset)
dataset = foz.load_zoo_dataset("quickstart-groups")

# Create output directory
output_dir = "images_with_detections"
os.makedirs(output_dir, exist_ok=True)

for sample in dataset:
  # Use the sample's ID as the base file name to be consistent with its sample ID
  base_name = str(sample.id)

  # Generate the plot, specifying to save instead of display
  plot = sample.plot_detections(
      labels=True, 
      figsize=(10, 10),  # Adjust size as necessary
      save_path=os.path.join(output_dir, f"{base_name}_detections.png") 
  )
```

*Commentary:* This example demonstrates the fundamental process. We first load a sample dataset. Then, we create an output directory for the resulting images. The for loop iterates through each sample within the dataset. `sample.plot_detections()` is called to overlay detections onto the image associated with that sample. We've specified `labels=True` to include class labels in the overlay and `figsize` to control the dimensions of the plotted image. The `save_path` argument ensures the rendered plot is saved to a file and not displayed in an interactive window. Crucially, `base_name` allows saving all the generated images in the `output_dir` using the dataset's `sample.id`.

**Example 2: Controlling Detection Appearance**

```python
import fiftyone as fo
import fiftyone.zoo as foz
import os

dataset = foz.load_zoo_dataset("quickstart-groups")
output_dir = "images_with_detections_custom"
os.makedirs(output_dir, exist_ok=True)

for sample in dataset:
  base_name = str(sample.id)

  plot = sample.plot_detections(
      labels=True,
      show_confidence=True, # Display detection confidence scores
      show_index=False,    # Disable bounding box index display
      bbox_linewidth=2, # Adjust the border width of bounding boxes
      bbox_color="red", # Use red color for bounding boxes
      label_color="yellow", # Use yellow for text labels
      figsize=(10, 10),
      save_path=os.path.join(output_dir, f"{base_name}_detections_custom.png")
  )

```
*Commentary:* This example expands upon the previous one by customizing the appearance of the overlaid detections. `show_confidence=True` enables the display of prediction scores along with the labels. We have disabled bounding box indices using `show_index=False`, and have customized the appearance of the bounding box outline using `bbox_linewidth=2` and `bbox_color="red"`. We have further customized the color of the text labels using `label_color="yellow"`. These parameters allow for fine-grained control over the visualization, ensuring the generated images are easy to interpret and match requirements.

**Example 3: Saving Specific Detection Fields**

```python
import fiftyone as fo
import fiftyone.zoo as foz
import os

dataset = foz.load_zoo_dataset("quickstart-groups")
output_dir = "images_with_detections_filtered"
os.makedirs(output_dir, exist_ok=True)

# Create a field only containing the label `cat` for the plot
for sample in dataset:
    cats_only = sample.detections.filter_labels(["cat"])
    sample['detections_cats_only'] = cats_only
    sample.save()


for sample in dataset:
  base_name = str(sample.id)

  plot = sample.plot_detections(
    detections_field='detections_cats_only', # plot only the filtered detections
    labels=True,
    figsize=(10, 10),
    save_path=os.path.join(output_dir, f"{base_name}_detections_filtered.png")
  )
```
*Commentary:* In this example, we demonstrate how to save images with overlays for *specific* detection fields. First, we create a new field named `detections_cats_only`, which contains detections of label `cat` only. This is done through the `filter_labels` method. Then we instruct the `plot_detections` method to use the field `detections_cats_only` through the `detections_field` parameter to plot only detections with label `cat`. This allows us to focus the visualizations on specific objects within a dataset. This approach is extremely beneficial when, for example, debugging an algorithm or looking at only a subset of your detections.

These examples illustrate a typical procedure, but the plotting capabilities in FiftyOne are extensive. The `plot_detections()` method accepts a range of parameters, as demonstrated, allowing for flexible customization of the visual output. Additional configurations include settings for bounding box styles, text colors, sizes, label formats, and confidence thresholds. Furthermore, one could explore advanced functionalities such as custom label mappings and embedding additional information (such as confidence scores) directly into the overlaid text.

For additional learning, I recommend exploring the official FiftyOne documentation on visualization and plotting. Look specifically for the resources regarding the `plot_detections()` function, the sample object's methods and the available parameters for custom visualizations. Also, researching the various fields that can be contained within a sample, like `detections` and how they are structured, is crucial to efficiently manipulate the data. In short, a solid understanding of how data is stored and visualized in FiftyOne is the cornerstone to efficiently creating images with overlaid detections. Finally, experimenting with various combinations of plotting options with your own datasets is the best way to gain confidence with this process.
