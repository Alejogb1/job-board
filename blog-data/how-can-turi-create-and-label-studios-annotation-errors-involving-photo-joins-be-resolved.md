---
title: "How can Turi Create and Label Studio's annotation errors involving photo joins be resolved?"
date: "2024-12-23"
id: "how-can-turi-create-and-label-studios-annotation-errors-involving-photo-joins-be-resolved"
---

,  I remember a particularly thorny project a few years back, involving satellite imagery analysis for a conservation effort. We used Turi Create for initial model training and Label Studio for annotating various features – forest boundaries, deforestation patches, that sort of thing. The project quickly ran into problems when we started dealing with adjacent image tiles; the edges became a nightmare. It turned out we were suffering from precisely the issue you’ve described – annotation inconsistencies along photo joins, especially when different annotators were involved. These errors stemmed from the way Turi Create and Label Studio handle geospatial data and the human element involved in the annotation process.

Here's the gist: Turi Create, when ingesting image data, doesn't intrinsically understand the geospatial relationship between adjacent images unless explicitly told to. It treats each image as a discrete entity. Consequently, when Label Studio presents those images for annotation, it often leads to inconsistencies where, for example, the same forest boundary is drawn slightly differently along the seam of two adjacent photos. This difference, though seemingly minor visually, can throw off model training significantly, leading to poor performance at those critical boundary regions.

The core issue boils down to the fact that we are essentially trying to reconstruct a continuous surface (the earth) from discrete, overlapping pieces and attempting to annotate that continuous surface when it's viewed in this fragmented way. The problem isn't necessarily inherent in Turi Create or Label Studio per se, but it arises from the interaction between their typical workflows and the realities of handling tiled, geospatial imagery. We can address this with careful preprocessing, consistent annotation guidelines, and, importantly, by incorporating overlap into the annotation workflow where possible.

First, consistent annotation practices and clear guidelines are paramount. The most common mistake I've seen is annotators not being fully aware of adjacent images and simply annotating in isolation. To fix this, we need to communicate the importance of seam consistency to the team. This can take the form of detailed written instructions as well as in-person training sessions. We also implemented what we called "overlap verification" where pairs of annotators reviewed boundary annotations created by their colleagues at tile edges. While this added a bit to our overall annotation time, it dramatically reduced inconsistencies.

Second, the preprocessing stage needs specific attention. This is the technical solution that will likely have the biggest impact. We need to explicitly create a system where Label Studio understands how the image tiles connect. While not natively supported, we can do this using a custom pre-processing script that generates a metadata file for each image. This metadata would be used to render adjacent images together or to highlight potential boundary inconsistencies for the annotator. We effectively create a visual aid within Label Studio, without altering its core functionality. This often involves using libraries like GDAL to extract spatial information from each image.

Here’s an example, using python for the metadata creation:

```python
import os
import json
from osgeo import gdal

def create_metadata(image_dir, output_file):
    metadata = {}
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.tif', '.tiff', '.jpg', '.jpeg', '.png')):
            filepath = os.path.join(image_dir, filename)
            ds = gdal.Open(filepath)
            if ds is None:
                print(f"Could not open {filename}")
                continue

            gt = ds.GetGeoTransform()
            width = ds.RasterXSize
            height = ds.RasterYSize

            #extract the bounding box using the spatial information
            top_left_x = gt[0]
            top_left_y = gt[3]
            bottom_right_x = top_left_x + width * gt[1]
            bottom_right_y = top_left_y + height * gt[5]

            metadata[filename] = {
                "bounding_box": [top_left_x,top_left_y,bottom_right_x,bottom_right_y],
                "width": width,
                "height": height
            }
    with open(output_file, "w") as f:
      json.dump(metadata, f, indent=4)

if __name__ == "__main__":
    image_directory = "path/to/your/image/tiles" # Replace with your image tile directory
    output_meta_file = "image_metadata.json"  # Name of the json file created
    create_metadata(image_directory, output_meta_file)
    print(f"Metadata file created: {output_meta_file}")
```

This script generates a `json` file containing spatial metadata for each image in your directory, which you can subsequently utilize within a custom Label Studio environment.

Now, when we integrate the generated metadata with Label Studio, we can create a custom display to highlight the image edges and improve annotation consistency. Label Studio is pretty flexible in how it presents images and this is where a small amount of custom Javascript might be required. The following simple example provides an overview of how to use the metadata to highlight the edges of images during the annotation process:

```javascript
// This JavaScript snippet would be part of a custom HTML template in Label Studio
function setupAdjacentImageHighlight(metadata, imageElement){
  const imageName = imageElement.src.split('/').pop();
  const imageMeta = metadata[imageName];
    if (!imageMeta){
        return; //If the image is not found in the metadata, return
    }

    //example only, needs specific styling to look like a good border
  imageElement.style.border = "2px solid red";

  //example of how to access the coordinates for debugging
  console.log("Image Name:", imageName, "Bounding Box:", imageMeta.bounding_box);
}

document.addEventListener('DOMContentLoaded', function() {
    const metadata_url = "image_metadata.json"; // Replace this with where the json file is available

    fetch(metadata_url)
        .then(response => response.json())
        .then(data => {

            // Select all the image elements currently in Label studio
            document.querySelectorAll('img').forEach(function(image) {
                setupAdjacentImageHighlight(data, image);
            });

            // Add a mutation observer, so if images are added dynamically to the page they get the highlighting
            const observer = new MutationObserver(mutations => {
               mutations.forEach(mutation => {
                mutation.addedNodes.forEach(node => {
                    if (node.tagName === 'IMG') {
                       setupAdjacentImageHighlight(data, node)
                     }
                    });
                });
            });
          observer.observe(document.body, { childList: true, subtree: true });

        });
});
```

This Javascript snippet fetches the previously created `json` metadata file and styles the image edge using the meta information. It utilizes a mutation observer to ensure that even dynamically added images to Label Studio get the styling. It's worth noting that this is a simple example to provide a quick visualization of the issue. In a complete solution, you might choose to draw the outline using SVG or a canvas object.

Finally, for more complex cases involving significant overlap, consider using the spatial data within the metadata to create a combined view of adjacent images within Label Studio. This involves stitching them together into a single canvas and using the metadata to ensure the correct alignment. I won’t provide the specific code here as it’s more complex, but the basic principle would be that you load the multiple images into a javascript canvas using information from the metadata file created previously.

While these Javascript examples aren’t ready for immediate plug-and-play, they should illustrate how you can create the visual information needed to address the issue. For a deeper understanding of these methods, I recommend consulting geospatial processing literature. Specifically, the *Handbook of Geographic Information Science* edited by Wilson & Fotheringham is a valuable resource for understanding the underlying spatial concepts and techniques. For details on raster manipulation within GDAL, the GDAL Cookbook is indispensable, though sometimes a bit technical, it offers a solid foundation. Additionally, the Label Studio documentation itself is excellent and explains how to customize its display.

These methods are what we used to resolve our boundary annotation issues. The key takeaway here is that addressing annotation errors related to photo joins involves a combination of technical preprocessing and clear, consistent human practices. It’s a multidisciplinary effort that requires careful planning but will ultimately lead to improved model accuracy and better overall results.
