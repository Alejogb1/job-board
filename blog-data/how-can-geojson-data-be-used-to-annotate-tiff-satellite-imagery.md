---
title: "How can geoJSON data be used to annotate TIFF satellite imagery?"
date: "2024-12-23"
id: "how-can-geojson-data-be-used-to-annotate-tiff-satellite-imagery"
---

Alright, let's tackle this. Geojson and tiff imagery, a classic pairing for spatial analysis, and one I've spent quite a bit of time on, especially back in my days working on agricultural yield mapping projects. It's not always straightforward, but understanding the core concepts makes the process manageable.

The crux of the issue lies in bridging two very different worlds: vector data (geojson, representing discrete features like polygons) and raster data (tiff images, representing continuous fields like satellite imagery). To annotate, we're essentially using the geojson to define areas of interest on the tiff. We’re not directly painting the geojson on the tiff; instead, we leverage the georeferencing metadata within both files to align and interpret the geojson data spatially in relation to the image pixels.

Before diving into specifics, let's outline the necessary components. Both geojson and tiff files, crucially, must share a common coordinate reference system (crs). This is absolutely fundamental; without it, your annotation will be misaligned, making the entire endeavor useless. Usually, this involves ensuring both are in something like epsg:4326 (latitude/longitude) or some specific projected coordinate system. You’ll often find that your satellite imagery, especially when sourced from commercial providers, comes pre-georeferenced, but confirming that is always step one. Geojson files might require explicit definition of crs. If these don't match, it's a preprocessing step needed to reproject one or both. We typically use libraries that handle reprojections and translations between different coordinate systems.

Now, let's discuss the practical application, focusing on how to use code to achieve the annotation. In my experience, I've mainly utilized three main strategies: using rasterio and geopandas, rasterizing geojson features, and generating masks.

**Strategy 1: Rasterio and Geopandas Integration**

This is a common approach when you need to work with both vector and raster data. The rasterio library allows you to read and work with tiff files, while geopandas provides the ability to read, manipulate, and spatially analyze geojson files.

Here’s a code snippet to demonstrate how to use these libraries for annotation.

```python
import rasterio
import geopandas
import numpy as np
from shapely.geometry import mapping
from rasterio.mask import mask

def annotate_tiff_with_geojson(tiff_path, geojson_path):
    try:
        with rasterio.open(tiff_path) as src:
            gdf = geopandas.read_file(geojson_path)

            if gdf.crs != src.crs:
                gdf = gdf.to_crs(src.crs)

            geometries = [mapping(shape) for shape in gdf.geometry]
            out_image, out_transform = mask(src, geometries, crop=True)
            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform
            })
            out_image = np.moveaxis(out_image,0,-1)

            return out_image, out_meta

    except Exception as e:
        print(f"error processing files: {e}")
        return None, None


if __name__ == '__main__':
  tiff_file = "path/to/your/image.tiff" # Replace with actual path
  geojson_file = "path/to/your/annotations.geojson"  # Replace with actual path
  annotated_image, meta = annotate_tiff_with_geojson(tiff_file, geojson_file)
  if annotated_image is not None:
     with rasterio.open("annotated_image.tiff", "w", **meta) as dest:
         dest.write(np.moveaxis(annotated_image,-1,0))
     print("annotation complete.")
  else:
    print("annotation failed.")


```
In this snippet, we open both the tiff and geojson files. The code checks for matching crs systems and reprojects if needed. We then extract the geometries from the geojson file, and `rasterio.mask.mask` function does the heavy lifting by masking the tiff based on the geojson geometries. We save the output as a tiff file containing the masked area.

**Strategy 2: Rasterizing GeoJSON Features**

Sometimes, you don't need to crop an entire image but instead need a rasterized representation of the geojson itself. Imagine you want to create a mask of the areas from geojson onto a blank canvas or even the same resolution/extent as the tiff image. This is where rasterization comes into play.

Here's some code to do this:

```python
import rasterio
import geopandas
import numpy as np
from rasterio import features

def rasterize_geojson(geojson_path, tiff_path, output_path):
    try:
        with rasterio.open(tiff_path) as src:
            gdf = geopandas.read_file(geojson_path)

            if gdf.crs != src.crs:
                gdf = gdf.to_crs(src.crs)

            shapes = [(geom, 1) for geom in gdf.geometry]
            rasterized_mask = features.rasterize(shapes,
                                                 out_shape=src.shape,
                                                 transform=src.transform,
                                                 fill=0,
                                                )
            with rasterio.open(output_path, 'w', driver='GTiff',
                            height=src.height, width=src.width,
                            count=1, dtype=rasterized_mask.dtype,
                            crs=src.crs, transform=src.transform) as dest:
                dest.write(rasterized_mask, indexes=1)
            print("rasterization complete.")

    except Exception as e:
        print(f"error during rasterization: {e}")


if __name__ == '__main__':
   tiff_file = "path/to/your/image.tiff"
   geojson_file = "path/to/your/annotations.geojson"
   output_file = "rasterized_mask.tiff"
   rasterize_geojson(geojson_file, tiff_file,output_file)
```

This approach rasterizes the geometries into a pixel-based mask based on the input tiff's extent and resolution. This resulting tiff image can then act as a binary mask or as a basis for further operations. Each pixel inside the geojson features has value '1' and outside has value '0'. This approach is especially useful when doing machine learning training data.

**Strategy 3: Generating Masks as Numpy Arrays**

Sometimes, your end goal is just a numpy array representation of a mask that you can use later directly without reading tiff images. This can be useful in many different situations, such as image processing workflows. Here's how you would do that:

```python
import rasterio
import geopandas
import numpy as np
from rasterio import features
from shapely.geometry import mapping

def create_geojson_mask(geojson_path, tiff_path):
    try:
        with rasterio.open(tiff_path) as src:
            gdf = geopandas.read_file(geojson_path)

            if gdf.crs != src.crs:
                gdf = gdf.to_crs(src.crs)

            shapes = [mapping(geom) for geom in gdf.geometry]
            mask = features.geometry_mask(shapes,
                                            out_shape=src.shape,
                                            transform=src.transform,
                                            invert=False,
                                            all_touched=True)

            return mask
    except Exception as e:
        print(f"error during mask generation: {e}")
        return None

if __name__ == '__main__':
  tiff_file = "path/to/your/image.tiff"
  geojson_file = "path/to/your/annotations.geojson"
  mask = create_geojson_mask(geojson_file, tiff_file)
  if mask is not None:
    print("mask generated.")
    # Do something with the mask.
  else:
    print("mask generation failed")
```

This method is similar to rasterizing, but instead of returning a tiff image, the output is a numpy array representing the mask. This can be useful if you just need the mask in memory.

**Further Reading:**

For a deeper understanding of the underlying concepts, I'd strongly recommend a couple of authoritative texts. For a comprehensive look at Geographic Information Systems, "Geographic Information Systems and Science" by Paul A. Longley, Michael F. Goodchild, David J. Maguire, and David W. Rhind is invaluable. For a detailed dive into spatial analysis, "Spatial Analysis: With R" by Bailey and Gatrell, while focusing on R, has many concepts applicable to other ecosystems. Also, the documentation for `rasterio` and `geopandas` is excellent, providing both theoretical background and practical examples. These are resources I've found invaluable over the years.

In summary, the successful annotation of tiff satellite imagery using geojson data hinges upon a correct coordinate reference system, proper handling of data types, and the employment of spatial analysis libraries such as `rasterio` and `geopandas`. These strategies, combined with solid foundational knowledge, will enable you to build sophisticated geospatial applications.
