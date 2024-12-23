---
title: "extracting metadata from large set of images?"
date: "2024-12-13"
id: "extracting-metadata-from-large-set-of-images"
---

 so you’re wrestling with extracting metadata from a ton of images right I get it Been there done that more times than I care to remember It's like the universe decided to dump all its picture data on your doorstep and said “have fun”

First off we are talking serious scale here right It's not like 20 pictures from your phone we are dealing with potentially thousands or hundreds of thousands or even more So forget about manually going through each file and pulling out Exif data by hand we need a proper automated pipeline

I recall when I first encountered this issue way back when I was working on a research project dealing with remote sensing data I had this massive archive of satellite images and I needed to get at information like acquisition date coordinates the sensor used and everything else to properly do analysis We initially tried some really janky batch scripts and it was pure pain They would hang crash randomly miss data and generally cause headaches It was a complete mess

The thing is the more images you have the slower that naive approach gets That is when i started to think we needed something more robust and scalable The key is to use a dedicated library for image processing that can handle that heavy lifting efficiently And the best thing is that a lot of them are out there

So what are your options specifically for python because thats my tool of choice? The `Pillow` library is a good starting point It handles basic image manipulation but it also lets you access metadata pretty easily. But as usual Pillow's support is pretty limited to the most common cases So for more advanced metadata extraction or formats you might need something more specialized. There are more specialized solutions for each file format

Here is a super basic example using Pillow:

```python
from PIL import Image
from PIL.ExifTags import TAGS

def extract_basic_metadata(image_path):
  try:
      image = Image.open(image_path)
      exifdata = image.getexif()

      metadata = {}
      for tag_id in exifdata:
        tag = TAGS.get(tag_id, tag_id)
        data = exifdata.get(tag_id)

        if isinstance(data, bytes):
            try:
              data = data.decode('utf-8')
            except UnicodeDecodeError:
              data = str(data)

        metadata[tag] = data
      return metadata
  except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


if __name__ == '__main__':
    image_file = "your_image.jpg" #replace this with the image path you have
    metadata = extract_basic_metadata(image_file)

    if metadata:
        for key, value in metadata.items():
            print(f"{key}: {value}")
    else:
        print("Could not extract metadata")
```

This is good if you're dealing with standard JPEG images but as you’ll see it doesn’t handle all image formats or all the edge cases so we will need more advanced tools. Notice that I have tried to handle the error cases too. This is because you will have files of all sorts and not all of them will behave the same.

If you find that your images include very specific metadata formats you may need to look at other libraries and techniques for example for medical images you may need to use libraries that understand the DICOM standard which is really different from a generic jpeg picture.

For example, if you have many TIFF files or GeoTIFF images the `rasterio` library is invaluable it’s specifically designed for geospatial raster data and handles all that geo-referencing stuff along with metadata.

Let's see an example of this with `rasterio`:

```python
import rasterio

def extract_geotiff_metadata(geotiff_path):
    try:
        with rasterio.open(geotiff_path) as src:
            metadata = {
                "crs": str(src.crs),
                "bounds": src.bounds,
                "transform": src.transform,
                "width": src.width,
                "height": src.height,
                "count": src.count,
                "driver": src.driver,
                "dtype": str(src.dtypes[0]),
                "nodata": src.nodata,
                "tags": src.tags(),
            }
            return metadata
    except Exception as e:
        print(f"Error processing {geotiff_path}: {e}")
        return None

if __name__ == '__main__':
    geotiff_file = "your_geotiff.tif" #replace this with your file path
    metadata = extract_geotiff_metadata(geotiff_file)

    if metadata:
        for key, value in metadata.items():
            print(f"{key}: {value}")
    else:
        print("Could not extract metadata")

```
See how this gives you coordinate reference systems bounds and much more all useful for geospatial data analysis. If you’re not dealing with this type of data you will probably ignore this example

Now another big thing to consider is the sheer volume of data you’re processing doing it all serially is probably not going to cut it. For example if you have 100000 images and processing one image takes let's say half a second you are talking about processing images for half a day without interruption. We should try to parallelize our processing and not do this sequentially

This brings us to the need for multiprocessing in python. This is where you can use your computer's CPU to do parallel operations. Here’s a snippet showing how we could parallelize metadata extraction with the multiprocessing module It is not the fastest but it works ok

```python
import os
import multiprocessing
from PIL import Image
from PIL.ExifTags import TAGS

def extract_metadata_parallel(image_path):
    try:
        image = Image.open(image_path)
        exifdata = image.getexif()

        metadata = {}
        for tag_id in exifdata:
            tag = TAGS.get(tag_id, tag_id)
            data = exifdata.get(tag_id)

            if isinstance(data, bytes):
              try:
                data = data.decode('utf-8')
              except UnicodeDecodeError:
                data = str(data)

            metadata[tag] = data
        return image_path , metadata
    except Exception as e:
        return image_path, None

def process_images(image_paths, num_processes):
    pool = multiprocessing.Pool(processes=num_processes)
    results = pool.map(extract_metadata_parallel, image_paths)
    pool.close()
    pool.join()
    return results

if __name__ == '__main__':
    image_dir = 'your_image_directory' #place your directory here
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir,f)) and f.lower().endswith(('.jpg','.jpeg','.png'))] #we are filtering the file list for images
    num_processes = multiprocessing.cpu_count()
    metadata_results = process_images(image_files, num_processes)

    for file_path, metadata in metadata_results:
      if metadata:
        print(f"Metadata for {file_path}:")
        for key , value in metadata.items():
          print(f"   {key}: {value}")
      else:
        print(f"Metadata not found for {file_path}")
```
Notice that this is a basic example for parallelism there are better faster ways with other tools but it serves to illustrate the approach

Also remember that I am only printing stuff here you will probably want to save this extracted metadata to a file something like a csv file for each image or to a database that you can later analyze. You probably need to create an appropriate database to store this metadata. Also consider that you may need a schema of the kind of metadata you are looking for.

Now a very important consideration is what kind of images are we talking about For example are these jpg files png files or are they raw camera files which may have very particular metadata formats Also there could be medical imaging data like DICOMs that are different altogether or satellite images

Finally keep in mind that not all images have all metadata some may have nothing others will have a lot and then some are just simply corrupted which means we have to handle all those edge cases. One time I spent a week debugging an issue and it was just because some corrupt image was causing the process to crash.

So my best advice for you is to always handle errors be ready to deal with different image formats or data types. Also make sure you choose a robust library with an active development community if you can and never forget to profile your code for performance and parallelization and also save your extracted metadata in an appropriate format. Don't make the same mistake that I did by losing tons of extracted metadata. Trust me it is way better to create a robust metadata pipeline from the get go than trying to fix things later because if you think about it it makes zero sense to do things wrong the first time.

And please if you can look at some of the papers from libraries you use they are good resources for understanding their implementations of metadata extraction tools for example: "Pillow: Python Imaging Library" or "Rasterio: Geographic Raster Data Processing with Python". These will give you a deeper understanding of the tools and help you debug faster in the long run. It's much more productive than spending hours copy-pasting from StackOverflow and hoping something will just magically work which is my favorite way to spend time of course.
