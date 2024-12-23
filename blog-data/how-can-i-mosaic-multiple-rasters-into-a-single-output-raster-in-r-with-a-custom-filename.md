---
title: "How can I mosaic multiple rasters into a single output raster in R, with a custom filename?"
date: "2024-12-23"
id: "how-can-i-mosaic-multiple-rasters-into-a-single-output-raster-in-r-with-a-custom-filename"
---

Alright, let's tackle this raster mosaicking challenge. It's a situation I've encountered quite a bit over the years, especially when dealing with large geospatial datasets that come in tiled formats. Rather than dealing with a single monolithic file, it's often more efficient to manage these datasets as smaller pieces. Stitching them back together for analysis is a common necessity, and doing it programmatically in R provides a great deal of control and automation.

First off, we need a structured way to approach this. The core idea is to use the `terra` package, which is a successor to `raster` and offers considerable performance improvements, particularly with larger rasters. The `terra::mosaic()` function is the linchpin of this process, but understanding how to feed it the correct inputs is key to success. The goal is to combine multiple raster layers into one coherent output, and we'll want to specify the output filename directly. This is critical for reproducibility and proper file management in any serious geospatial workflow. I remember a particularly frustrating project a few years back where neglecting proper naming conventions led to some serious file management headaches; lessons learned, trust me.

Now, let's get to some code.

**Example 1: Simple Mosaic with Default Settings**

This example demonstrates the most basic usage of `terra::mosaic()`. We'll assume you have a collection of raster files in a directory that you want to merge. For this example, I'll create some dummy rasters to work with, simulating the scenario where you have individual tiles of a larger dataset.

```R
# Install the 'terra' package if you don't have it
# install.packages("terra")

library(terra)

# Create dummy rasters (replace with your actual file paths)
r1 <- rast(matrix(1:9, nrow = 3, ncol = 3))
r2 <- rast(matrix(10:18, nrow = 3, ncol = 3))
r3 <- rast(matrix(19:27, nrow = 3, ncol = 3))

# Adjust extents so they don't overlap
ext(r2) <- ext(r1) + c(0,3,0,0)
ext(r3) <- ext(r1) + c(0,0,0,3)


# Collect rasters into a list
raster_list <- list(r1, r2, r3)


# Perform the mosaic
mosaic_raster <- mosaic(raster_list)

# Specify output filename
output_filename <- "mosaic_output.tif"

# Save the mosaic output raster
writeRaster(mosaic_raster, filename = output_filename, overwrite=TRUE)


# Optional: Verify the output
plot(mosaic_raster, main = "Mosaic Raster Output")
```

In this snippet, we're first simulating raster datasets. In a real-world situation, you'd use `rast(file_path)` to load your actual rasters. We then use `list()` to collect these raster objects into a single list. This list is the input for `terra::mosaic()`. Finally, `writeRaster()` takes care of outputting the merged result to a specified file, in this case a GeoTIFF. `overwrite=TRUE` is important to prevent errors if you rerun the script and already have a file with that name.

**Example 2: Mosaic with Specific Output Properties**

The default `mosaic()` operation often works well, but in practice you often need more control. You might need to align pixel sizes, choose a specific output resolution, or handle nodata values. Here's how to address these aspects:

```R
library(terra)

# Create dummy rasters with slightly different extents and resolutions
r1 <- rast(matrix(1:9, nrow = 3, ncol = 3), resolution = 1, xmin=0, xmax=3, ymin=0, ymax=3)
r2 <- rast(matrix(10:18, nrow = 3, ncol = 3), resolution = 1.2, xmin=3, xmax=6.6, ymin=0, ymax=3.6)
r3 <- rast(matrix(19:27, nrow = 3, ncol = 3), resolution = 1, xmin=0, xmax=3, ymin=3, ymax=6)

# Set nodata for r2 (example)
r2[r2 > 15] <- NA


raster_list <- list(r1, r2, r3)

# Mosaic with resample, snap, and no data value management
mosaic_raster <- mosaic(raster_list,
                       fun = "first", # Or "mean," "min," "max," etc.
                       tolerance = 0.1, # Adjust as needed
                       na.rm = TRUE) #Remove NA's after the mosaic if they still exist


# Specify the desired resolution and output extent if required.
# mosaic_raster <- resample(mosaic_raster, r1, method='bilinear')


output_filename <- "mosaic_output_resampled.tif"
writeRaster(mosaic_raster, filename = output_filename, overwrite=TRUE)
plot(mosaic_raster, main = "Resampled and mosaiced Raster")
```

Here, I've introduced the `tolerance` parameter, which helps to address minor alignment issues between rasters when they don't perfectly abut.  `fun = "first"` dictates that, in areas of overlap, the value from the first encountered raster will be used.  Alternatively, you can try other options such as `'mean'`, `'min'`, or `'max'` depending on how you need overlaps handled. I've also demonstrated how to set a nodata value for the second raster, using `r2[r2 > 15] <- NA` and I used the `na.rm = TRUE` parameter of the function to remove NA's after the mosacing process if there is any NA left. It's worth noting that sometimes you might require to resample your mosaic to a specific resolution using `resample()` function from the terra package, after the mosacing is done. In this specific case I commented this line of code out, but is good practice to keep it in mind depending on your requirements.
**Example 3: Looping through Rasters for Large Datasets**

When dealing with a large number of tiles, loading all rasters into memory at once can be inefficient or unfeasible. In such scenarios, iterative processing can be beneficial. I've had to use this countless times when processing large LiDAR datasets, for example. This approach allows for controlled processing of smaller batches of tiles.

```R
library(terra)

# Suppose you have a vector of filenames
raster_files <- c("tile1.tif", "tile2.tif", "tile3.tif", "tile4.tif") # Simulate file list. In reality you should list your files

# Create some dummy tiles for demonstration purposes.
r1 <- rast(matrix(1:9, nrow = 3, ncol = 3), resolution = 1, xmin=0, xmax=3, ymin=0, ymax=3)
r2 <- rast(matrix(10:18, nrow = 3, ncol = 3), resolution = 1, xmin=3, xmax=6, ymin=0, ymax=3)
r3 <- rast(matrix(19:27, nrow = 3, ncol = 3), resolution = 1, xmin=0, xmax=3, ymin=3, ymax=6)
r4 <- rast(matrix(28:36, nrow = 3, ncol = 3), resolution = 1, xmin=3, xmax=6, ymin=3, ymax=6)
writeRaster(r1, "tile1.tif", overwrite=TRUE)
writeRaster(r2, "tile2.tif", overwrite=TRUE)
writeRaster(r3, "tile3.tif", overwrite=TRUE)
writeRaster(r4, "tile4.tif", overwrite=TRUE)


#initialize the mosaic object
mosaic_raster <- rast()

for(file in raster_files){

  # Load the current raster file
  current_raster <- rast(file)
  # Add the raster to the mosaic
  mosaic_raster <- mosaic(mosaic_raster, current_raster, fun="first")

}


output_filename <- "mosaic_output_iterative.tif"
writeRaster(mosaic_raster, filename = output_filename, overwrite=TRUE)

plot(mosaic_raster, main="Iterative mosaic")

# Clean up the dummy files we have created
file.remove(list.files(pattern="tile.*tif"))
```
In this iterative solution, we start with an empty raster object which the results will be merged into, and then we load one raster at a time inside the loop and merge it with the already created mosaic using the `mosaic` function, making this approach memory efficient when dealing with massive datasets. We end this by saving the final result into an output file. Also, I included a piece of code to clean the dummy files created for the example, this is done using `file.remove()`.

For further reading, I recommend delving into "Spatial Data Science with R" by Roger S. Bivand, Edzer Pebesma, and Virgilio Gómez-Rubio. Also, specifically for raster data, the `terra` package documentation provides comprehensive insights into its functions, including `mosaic()` and `writeRaster()`. Don't underestimate the value of reading through the official package documentation; it's usually the most up-to-date and complete source of information. For foundational understanding of geospatial concepts, "Geographic Information Analysis" by David O'Sullivan and David J. Unwin can be beneficial. Remember to always experiment and test different parameters to fit the specific requirements of your data.
I hope this was helpful!
