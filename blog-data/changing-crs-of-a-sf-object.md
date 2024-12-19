---
title: "changing crs of a sf object?"
date: "2024-12-13"
id: "changing-crs-of-a-sf-object"
---

Alright listen up I've seen this rodeo a few times you want to change the coordinate reference system CRS of a simple features sf object in R right That's like trying to change the language of a document after its written tricky but doable Been there done that got the t-shirt And by t-shirt I mean several late nights wrestling with geospatial data

So the question is "changing CRS of a sf object" yeah that's pretty vague Lets dive into it its about spatial data and how its referenced on earth That's the core stuff of geospatial analysis I've personally spent probably months of my life on this you know that feeling of getting something to align properly finally after a day of errors that's the vibe I get when someone asks this

Let's start with the basics An sf object it’s basically a fancy table but with geometry it has columns and it also has a special column that holds the coordinates representing points lines or polygons This special column is tied to a CRS that CRS is how the coordinates are interpreted on our planet like a map grid reference you know

Now you want to change it why? Probably you have data from different sources with different coordinate systems And if you try to plot them together they'll be all over the place like they're having a party in different dimensions That's not a good party man I mean like at all

Changing a CRS is not simply modifying the numbers in your data what it means is reprojecting the geometry from one reference system to another It’s like translating from English to Spanish the numbers change but the spatial relationships remain the same if you do it right that is

Here are the usual ways to do that in R with the sf package I'll break it down with code snippets because who needs more theory here

**Snippet 1: The Usual `st_transform` Method**

This is your bread and butter way the go-to function and I really mean it its the first stop when you think about changing a projection it is the primary function for reprojecting the geometries of an sf object to a new coordinate system lets see some code:

```r
library(sf)

# Let's say you have an sf object called my_sf_data
# Lets assume it's in WGS 84 which is EPSG code 4326
# And you want it in UTM zone 12N which is EPSG code 32612
# Its dummy data and it does not matter what it is

my_sf_data <- st_sf(
  data.frame(id = 1:3),
  geometry = st_sfc(
    st_point(c(-105, 40)),
    st_point(c(-104, 41)),
    st_point(c(-103, 39)),
    crs = 4326
  )
)


# Re-project to UTM zone 12N
my_sf_data_transformed <- st_transform(my_sf_data, crs = 32612)

# Just check if the crs changed
st_crs(my_sf_data_transformed)

# It's done!
```

What’s happening here? We load `sf` obviously. We create a simple sf object with dummy point data using  `st_sfc` it is just a small example it could be a polygon a line just does not matter We specify the original CRS as 4326 (WGS 84 which is the geographic coordinate system that everyone uses at least most of people if you are not living in mars or something) Then the magic happens the `st_transform` function takes the sf object and the new CRS specified as an EPSG code this way we are using numbers not names to specify what CRS is and we want which makes it easier to handle and faster in most cases and we obtain a new object with the transformed coordinates and the new CRS assigned to it If you do `st_crs` of the transformed data it shows you the new projection.

**Snippet 2: Handling Custom PROJ Strings**

Sometimes you won't find your target CRS as a ready-made EPSG code then what? Then you have to dive into PROJ strings that is a text format that describes any CRS on earth well that is the idea at least I remember when I started to see PROJ strings I thought it was greek but it is not I mean at the end of the day it is a string of text and if you know what it means you can understand it It is like understanding a config file a yaml or json its just more specialized
PROJ strings gives you extreme customization for CRS definitions.

```r
library(sf)

# Create a sample sf object with a predefined CRS
my_sf_data <- st_sf(
  data.frame(id = 1:3),
  geometry = st_sfc(
    st_point(c(-105, 40)),
    st_point(c(-104, 41)),
    st_point(c(-103, 39)),
    crs = 4326
  )
)


# Custom PROJ string example
my_custom_proj <- "+proj=lcc +lat_1=33 +lat_2=45 +lat_0=39 +lon_0=-96 +x_0=0 +y_0=0 +ellps=GRS80 +datum=NAD83 +units=m +no_defs"

# Transform with the custom proj string
my_sf_data_custom_proj <- st_transform(my_sf_data, crs = my_custom_proj)

# Check the new CRS which is now a PROJ string
st_crs(my_sf_data_custom_proj)

```

See? Here we created a custom PROJ string that defines a Lambert Conformal Conic projection which is another projection type and it's something we define from scratch I mean that means that this is not a standard one It is not in the epsg database we feed it to `st_transform` and voila your sf object now in the new coordinate system with that PROJ string
Now why would you do this Well sometimes if your data is not a usual thing like not something used in the usual mapping stuff it is something special like a study a research you have to build your own from the ground up this is where those things shine.

**Snippet 3: Dealing with Undefined CRSs**

And now the fun part it’s when you have data that is missing its CRS information or it is defined incorrectly This happens way too often in my experience and it's like having a treasure map without the key not fun not at all you think you know where you are but you are completely wrong
First you need to figure out what the correct CRS should be and then assign it manually then you can reproject it

```r
library(sf)

# Say you have an sf object with undefined CRS
my_sf_data_no_crs <- st_sf(
  data.frame(id = 1:3),
  geometry = st_sfc(
    st_point(c(-105, 40)),
    st_point(c(-104, 41)),
    st_point(c(-103, 39))
  )
)

# Check CRS it will return NA
st_crs(my_sf_data_no_crs)

# Lets say you know the correct CRS is WGS84
# lets assume it is 4326
my_sf_data_with_crs <- st_set_crs(my_sf_data_no_crs, 4326)

# Now check CRS, It is 4326 now
st_crs(my_sf_data_with_crs)

# Now you can transform
my_sf_data_transformed <- st_transform(my_sf_data_with_crs, 32612)

# All done and working with the correct projection
st_crs(my_sf_data_transformed)

```

Here’s what we do: `st_crs` confirms there is no CRS this means that the geometry data is there but it is not related to earth correctly Then we use `st_set_crs` to tell R what the CRS actually is it does not actually change coordinates only it tells R what CRS it has to deal with then now you can do the transformation to 32612 with `st_transform` it is a two step process that will save you a lot of headaches if you work with legacy or old data

**Important Considerations**

*   **Accuracy:** Reprojection involves mathematical transformations the original accuracy of the coordinates is preserved or sometimes can be lost if you do not know what you are doing but in general the accuracy is the same.

*   **Units:** Pay attention to units Different CRSs have different units for distances like meters degrees or feet and you need to know what is going on with those otherwise you might end up with a small object that you thought was huge or vice versa.

*   **EPSG Codes:** These are unique identifiers for coordinate systems a great resource is the `epsg.io` website which is a pretty standard thing in the geospatial world.

*  **Datum transformations:** sometimes, you might need to specify the exact datum transformation if your data is really old. That’s a topic for another day but know it exists. I once spent 3 days on that trying to figure out why a dataset looked like it had gone through a blender.

* **PROJ String Documentation:**  The definitive resource for understanding PROJ strings is the official PROJ documentation they are pretty deep and are not simple to understand but they are like the bible for PROJ strings and also you can find some introductory stuff in papers related to geodesy not the usual mapping stuff but it is related to the same.

And finally a joke because you asked for one I spent so much time learning about projections I think I can now project my life in multiple parallel realities just to compare them. Hah ha.

So there you have it Changing the CRS of an sf object in R is fundamental for working with geospatial data it is a required skill if you want to handle these kinds of things Be careful use the right tools for the job and always double check that you are using the correct CRS this process is a cornerstone of geospatial analysis and doing it correctly the first time will avoid issues later on.
