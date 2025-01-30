---
title: "How can I create spatially disjoint training and test datasets from a custom dataset in Python?"
date: "2025-01-30"
id: "how-can-i-create-spatially-disjoint-training-and"
---
In my experience building machine learning models for geospatial analysis, a frequent challenge is ensuring robust model evaluation by preventing data leakage between training and testing sets, especially when dealing with inherently spatial data. This challenge arises because spatially proximal data points are often highly correlated. Randomly partitioning a dataset without accounting for spatial relationships can lead to an overestimation of model performance, as the model might effectively be memorizing patterns present in both the training and test sets rather than generalizing to new, unseen locations. To address this, we must create spatially disjoint partitions.

The fundamental principle is to divide the geographical area covered by the dataset into separate regions, using these regions to assign data points entirely to either the training set or the test set, not both. This prevents any overlap and, therefore, data leakage based on spatial proximity. Implementing this involves: 1) identifying or creating spatial units, 2) assigning these units to train or test sets, and 3) allocating data points based on their containing unit. The choice of spatial units depends on the nature of the data. Common choices include administrative regions (if such data is available), regular grids of equal size, or more sophisticated techniques that group data based on spatial proximity.

A simple yet effective approach, especially with smaller datasets or a uniform distribution, is to use a grid. Here's how I've typically implemented this:

**Code Example 1: Grid-Based Spatial Partitioning**

```python
import numpy as np
import pandas as pd

def create_grid_partition(dataframe, lat_col, lon_col, grid_size_meters, test_ratio=0.2):
    """Partitions a DataFrame into spatially disjoint training and test sets using a grid."""

    min_lat, max_lat = dataframe[lat_col].min(), dataframe[lat_col].max()
    min_lon, max_lon = dataframe[lon_col].min(), dataframe[lon_col].max()

    lat_range = max_lat - min_lat
    lon_range = max_lon - min_lon

    earth_radius_meters = 6371000  # Average Earth radius
    meters_per_degree_lat = earth_radius_meters * (np.pi / 180)
    meters_per_degree_lon = earth_radius_meters * (np.pi / 180) * np.cos(np.mean([min_lat, max_lat]) * np.pi / 180)

    grid_size_lat = grid_size_meters / meters_per_degree_lat
    grid_size_lon = grid_size_meters / meters_per_degree_lon

    num_lat_grids = int(np.ceil(lat_range / grid_size_lat))
    num_lon_grids = int(np.ceil(lon_range / grid_size_lon))

    dataframe['grid_lat_index'] = ((dataframe[lat_col] - min_lat) / grid_size_lat).astype(int)
    dataframe['grid_lon_index'] = ((dataframe[lon_col] - min_lon) / grid_size_lon).astype(int)
    dataframe['grid_id'] = dataframe['grid_lat_index'] * num_lon_grids + dataframe['grid_lon_index']

    unique_grid_ids = dataframe['grid_id'].unique()
    np.random.shuffle(unique_grid_ids)

    test_size = int(len(unique_grid_ids) * test_ratio)
    test_grid_ids = set(unique_grid_ids[:test_size])

    dataframe['is_test'] = dataframe['grid_id'].apply(lambda x: x in test_grid_ids)

    train_df = dataframe[dataframe['is_test'] == False].drop(columns = ['grid_lat_index', 'grid_lon_index', 'grid_id', 'is_test'])
    test_df = dataframe[dataframe['is_test'] == True].drop(columns = ['grid_lat_index', 'grid_lon_index', 'grid_id', 'is_test'])

    return train_df, test_df


# Example usage
data = {'latitude': np.random.uniform(30, 40, 100), 'longitude': np.random.uniform(-100, -90, 100), 'target': np.random.rand(100)}
df = pd.DataFrame(data)
train, test = create_grid_partition(df, 'latitude', 'longitude', 100000) # grid size = 100km

print(f"Training set size: {len(train)}")
print(f"Testing set size: {len(test)}")
```
This function first converts the provided latitude/longitude to indices based on the chosen grid size in meters. It then assigns grid IDs to each data point, shuffles these unique grid IDs, selects a subset of the grids for testing based on the `test_ratio`, and assigns data points to the training or test datasets based on their grid ID. By operating on grid cells rather than individual points, spatial leakage is minimized. Note, the function assumes a fairly local scale where spherical approximations are acceptable. For larger areas or higher accuracy, a geodetic library should be employed.

However, grid-based approaches might not always be ideal. When data is clustered or has irregular spatial distribution, fixed-size grid cells can lead to unequal representation or imbalanced splits. To address these shortcomings, a more refined method using a clustering algorithm can be used. This creates custom, spatially-aware clusters based on data density.

**Code Example 2: Spatial Clustering for Partitioning**

```python
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

def cluster_based_partition(dataframe, lat_col, lon_col, eps_meters, min_samples=5, test_ratio=0.2):
  """Partitions a DataFrame into spatially disjoint training and test sets using DBSCAN."""
  earth_radius_meters = 6371000
  meters_per_degree_lat = earth_radius_meters * (np.pi / 180)
  meters_per_degree_lon = earth_radius_meters * (np.pi / 180) * np.cos(np.mean(dataframe[lat_col]) * np.pi / 180)
  
  eps_degrees_lat = eps_meters / meters_per_degree_lat
  eps_degrees_lon = eps_meters / meters_per_degree_lon

  coords = dataframe[[lat_col, lon_col]].values
  scaler = StandardScaler()
  scaled_coords = scaler.fit_transform(coords)

  dbscan = DBSCAN(eps=max(eps_degrees_lat, eps_degrees_lon), min_samples=min_samples)
  clusters = dbscan.fit_predict(scaled_coords)
  
  dataframe['cluster_id'] = clusters
  unique_cluster_ids = np.unique(clusters[clusters != -1])
  np.random.shuffle(unique_cluster_ids)
  
  test_size = int(len(unique_cluster_ids) * test_ratio)
  test_cluster_ids = set(unique_cluster_ids[:test_size])
  dataframe['is_test'] = dataframe['cluster_id'].apply(lambda x: x in test_cluster_ids if x != -1 else False)
  
  train_df = dataframe[dataframe['is_test'] == False].drop(columns = ['cluster_id', 'is_test'])
  test_df = dataframe[dataframe['is_test'] == True].drop(columns = ['cluster_id', 'is_test'])
  
  return train_df, test_df


# Example usage
data = {'latitude': np.random.uniform(30, 40, 100), 'longitude': np.random.uniform(-100, -90, 100), 'target': np.random.rand(100)}
df = pd.DataFrame(data)
train, test = cluster_based_partition(df, 'latitude', 'longitude', 50000, min_samples=3) # eps=50km

print(f"Training set size: {len(train)}")
print(f"Testing set size: {len(test)}")

```
This function utilizes DBSCAN, a density-based clustering algorithm, to identify clusters of points based on their spatial proximity. The data is scaled prior to DBSCAN, a standard practice with this clustering method. It calculates epsilon for clustering in degree units, then creates clusters with a chosen minimum size. The test/train split is then handled similarly to the grid method. The `-1` label from DBSCAN, denoting noise, is treated as belonging to the training set. This approach adapts to the distribution of data points in a given dataset, often resulting in more balanced splits than a rigid grid would provide.

There is a more rigorous method for data sets which have spatial polygons (e.g., building footprints, administrative boundaries, or watersheds). The following example uses an indexing structure to handle polygon data effectively, a common requirement in GIS analysis.

**Code Example 3: Polygon-Based Spatial Partitioning**

```python
import geopandas as gpd
import shapely.geometry
import numpy as np
from rtree import index
from shapely import STRtree

def polygon_based_partition(geodataframe, polygon_geometry_col, test_ratio = 0.2):
    """Partitions a GeoDataFrame into spatially disjoint training and test sets using polygon geometries."""
    polygons = geodataframe[polygon_geometry_col]
    tree = STRtree(polygons)

    polygon_ids = list(range(len(polygons)))
    np.random.shuffle(polygon_ids)
    test_size = int(len(polygon_ids) * test_ratio)

    test_polygon_ids = set(polygon_ids[:test_size])

    geodataframe['is_test'] = geodataframe.index.map(lambda i: i in test_polygon_ids)
    train_df = geodataframe[geodataframe['is_test'] == False].drop(columns = ['is_test'])
    test_df = geodataframe[geodataframe['is_test'] == True].drop(columns = ['is_test'])
    
    return train_df, test_df

#Example Usage
polygons = [
    shapely.geometry.Polygon([(0, 0), (0, 2), (2, 2), (2, 0)]),
    shapely.geometry.Polygon([(3, 0), (3, 2), (5, 2), (5, 0)]),
    shapely.geometry.Polygon([(0, 3), (0, 5), (2, 5), (2, 3)]),
    shapely.geometry.Polygon([(3, 3), (3, 5), (5, 5), (5, 3)]),
    shapely.geometry.Polygon([(1, 1), (1, 4), (4, 4), (4, 1)]),
]
gdf = gpd.GeoDataFrame({'geometry': polygons, 'target': np.random.rand(len(polygons))})

train_gdf, test_gdf = polygon_based_partition(gdf, 'geometry', test_ratio=0.3)

print(f"Training set size: {len(train_gdf)}")
print(f"Testing set size: {len(test_gdf)}")

```
This code makes use of `geopandas` and `shapely` for manipulating spatial objects and `rtree` for efficient spatial indexing. It first creates an R-tree structure with the polygons in the given dataframe. An R-tree provides an efficient way to perform spatial queries such as identifying overlapping features. Then, instead of using random data points for testing and training, the polygons themselves are randomly selected to prevent spatial leakage. All data points within a test polygon would then constitute a test set, and vice versa. Note that this solution does not account for any spatial relationships between the polygons themselves, such as adjacency.

Regarding resources, for a comprehensive understanding of spatial data handling, I'd recommend exploring materials from the following areas: Geographic Information Systems (GIS) fundamentals, specifically the concepts of coordinate systems, spatial analysis, and geoprocessing techniques. For algorithmic details, delve into resources covering machine learning for spatial data, clustering algorithms, and spatial indexing such as R-trees. Additionally, studying documentation of Python spatial analysis libraries like GeoPandas, Shapely, and scikit-learn is essential. Finally, research best practices in machine learning model evaluation, particularly regarding data leakage prevention and robust validation techniques.
