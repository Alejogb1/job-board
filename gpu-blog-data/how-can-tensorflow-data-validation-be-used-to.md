---
title: "How can TensorFlow Data Validation be used to identify and extract anomalous data rows?"
date: "2025-01-30"
id: "how-can-tensorflow-data-validation-be-used-to"
---
TensorFlow Data Validation (TFDV) is not solely designed for direct extraction of anomalous rows during inference, but rather for identifying and analyzing anomalies within a dataset to improve model training and data understanding. I’ve found that its core strength lies in its ability to automatically generate a schema representing the expected data characteristics, allowing for the identification of deviations. This understanding forms the foundation for anomaly handling.

TFDV accomplishes this through a statistical analysis of a representative dataset. This analysis produces a schema that details the expected data types, value ranges, and presence of each feature. When new data is processed using this schema, TFDV identifies instances where the data violates these learned characteristics. The primary purpose of this analysis is not direct data extraction; it’s data validation, schema evolution, and ensuring consistency. While extraction isn't a primary function, TFDV provides the tools needed to identify anomalies, and these identified anomalies can then be filtered or extracted using other data processing methods.

Here’s a breakdown of how TFDV can be used to accomplish the task of anomaly identification and provide the foundation for extraction:

First, a schema is inferred from a training dataset. This schema defines the expected structure and properties of your data. Anomalies occur when incoming data violates this schema. Common anomalies include: unexpected values, data type mismatches, and missing features. The following examples clarify the practical steps for identifying these anomalies:

**Example 1: Basic Schema Generation and Anomaly Detection**

In this example, I'll demonstrate how to generate a schema from a pandas dataframe and detect anomalies. My experience has shown that starting with a pandas dataframe is often the easiest way to begin using TFDV.

```python
import pandas as pd
import tensorflow_data_validation as tfdv
from tensorflow_data_validation.statistics.stats_options import StatsOptions

# Sample training data
train_data = pd.DataFrame({
    'age': [25, 30, 40, 22, 28, 65],
    'income': [50000, 70000, 80000, 45000, 60000, 120000],
    'city': ['New York', 'London', 'Paris', 'New York', 'London', 'Tokyo']
})

# Sample data to validate against training data
validate_data = pd.DataFrame({
    'age': [23, 32, 45, 21, 27, 70, 'string'],
    'income': [52000, 72000, 90000, 46000, 62000, 130000, 'invalid'],
    'city': ['New York', 'London', 'Paris', 'New York', 'London', 'Seoul', '']
})

#Generate statistics from training data, setting num_top_values and num_histogram_buckets for better schema generation.
stats_options = StatsOptions(num_top_values=20, num_histogram_buckets=50)
train_stats = tfdv.generate_statistics_from_dataframe(train_data, stats_options=stats_options)

# Infer the schema from the training statistics.
schema = tfdv.infer_schema(statistics=train_stats)

# Validate the data against the schema
val_stats = tfdv.generate_statistics_from_dataframe(validate_data, stats_options=stats_options)
anomalies = tfdv.validate_statistics(statistics=val_stats, schema=schema)

# Print the anomalies
tfdv.display_anomalies(anomalies)
```
The code begins by importing necessary libraries. It then constructs two sample pandas dataframes: `train_data` to establish the expected data characteristics, and `validate_data` which contains deliberate anomalies. Using `tfdv.generate_statistics_from_dataframe` the statistics for training are created. Setting `num_top_values` and `num_histogram_buckets` is often helpful to achieve a schema that is not overly sensitive. The schema is then inferred from these statistics. Finally, statistics for `validate_data` are created, and `tfdv.validate_statistics` is used to compare the `validate_data` against the inferred schema. The function `tfdv.display_anomalies` prints the anomalies detected which can be inspected. This first example demonstrates how TFDV can be used to establish a baseline expectation and detect simple deviations such as type mismatches.

**Example 2: Refining Schema with String Domains and Custom Validations**

Often data includes categorical variables that require specific string values. In this example I'll demonstrate refining the schema to handle this case. I’ve found this particularly useful for validating inputs when dealing with structured data from different sources.

```python
import pandas as pd
import tensorflow_data_validation as tfdv
from tensorflow_data_validation.utils import display_util
from tensorflow_metadata.proto.v0 import schema_pb2

# Sample training data with a categorical feature
train_data = pd.DataFrame({
    'color': ['red', 'blue', 'green', 'red', 'blue'],
    'value': [10, 20, 30, 12, 22]
})


# Sample data with some anomalous categorical features
validate_data = pd.DataFrame({
    'color': ['red', 'blue', 'yellow', 'purple', 'blue'],
    'value': [11, 21, 33, 15, 25]
})

# Generate statistics from training data
train_stats = tfdv.generate_statistics_from_dataframe(train_data)

# Infer schema from training stats
schema = tfdv.infer_schema(statistics=train_stats)

# Modify the schema to set a string domain for the 'color' feature
color_feature = tfdv.get_feature(schema, 'color')
color_feature.domain = schema_pb2.Domain(name='color_domain', value_list=schema_pb2.ValueList(strings=['red','blue','green']))

# Validate the data using the refined schema
validate_stats = tfdv.generate_statistics_from_dataframe(validate_data)
anomalies = tfdv.validate_statistics(statistics=validate_stats, schema=schema)

# Print the anomalies
tfdv.display_anomalies(anomalies)
```

In this code, following the same steps as Example 1, I generate a schema based on the training data. Then using `tfdv.get_feature` and the `schema_pb2` library, the schema is modified to specify a string domain for the `color` feature, defining the acceptable values. This provides more precise anomaly detection. Finally, validating using the modified schema identifies the specific `color` feature anomalies, demonstrating how TFDV can be customized to handle categorical values with predefined string domains, preventing unexpected values from affecting further processing.

**Example 3: Using Anomaly Locations and Filtering**

The final example will demonstrate how you can process the anomaly information to implement a filtering or extraction strategy. While TFDV does not directly support filtering or extraction, the locations of anomalous rows can be used in subsequent processing.

```python
import pandas as pd
import tensorflow_data_validation as tfdv
import numpy as np
from tensorflow_data_validation.statistics.stats_options import StatsOptions

# Sample training data
train_data = pd.DataFrame({
    'age': [25, 30, 40, 22, 28, 65],
    'income': [50000, 70000, 80000, 45000, 60000, 120000],
    'city': ['New York', 'London', 'Paris', 'New York', 'London', 'Tokyo']
})

# Sample data to validate against training data
validate_data = pd.DataFrame({
    'age': [23, 32, 45, 21, 27, 70, 'string'],
    'income': [52000, 72000, 90000, 46000, 62000, 130000, 'invalid'],
    'city': ['New York', 'London', 'Paris', 'New York', 'London', 'Seoul', '']
})

#Generate statistics from training data, setting num_top_values and num_histogram_buckets for better schema generation.
stats_options = StatsOptions(num_top_values=20, num_histogram_buckets=50)
train_stats = tfdv.generate_statistics_from_dataframe(train_data, stats_options=stats_options)

# Infer the schema from the training statistics.
schema = tfdv.infer_schema(statistics=train_stats)

# Validate the data against the schema
val_stats = tfdv.generate_statistics_from_dataframe(validate_data, stats_options=stats_options)
anomalies = tfdv.validate_statistics(statistics=val_stats, schema=schema)

# Get anomaly locations
anomaly_locations = tfdv.get_anomalies_dataframe(anomalies)

# If anomalies exist, print the indexes of anomaly containing rows.
if not anomaly_locations.empty:
    print("Anomalous row indexes:")
    print(anomaly_locations['example_index'].unique().tolist())
    # Extract the anomalous rows using a boolean mask.
    anomalous_rows = validate_data.iloc[anomaly_locations['example_index'].unique().tolist()]
    print("\nAnomalous data:")
    print(anomalous_rows)
else:
    print("No anomalies detected.")
```
Here, I followed similar steps to previous examples to create and validate data against a generated schema. Once anomalies are detected, I use `tfdv.get_anomalies_dataframe` to get the anomaly details including the row index, feature, and description. Using the dataframe returned, I extract the unique indices containing anomalies, then extract these rows from the validation dataframe for further processing. This approach demonstrates how, after anomaly identification, further processing can occur to extract or filter the data based on its compliance with the schema.

While TFDV doesn't directly output anomalous rows, it provides the machinery to *locate* such instances, which you can then use to extract or filter as needed. The `tfdv.get_anomalies_dataframe()` function proves useful in facilitating this.

**Resource Recommendations:**

For a detailed guide on leveraging TFDV, the official TensorFlow Data Validation documentation provides comprehensive tutorials and API references. This is the most thorough and up-to-date resource available. Further, the 'TensorFlow Extended (TFX) Guide' includes end-to-end examples using TFDV within a complete machine learning pipeline, detailing schema generation, anomaly detection, and data transformations. For understanding the statistical basis of schema generation, resources explaining basic statistical analysis in data science can be valuable, helping you better understand schema generation options. Finally, examples using TFDV for real-world use cases can be beneficial for implementation; often you can find these examples on various machine learning blogs and tutorials dedicated to TensorFlow and TFX.
