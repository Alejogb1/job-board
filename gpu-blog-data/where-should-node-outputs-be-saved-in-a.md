---
title: "Where should node outputs be saved in a Kedro pipeline?"
date: "2025-01-30"
id: "where-should-node-outputs-be-saved-in-a"
---
The optimal location for saving node outputs within a Kedro pipeline hinges critically on the intended lifespan and accessibility requirements of that data.  My experience developing and maintaining large-scale data pipelines has shown that a rigid, one-size-fits-all approach is rarely effective.  Instead, a nuanced strategy incorporating Kedro's inherent flexibility is necessary.  This necessitates careful consideration of data lineage, version control, and the overall pipeline architecture.

**1. Clear Explanation**

Kedro's strength lies in its modular design, allowing for flexible data management.  While Kedro doesn't enforce a single directory structure for node outputs, adhering to a consistent, well-defined convention is crucial for maintainability and reproducibility.  My preferred strategy involves leveraging Kedro's built-in catalog and its ability to interact with various data stores.  This allows for a tailored approach based on the data characteristics.

The fundamental decision involves classifying data by its intended use:

* **Intermediate Data:**  Data generated and consumed exclusively within the pipeline. These outputs are typically ephemeral and don't require long-term storage or external accessibility.  These can be stored in a temporary directory, managed within the Kedro context, and automatically cleaned up after pipeline execution.  This minimizes disk space usage and avoids cluttering the project directory.

* **Persistent Data:** Data intended for reuse across multiple pipeline runs, analysis, or external consumption.  This data necessitates robust storage and versioning.  My recommendation is to use a dedicated directory within the project's `data` directory, structured by run ID or a timestamp to ensure traceability.  Consider using a version control system (like Git LFS) for large datasets to manage their evolution.  Furthermore, integrating with cloud storage solutions (like AWS S3 or Google Cloud Storage) becomes beneficial for scalability and accessibility.

* **Final Outputs (Results):**  The final outcome of the pipeline, representing the actionable insights or deliverables. These are typically stored in a dedicated `results` directory within the project's `data` directory, again with clear versioning and timestamping.  The format (e.g., CSV, Parquet, JSON) should align with the intended consumers of these results.  Consider employing a results database for enhanced searchability and management if dealing with many final outputs.

This layered approach ensures that data is managed effectively based on its significance and longevity.  The choice of storage (local file system, cloud storage, database) should align with the specific needs of each data category.  Overlooking this distinction often leads to disorganized projects and difficulties in managing and accessing generated data.


**2. Code Examples with Commentary**

The following examples illustrate saving node outputs to different locations, highlighting the versatility of Kedro's catalog and data handling capabilities.

**Example 1: Saving intermediate data to a temporary directory.**

```python
from kedro.io import MemoryDataset

def my_intermediate_node(data):
    # ...perform some operation...
    intermediate_result = #... the result...
    return intermediate_result

# In the Kedro Catalog:
my_intermediate_node:
    type: MemoryDataset
```

This example leverages `MemoryDataset` to store intermediate results in memory, avoiding persistent storage and disk I/O overhead.  This is suitable for temporary data that won't be used beyond the current pipeline run.  Cleaning up happens implicitly when the pipeline completes.  Note that for larger datasets this might not be memory efficient and a temporary file-based solution would be preferable within Kedro's temporary directory structure.

**Example 2: Saving persistent data to a versioned directory.**

```python
from kedro.io import CSVDataSet
import pandas as pd
from datetime import datetime

def my_persistent_node(data):
    # ...perform operations...
    persistent_data = pd.DataFrame(...)  # Example DataFrame
    return persistent_data

# In the Kedro Catalog:
my_persistent_node:
    type: CSVDataSet
    filepath: data/persistent/output_{run_id}.csv
```

This demonstrates saving the output as a CSV file to the `data/persistent` directory.  The `{run_id}` placeholder ensures that each pipeline run creates a uniquely identified file, avoiding overwrites and providing clear version control.  Using a timestamp instead of `run_id` provides similar versioning.  Replacing `CSVDataSet` with other data sets allows flexible persistence.

**Example 3: Saving final results to a dedicated results directory.**

```python
from kedro.io import JSONDataSet

def my_final_results_node(data):
    # ...perform operations...
    final_results = {"key1": value1, "key2": value2}
    return final_results

# In the Kedro Catalog:
my_final_results_node:
    type: JSONDataSet
    filepath: data/results/final_results_{run_id}.json
```

This example saves the final results as a JSON file in the `data/results` directory, again leveraging `{run_id}` for versioning.  The choice of JSON is suitable for structured data.  For other data types, such as images or large matrices, the appropriate Kedro data set needs to be used.  Consider integrating with a results database for sophisticated result tracking and retrieval if needed.


**3. Resource Recommendations**

For a deeper understanding of data management within Kedro, I highly recommend consulting the official Kedro documentation.  Thoroughly reviewing the available datasets and their functionalities will be extremely beneficial.  Understanding the concepts of data lineage and reproducibility is crucial for designing effective pipelines.  Furthermore, exploring the best practices outlined in software development literature on data management and version control will prove invaluable.  Finally, familiarizing yourself with various data storage solutions (local file systems, cloud storage, databases) and their respective strengths and weaknesses is paramount.  These resources provide a comprehensive foundation for architecting robust and maintainable Kedro pipelines.
