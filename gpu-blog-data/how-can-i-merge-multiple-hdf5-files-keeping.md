---
title: "How can I merge multiple HDF5 files, keeping only rows that are unique based on a custom metric?"
date: "2025-01-30"
id: "how-can-i-merge-multiple-hdf5-files-keeping"
---
Merging multiple HDF5 files while ensuring row uniqueness based on a custom metric requires careful data handling to avoid excessive memory consumption and maintain data integrity. The core challenge resides in efficiently identifying and discarding duplicate rows across all input files based on a user-defined function. The most effective approach I've found involves reading data in manageable chunks, comparing new chunks against previously seen unique rows, and then writing the merged, de-duplicated data to a new HDF5 file.

The process, as I've developed it over several projects involving large-scale scientific datasets, can be broken down into several distinct stages: file iteration, chunked data reading, custom metric calculation and comparison, unique row tracking, and finally, output file writing. It is crucial to use HDF5's features to advantage, particularly the ability to incrementally append to datasets, rather than loading an entire dataset into memory.

Let's examine the technical specifics. First, we need to loop through the input HDF5 files. Within each file, we'll iterate through the target dataset in chunks. The size of each chunk is crucial; choosing it appropriately is key to memory efficiency and processing speed. Reading large chunks can potentially lead to out-of-memory errors, while smaller chunks add overhead through repeated function calls and data access. A good starting point is a few megabytes of data, but ultimately the size will depend on the number of rows, number of columns, and data type for each row.

Once a chunk of data is loaded, the custom uniqueness metric is computed for each row. This metric calculation is user-defined, allowing flexibility for different data types and uniqueness criteria. The simplest metric might be a hash of the row's contents, or it might involve a more complex calculation. The computed metric for each row is then used to determine if it is already present in the set of previously observed unique rows. The comparison requires a data structure that facilitates fast lookups; a Python set, in my experience, provides excellent performance for these kinds of tasks. If the metric of a row is not found within this structure, it’s considered a new unique row, and is added to both the set of observed rows, and the output dataset.

Finally, the unique row is added to the output HDF5 file. Crucially, I append data to the output dataset in chunks, avoiding the memory costs associated with keeping the entire result dataset in memory. By repeatedly reading chunks from input, determining the uniqueness based on the metric, and appending the unique rows, we can handle very large datasets without requiring memory equal to the full size of merged datasets.

Now, let’s consider some concrete examples, using Python with the `h5py` library, which I find to be the most convenient for HDF5 data manipulation.

```python
import h5py
import numpy as np
import hashlib

def hash_metric(row):
  """Example custom metric - hash of the row data"""
  return hashlib.sha256(row.tobytes()).hexdigest()


def merge_h5_files(input_files, output_file, dataset_name, chunk_size=1000):
  """Merges h5 files, keeping only unique rows based on the hash metric."""
  seen_metrics = set()
  first_file_processed = False

  with h5py.File(output_file, 'w') as outfile:
    for filename in input_files:
      with h5py.File(filename, 'r') as infile:
          dataset = infile[dataset_name]
          num_rows = dataset.shape[0]

          for start in range(0, num_rows, chunk_size):
              end = min(start + chunk_size, num_rows)
              chunk = dataset[start:end]
              unique_rows = []

              for row in chunk:
                  metric = hash_metric(row)
                  if metric not in seen_metrics:
                    seen_metrics.add(metric)
                    unique_rows.append(row)

              if unique_rows:
                  if not first_file_processed:
                      # create output dataset, inheriting data type from first chunk
                      dtype = dataset.dtype
                      shape = (0,) + unique_rows[0].shape
                      out_dset = outfile.create_dataset(dataset_name, dtype=dtype, shape=shape, maxshape=(None,) + unique_rows[0].shape)
                      first_file_processed = True

                  # Append unique rows to output dataset
                  current_size = out_dset.shape[0]
                  new_rows = np.array(unique_rows)
                  out_dset.resize((current_size + new_rows.shape[0],) + new_rows.shape[1:])
                  out_dset[current_size:] = new_rows

```

In this first example, the `hash_metric` function calculates a SHA256 hash of each row. The `merge_h5_files` function iterates through input files, reads data in chunks, checks for uniqueness, and appends unique rows to an output dataset. The use of a `set` for `seen_metrics` ensures fast lookups, and dataset resizing enables incremental writing. This method also supports varying shapes along the first axis, given that data type and column shapes remain consistent between input files.

Now let's consider a scenario where, rather than a simple hash, we need to compare rows based on a subset of columns. Assume we want to treat rows as duplicates based solely on values in the first three columns.

```python
import h5py
import numpy as np
import hashlib

def column_metric(row, columns=[0, 1, 2]):
  """Custom metric based on a subset of columns"""
  key_values = [row[col] for col in columns]
  return hashlib.sha256(str(key_values).encode()).hexdigest()

def merge_h5_files_column_metric(input_files, output_file, dataset_name, columns, chunk_size=1000):
   """Merges h5 files, keeping only unique rows based on a specific column subset"""
   seen_metrics = set()
   first_file_processed = False

   with h5py.File(output_file, 'w') as outfile:
      for filename in input_files:
          with h5py.File(filename, 'r') as infile:
            dataset = infile[dataset_name]
            num_rows = dataset.shape[0]

            for start in range(0, num_rows, chunk_size):
                end = min(start + chunk_size, num_rows)
                chunk = dataset[start:end]
                unique_rows = []

                for row in chunk:
                    metric = column_metric(row, columns)
                    if metric not in seen_metrics:
                        seen_metrics.add(metric)
                        unique_rows.append(row)

                if unique_rows:
                    if not first_file_processed:
                        dtype = dataset.dtype
                        shape = (0,) + unique_rows[0].shape
                        out_dset = outfile.create_dataset(dataset_name, dtype=dtype, shape=shape, maxshape=(None,) + unique_rows[0].shape)
                        first_file_processed = True
                    current_size = out_dset.shape[0]
                    new_rows = np.array(unique_rows)
                    out_dset.resize((current_size + new_rows.shape[0],) + new_rows.shape[1:])
                    out_dset[current_size:] = new_rows
```

This second example modifies the metric function to consider only the specified columns when determining row uniqueness. The `column_metric` function now extracts values from the desired columns before computing the hash. The `merge_h5_files_column_metric` remains otherwise structurally identical to the original function.

Finally, it might be that your uniqueness criteria needs to consider a normalized version of the data. In this case, your custom metric function would take care of that before comparing. Here is the final example that showcases this.

```python
import h5py
import numpy as np
import hashlib

def normalized_metric(row, normalization_cols, normalize_fn = lambda x: x/np.sum(x)):
    """Computes a normalized metric before hashing"""
    
    normalized_data = np.array([normalize_fn(row[col]) if col in normalization_cols else row[col] for col in range(len(row))])
    return hashlib.sha256(normalized_data.tobytes()).hexdigest()

def merge_h5_files_normalized_metric(input_files, output_file, dataset_name, normalization_cols, chunk_size=1000):
   """Merges h5 files, keeping unique rows after normalization."""
   seen_metrics = set()
   first_file_processed = False

   with h5py.File(output_file, 'w') as outfile:
      for filename in input_files:
          with h5py.File(filename, 'r') as infile:
            dataset = infile[dataset_name]
            num_rows = dataset.shape[0]
            for start in range(0, num_rows, chunk_size):
                end = min(start + chunk_size, num_rows)
                chunk = dataset[start:end]
                unique_rows = []
                for row in chunk:
                    metric = normalized_metric(row, normalization_cols)
                    if metric not in seen_metrics:
                        seen_metrics.add(metric)
                        unique_rows.append(row)
                if unique_rows:
                   if not first_file_processed:
                      dtype = dataset.dtype
                      shape = (0,) + unique_rows[0].shape
                      out_dset = outfile.create_dataset(dataset_name, dtype=dtype, shape=shape, maxshape=(None,) + unique_rows[0].shape)
                      first_file_processed = True

                   current_size = out_dset.shape[0]
                   new_rows = np.array(unique_rows)
                   out_dset.resize((current_size + new_rows.shape[0],) + new_rows.shape[1:])
                   out_dset[current_size:] = new_rows
```

In this final example, the `normalized_metric` function calculates a normalization, in this case summing to one, of specified columns and hashes the normalized data. This metric handles more complex duplication criteria based on normalization parameters.

For a deeper understanding, I recommend exploring the official h5py documentation, as well as resources on efficient memory management in Python with NumPy. Additionally, reading materials covering techniques for designing custom hash functions would be beneficial, specifically when you move away from using simple hashes such as provided by `hashlib`. Understanding data type considerations when operating with NumPy will also be highly beneficial to the process. These resources, combined with the approaches described above, should be more than sufficient for effectively merging and de-duplicating your HDF5 datasets based on your unique use case.
