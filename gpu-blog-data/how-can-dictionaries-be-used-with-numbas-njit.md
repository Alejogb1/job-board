---
title: "How can dictionaries be used with Numba's `njit` for parallel acceleration?"
date: "2025-01-30"
id: "how-can-dictionaries-be-used-with-numbas-njit"
---
Dictionary usage within Numba-decorated functions, particularly those intended for parallel execution via `njit`, presents a unique challenge. Numba's core strength lies in its ability to compile Python code into optimized machine code, but this optimization is predicated on static typing. Standard Python dictionaries, inherently dynamic in their typing and structure, are therefore not directly compatible with Numba's Just-In-Time (JIT) compilation process. The crux of the issue is that Numba needs to know the types of keys and values at compile time to generate efficient machine code. This differs significantly from Python's runtime type resolution. I've spent considerable time optimizing numerical simulations and data processing pipelines using Numba, and the dictionary interaction is a frequent optimization hurdle I've encountered.

The primary strategy to circumvent this limitation involves avoiding the direct use of standard Python dictionaries within `njit`-decorated functions, especially where parallel execution via `parallel=True` is desired. Instead, one must opt for data structures that Numba can readily compile or, more specifically, ones that have consistent, known types at compilation. Several approaches facilitate dictionary-like behavior with Numba’s `njit` for parallel processing. They fall broadly into: leveraging typed Numba structures, translating to indexed arrays, and employing external data structures within a controlled scope.

The simplest approach, where applicable, involves pre-processing your dictionary into a form that Numba can handle more readily. This often means converting the dictionary to a series of NumPy arrays or Numba's typed data structures. I frequently employ this where a dictionary serves as a lookup table or mapping. Consider a scenario where you need to map categories in a time-series dataset to a numerical code for efficient processing.

**Code Example 1: Translation to Indexed Arrays**

```python
import numpy as np
from numba import njit, prange

@njit
def process_timeseries(time_series, category_keys, category_values, result):
    for i in prange(time_series.shape[0]):
        category = time_series[i]
        for j in range(len(category_keys)):
            if category_keys[j] == category:
                result[i] = category_values[j]
                break
    return result

# Example Data
categories = ["A", "B", "C", "A", "B", "C", "A"]
category_to_code = {"A": 1, "B": 2, "C": 3}
keys = np.array(list(category_to_code.keys()))
values = np.array(list(category_to_code.values()))
result_codes = np.empty(len(categories), dtype=np.int32)

# Execution
result = process_timeseries(np.array(categories), keys, values, result_codes)
print(f"Mapped codes: {result}")
```

This example translates the dictionary into two NumPy arrays: `category_keys` and `category_values`.  The `process_timeseries` function then iterates over these arrays, performing a lookup using a sequential comparison. Notice that the `prange` construct signals a parallelizable loop. It is crucial that `category_keys` and `category_values` have a fixed type and size. While the linear search is less optimal than the original dictionary lookup in Python, Numba's optimization and parallel execution offer significant speed gains for large datasets. I've observed order-of-magnitude speed-ups in my own simulations. It is essential, however, to avoid complex object types (including arbitrary Python classes) in the keys and values when using this method.

Another useful strategy is to leverage Numba's built-in typed containers, although they do not fully replicate Python's dictionary behavior. Numba's `typed.List` and `typed.Dict` classes, when used with type specifications, can provide a more direct means of managing data within `njit`-decorated functions.  However, crucial limitations exist. `typed.Dict` objects are not directly supported with `parallel=True`, and even when used without parallelization, modifications within `njit` are constrained. I've found they work well for cases requiring lookup or iteration over a pre-defined, static mapping.

**Code Example 2: Numba Typed List and Dictionary (without parallel)**

```python
from numba import njit, typed
import numpy as np

@njit
def lookup_codes(time_series, category_mapping):
  result_codes = np.empty(len(time_series), dtype=np.int32)
  for i in range(len(time_series)):
      category = time_series[i]
      if category in category_mapping:
          result_codes[i] = category_mapping[category]
      else:
         result_codes[i] = -1
  return result_codes

# Example Data
categories = ["A", "B", "C", "A", "B", "D", "A"]
category_to_code = typed.Dict.empty(key_type=typed.unicode_type, value_type=numba.int32)
category_to_code["A"] = 1
category_to_code["B"] = 2
category_to_code["C"] = 3
time_series = np.array(categories)

# Execution
result = lookup_codes(time_series, category_to_code)
print(f"Mapped codes: {result}")

```
In this example, a `typed.Dict` called `category_to_code` is created with type specifications for keys (unicode) and values (32-bit integers). This allows Numba to compile the `lookup_codes` function. I have deliberately avoided parallel execution here because of limitations on updating typed dictionaries in this way. Modifications to the dictionary within a parallel loop will typically raise an exception. Observe how we do not modify the dictionary inside the numba function. It is only used for lookup. Note also that `typed.Dict` is not fully equivalent to a Python dictionary; its use requires careful adherence to type constraints and a better understanding of its limitations.

Finally, one can also consider structuring the overall computation in a manner that isolates the dictionary usage outside the core parallel loop. I've applied this pattern in scenarios involving pre-computed data or reference information. One can pre-process lookup tables or maps outside `njit` and then pass the transformed structures into functions suitable for parallel processing.

**Code Example 3: Pre-computed Structures and Parallel Processing**

```python
from numba import njit, prange
import numpy as np
import pandas as pd

def preprocess_data(data):
    unique_categories = data["category"].unique()
    category_map = {cat: i for i, cat in enumerate(unique_categories)}
    return category_map, unique_categories

@njit(parallel=True)
def process_numerical_data(numerical_data, category_codes):
    rows = numerical_data.shape[0]
    results = np.zeros(rows)
    for i in prange(rows):
        category_index = category_codes[i]
        results[i] = numerical_data[i] * (category_index + 1)
    return results

#Example Data
data = pd.DataFrame({'category': ["A", "B", "C", "A", "B", "C", "A"], 'numerical_val':[1,2,3,4,5,6,7]})

# Preprocessing
category_map, categories = preprocess_data(data)
mapped_categories = np.array([category_map[cat] for cat in data["category"]])
numerical_data = data["numerical_val"].to_numpy()

# Execution
result = process_numerical_data(numerical_data, mapped_categories)
print(f"Processed data: {result}")
```
In this instance, the `preprocess_data` function, executed outside `njit`, generates the `category_map` which translates string category labels to numerical indexes and an array of `categories`. The `process_numerical_data` function, which *is* compiled with `njit`, receives *only* the processed arrays of numerical data and category indexes. Critically, all dictionary handling occurs *before* the computationally intensive part that is parallelized by Numba. This avoids the complexity associated with mutable data structures inside the `njit` decorated function, while retaining parallel processing benefits.  This approach effectively separates the preprocessing step from the core computational kernel.

For further exploration of Numba's capabilities, I highly recommend examining the official Numba documentation, particularly the sections on supported data types and parallelization. Additionally, the Numba examples repository and the discussions on the Numba issue tracker often provide practical insights into common usage scenarios and optimization strategies. Studying the various examples in the Numba documentation, with particular emphasis on parallel processing and supported types, will significantly improve understanding of Numba’s best practices for accelerated computation.
