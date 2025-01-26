---
title: "Why is TabularList undefined in my fastai code?"
date: "2025-01-26"
id: "why-is-tabularlist-undefined-in-my-fastai-code"
---

The `TabularList` class is no longer directly available in the `fastai` library after version 2. It was a key component of the v1 API for handling tabular data, but in the shift to a more modular and flexible design in v2 and later versions, its functionality was integrated into a combination of `TabularPandas` and the broader `DataBlock` API. Encountering an "undefined" error indicates that legacy code using the former `TabularList` is being executed with a newer `fastai` installation. I've personally wrestled with this transition numerous times while upgrading projects and mentoring less experienced colleagues, and the debugging process, while initially frustrating, consistently boils down to understanding the design shift.

The core issue is that `TabularList` was an early attempt at a complete data abstraction for tabular data. It was responsible for everything from loading data from a pandas DataFrame, to applying preprocessing steps like categorical encoding and normalization, and finally, to creating a `DataLoader`. The v2 and later approach separates these concerns more distinctly, providing greater control and customizability. Specifically, `TabularPandas` handles the data loading and preprocessing, and the `DataBlock` provides the overall structure for defining how the data is transformed into batches for training.

To replace the usage of `TabularList`, a two-step process is required: (1) the creation of a `TabularPandas` object, and (2) the definition of a `DataBlock` utilizing this object. `TabularPandas` is essentially a subclass of a Pandas DataFrame with additional metadata about the data, such as categorical or continuous variables, and allows for efficient manipulation and preprocessing. `DataBlock`, on the other hand, provides the blueprint for how to create `DataLoader` objects, detailing which columns are inputs, which are targets, and what transformations to apply to each. This decoupling enables creating highly customized data loading pipelines.

Here’s how this looks in practice, along with commentary:

**Code Example 1: Legacy `TabularList` (What NOT to do):**

```python
#This code will produce an error if using fastai v2 or later

from fastai.tabular import *
import pandas as pd

# Sample data (replace with your actual data)
data = {'col1': [1, 2, 3, 4, 5],
        'col2': ['a', 'b', 'a', 'c', 'b'],
        'target': [0, 1, 0, 1, 0]}
df = pd.DataFrame(data)

cat_names = ['col2']
cont_names = ['col1']
dep_var = 'target'

# This code is obsolete, it will raise an error
data_list = (TabularList.from_df(df, path=".", cat_names=cat_names, cont_names=cont_names)
            .split_by_idx(list(range(len(df) // 2)))
            .label_from_df(cols=dep_var)
            .databunch())

```

**Commentary on Example 1:**

This snippet shows the pre-v2 way of creating a `DataBunch` (the precursor to a `DataLoader`) from tabular data. The core line utilizing `TabularList` will throw an `AttributeError` in fastai v2 and later, because this class no longer exists in that manner. This demonstrates the precise situation prompting this error. The `from_df`, `split_by_idx`, `label_from_df`, and `databunch` functions were all chained together on a `TabularList` object.  This is a concise approach for simple problems but lacks customization and control.

**Code Example 2: Modern `TabularPandas` and `DataBlock` (Correct Usage):**

```python
from fastai.tabular.all import *
import pandas as pd

# Sample data (replace with your actual data)
data = {'col1': [1, 2, 3, 4, 5],
        'col2': ['a', 'b', 'a', 'c', 'b'],
        'target': [0, 1, 0, 1, 0]}
df = pd.DataFrame(data)

cat_names = ['col2']
cont_names = ['col1']
dep_var = 'target'

# 1. Create TabularPandas
procs = [Categorify, Normalize]
to = TabularPandas(df, procs=procs, cat_names=cat_names, cont_names=cont_names, y_names=dep_var)

# 2. Create DataBlock
dls = TabularDataLoaders.from_df(df, path=".",
                                cat_names=cat_names, cont_names=cont_names,
                                y_names=dep_var, procs=procs,
                                bs=2) # Specify the batch size for DataLoader


dls.show_batch()
```

**Commentary on Example 2:**

Here, I've replaced the `TabularList` approach with the `TabularPandas`/`DataBlock` approach.  First, we construct a `TabularPandas` object, `to`, specifying the categorical (`cat_names`), continuous (`cont_names`), and dependent variables (`y_names`). Also crucially, data processing steps such as `Categorify` and `Normalize` are defined as a list and provided as `procs` arguments.  Next, we create the `DataLoaders` using  `TabularDataLoaders`.  `from_df` takes the `DataFrame`, path, column names, the `procs`, and a batch size. This separation provides a clearer structure and enables the preprocessing steps to be done directly within the `TabularPandas`. This method provides greater control over data preparation.

**Code Example 3:  Explicit DataBlock (Alternative approach, more flexible):**

```python
from fastai.tabular.all import *
import pandas as pd

# Sample data (replace with your actual data)
data = {'col1': [1, 2, 3, 4, 5],
        'col2': ['a', 'b', 'a', 'c', 'b'],
        'target': [0, 1, 0, 1, 0]}
df = pd.DataFrame(data)

cat_names = ['col2']
cont_names = ['col1']
dep_var = 'target'

#1. Create DataBlock using TabularPandas
dblock = DataBlock(blocks=(TabularBlock(cat_names, cont_names), CategoryBlock),
                get_x=ColReader(cat_names + cont_names),
                get_y=ColReader(dep_var),
                splitter=RandomSplitter(valid_pct=0.2),
                procs=[Categorify, FillMissing, Normalize])

dls = dblock.dataloaders(df, bs=2)

dls.show_batch()
```

**Commentary on Example 3:**

This code offers the most flexible approach. Instead of relying on `TabularDataLoaders`, we are manually constructing a `DataBlock`. Here, the `blocks` argument defines the data input/target types, using `TabularBlock` for input columns and `CategoryBlock` for the target variable.  `get_x` and `get_y` define how to access input and target columns using `ColReader`. `splitter` defines how the data will be split into training and validation sets using `RandomSplitter`, providing more direct control than `split_by_idx`. Finally, processing steps are still defined in procs, which are now integrated with the data block.  Using a DataBlock object gives the programmer the most explicit control over how data is processed. Using explicit `DataBlock` structures offers greater flexibility for complex scenarios, where you may want to perform custom splits, transforms, and data augmentations.

**Resource Recommendations:**

For a deeper dive into the current `fastai` API, I would recommend consulting the official documentation. The API reference is thorough and is the best first-stop for any issues. I also found that reviewing the tutorials and notebooks provided on the fastai website useful, especially those focused on tabular data. The community forums can also be beneficial for seeing practical examples, including more intricate preprocessing pipelines, from other users, but should always be referenced with a critical eye.  Finally, examining source code of `fastai` modules on the project’s repository is useful for those seeking granular understanding of the library's internal processes, which is particularly useful when building complex workflows or addressing edge case issues. These resources will assist in navigating the nuances of the `fastai` framework and its evolution from v1 to current iterations.
