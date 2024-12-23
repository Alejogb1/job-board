---
title: "How do I resolve a 'NameError: name 'TabularList' is not defined' in fastai?"
date: "2024-12-23"
id: "how-do-i-resolve-a-nameerror-name-tabularlist-is-not-defined-in-fastai"
---

,  It’s not unusual to stumble upon `NameError` exceptions when diving into fastai, especially the specific one you’re seeing with `TabularList`. I remember a project a while back, working on a predictive maintenance model for some industrial equipment. We were using tabular data, and I hit this exact error while transitioning between fastai versions. It can be a bit frustrating, but it usually points to a straightforward issue: the class you’re trying to use isn’t accessible in the current scope, most often due to a missing import or an incorrect api usage. Let’s break down what this error typically means, and then I’ll give you some practical solutions with code.

The `NameError: name 'TabularList' is not defined` error indicates that the Python interpreter cannot find a variable, class, or function with the name `TabularList` within the current namespace. In fastai, `TabularList` used to be a foundational class for working with tabular data. However, things have evolved; fastai is quite actively developed, and certain structures have been reorganized or even replaced. The specific problem here usually arises because of changes to the fastai library's structure.

Historically, you would import it directly from something akin to `fastai.tabular`, but modern fastai encourages a slightly different approach. The primary issue is that `TabularList` has been deprecated and moved within the fastai library. It's not directly accessible in the way it used to be. What has replaced the class and the process is now located in the `fastai.tabular` module within the `TabularDataLoaders`.

So how do we actually go about fixing this? I'll give you three snippets, building up in complexity. These are not theoretical, but something I have had to do in practice.

**Example 1: The Basic Replacement (fastai v2 or later)**

The most straightforward fix is to use the new recommended approach utilizing `TabularDataLoaders`. If you’re using fastai v2 or a later version, `TabularList` is not needed anymore; you construct your dataloaders using the `TabularDataLoaders.from_csv` or similar methods directly. Here is how you'd structure the code:

```python
from fastai.tabular.all import *
import pandas as pd

# Assume you have a data.csv file
df = pd.read_csv('data.csv')

# Specify categorical and continuous variables
cat_names = ['categorical_column1', 'categorical_column2'] # Replace with your actual categorical column names
cont_names = ['continuous_column1', 'continuous_column2']  # Replace with your actual continuous column names

# Define the dependent variable name
dep_var = 'dependent_variable'  # Replace with your target variable column name

# Create the dataloaders
dls = TabularDataLoaders.from_df(df, path='.', y_names=dep_var,
                                 cat_names=cat_names, cont_names=cont_names,
                                 procs=[Categorify, FillMissing, Normalize])

dls.show_batch() # Optional, you can check your batch here
```

Here, we’ve used `TabularDataLoaders.from_df` to build our dataloaders directly from a pandas DataFrame. We specify the column names for categorical features, continuous features, and our dependent variable. The `procs` argument is essential – it's where we define preprocessing steps such as converting categorical variables to numerical representations via `Categorify`, handling missing values via `FillMissing`, and normalizing numeric features using `Normalize`. No more `TabularList`, and it's much cleaner. This was a core change in the api.

**Example 2: Using a Specific Split (fastai v2 or later)**

Sometimes, you don't want a random split. Suppose you need to use a specific column in your data to split train and validation sets. This might be the case when working with timeseries, where you need to make sure your validation set is always “after” your training data.

```python
from fastai.tabular.all import *
import pandas as pd

df = pd.read_csv('data.csv')

cat_names = ['categorical_column1', 'categorical_column2'] # Replace with your actual categorical column names
cont_names = ['continuous_column1', 'continuous_column2']  # Replace with your actual continuous column names
dep_var = 'dependent_variable'  # Replace with your target variable column name

# Define a function to split based on a specific column
def splitter(df):
    train_idx = df.index[df['split_column'] == 'train'].tolist() # 'split_column' can be any column name
    valid_idx = df.index[df['split_column'] == 'valid'].tolist()
    return train_idx, valid_idx

dls = TabularDataLoaders.from_df(df, path='.', y_names=dep_var,
                                 cat_names=cat_names, cont_names=cont_names,
                                 procs=[Categorify, FillMissing, Normalize],
                                 splitter=splitter)

dls.show_batch() # You can check your batch here
```

Here, we’ve defined a `splitter` function that leverages a column (`split_column`) in the DataFrame to determine which rows belong in the training and validation sets. This custom splitting is essential in many real-world scenarios where data must be split strategically. The key is providing the function to the `splitter` argument.

**Example 3: Legacy Compatibility (If you absolutely must use fastai v1 - not recommended)**

If you're somehow stuck with a codebase using fastai v1, and cannot upgrade, here’s how you might be able to use `TabularList` (though, it is **strongly** advised to upgrade for various reasons, including performance and better error handling):

```python
from fastai.tabular import *
import pandas as pd

df = pd.read_csv('data.csv')

cat_names = ['categorical_column1', 'categorical_column2'] # Replace with your actual categorical column names
cont_names = ['continuous_column1', 'continuous_column2'] # Replace with your actual continuous column names
dep_var = 'dependent_variable' # Replace with your target variable column name

data = (TabularList.from_df(df, path='.', cat_names=cat_names, cont_names=cont_names, procs=[Categorify, FillMissing])
          .split_by_rand_pct(0.2)
          .label_from_df(cols=dep_var)
          .databunch())

data.show_batch()  # check your batch
```

Notice, in this specific example, if you were running fastai v2 or later, it would throw a `NameError`. This snippet is provided for completeness but should not be a long-term solution, and again you are strongly encouraged to upgrade your version of fastai. You will also note that the API is quite different.

In summary, resolving the `NameError: name 'TabularList' is not defined` error in fastai usually entails migrating away from the older `TabularList` class and using the more modern `TabularDataLoaders` interface. These loaders, available in fastai v2 and later, provide a more flexible and robust mechanism for handling tabular data.

For further study, I’d recommend reviewing the official fastai documentation, which is very comprehensive and is the first stop for any fastai user: [https://docs.fast.ai/](https://docs.fast.ai/). There are also several excellent blog posts available that compare different approaches, including [https://forums.fast.ai/](https://forums.fast.ai/), where community members often discuss these changes and provide helpful insights. Specifically, I'd advise taking a close look at the tabular data section of the fastai docs. Moreover, the paper 'fastai: A Layered API for Deep Learning' by Howard and Gugger, which is available on arxiv, although quite technical, can provide deeper insight into the overall design philosophy behind the library and the reason for API changes.

Remember, software libraries evolve, and it's part of the development process to adapt to those changes. By staying updated and understanding the underlying principles, you'll become more effective in resolving issues and more efficient in your machine learning projects using tools like fastai.
