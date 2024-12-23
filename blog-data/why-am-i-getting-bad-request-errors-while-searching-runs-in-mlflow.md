---
title: "Why am I getting bad request errors while searching runs in mlflow?"
date: "2024-12-23"
id: "why-am-i-getting-bad-request-errors-while-searching-runs-in-mlflow"
---

Alright, let’s get into this. Bad request errors during mlflow run searches can be surprisingly frustrating, especially when everything *appears* to be configured correctly. I've personally chased down a few of these gremlins over the years, and they often boil down to a mismatch between what you're *asking* mlflow to find and what it's actually *capable* of delivering given the data it has, or how you're phrasing your requests.

The core issue, in my experience, frequently stems from the structure and content of the search criteria you’re providing to mlflow’s search_runs() function. This function, while powerful, is quite particular about the query syntax and data types. The errors themselves might seem vague at first, but they usually point towards a few common pitfalls. Let's break them down.

First, let's consider the ‘filter’ parameter of the search_runs() function. This is where you specify conditions for filtering runs based on their metadata, parameters, or metrics. Mlflow's query language isn't a full-blown SQL dialect, and it follows a specific grammar that, if violated, will result in a bad request. For example, if you're filtering based on a numeric metric, attempting to apply a string operation or comparing it against a non-numeric value will undoubtedly cause an error. I recall debugging a system where a junior team member mistakenly tried to use the "LIKE" operator on a metric field that was stored as float, and it resulted in a long stream of bad request errors.

Another frequent culprit is data type mismatches when using list comprehensions or variable substitutions in the filter string. Let's say you have a list of experiment IDs that you're trying to filter against. If these IDs are represented as integers in the mlflow backend, but your query treats them as strings, it's not going to work. It's a subtle difference that’s often overlooked.

Let's illustrate this with a code snippet demonstrating the right way and the wrong way to approach filtering.

```python
import mlflow
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Correct example using integer comparisons for an integer parameter.
experiment_ids = [1, 2, 3] # Assuming these are valid integer experiment_ids
runs = client.search_runs(
    experiment_ids=experiment_ids,
    filter=f"params.learning_rate > 0.01 and params.epochs < 10",
    view_type=ViewType.ACTIVE_ONLY
)
print(f"Found {len(runs)} runs matching correct filter.")

# Incorrect example using string comparison on a numerical parameter.
try:
  runs = client.search_runs(
    experiment_ids=experiment_ids,
    filter=f"params.learning_rate like '0.0%'", # Incorrect string comparison
    view_type=ViewType.ACTIVE_ONLY
  )
  print("This line should not be executed") # Should not happen due to error.
except Exception as e:
    print(f"Error during incorrect filtering example: {e}")

# Incorrect example using a string for a comparison against an integer
try:
  runs = client.search_runs(
    experiment_ids=experiment_ids,
    filter=f"params.epochs = '10'", #incorrect use of string for comparison.
    view_type=ViewType.ACTIVE_ONLY
  )
  print("This line should also not be executed.")
except Exception as e:
    print(f"Error during incorrect filtering (string number): {e}")
```

In the first example, `params.learning_rate > 0.01` and `params.epochs < 10` are both valid since they are using numerical comparisons with numerical parameters. In the second example,  `params.learning_rate like '0.0%'` is incorrect and results in an error, because like is a string operation. Similarly, the third one `params.epochs = '10'` is incorrect as we are comparing integer field to string value. Always be mindful of the underlying types of the data you are filtering against.

Another common source of bad requests lies in the way you specify experiment IDs. If you pass a list of strings as experiment ids when mlflow expects integers, you will encounter a bad request error. The mlflow UI can show ids as strings when displayed, but the internal representation in the storage backend is typically numeric.

Here’s a snippet highlighting this issue.

```python
import mlflow
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient

client = MlflowClient()

#Assume that experiment_ids are integers.
experiment_ids_integer = [1, 2, 3]

#Correct use of integers as experiment_ids
runs = client.search_runs(
    experiment_ids=experiment_ids_integer,
    view_type=ViewType.ACTIVE_ONLY
)
print(f"Found {len(runs)} runs with integer experiment ids.")

#Incorrect use of string format as experiment ids.
experiment_ids_string = ['1', '2', '3']
try:
  runs = client.search_runs(
    experiment_ids=experiment_ids_string,
    view_type=ViewType.ACTIVE_ONLY
  )
  print("This should not be executed.")
except Exception as e:
  print(f"Error during filtering with string experiment ids: {e}")
```

This code shows clearly, how using correct integer based experiment ids does not cause an error, while using a list of string ids results in an error. The critical point is that when using ids with the mlflow client, they must be in the correct format, so be mindful of the storage schema of the mlflow backend you are using.

Lastly, be cautious when dealing with parameters that are not consistently logged across all runs. If your filter query references a parameter that's missing in some of the runs within the specified experiments, you will likely encounter a bad request error.  Mlflow expects that all the fields that are referenced in the `filter` string must exists in all the runs. It doesn't handle this scenario gracefully. So, before creating your filter, make sure that the parameters you are using exist in the run metadata. Using a try-except block to handle any exceptions arising from `search_runs()` is always a good practice.

```python
import mlflow
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient

client = MlflowClient()

experiment_ids = [1,2] #Valid ids. Assume not all runs have `my_custom_param`
try:
  # Incorrect usage - some runs might not have 'my_custom_param'
  runs = client.search_runs(
    experiment_ids=experiment_ids,
    filter="params.my_custom_param > 0.5",
    view_type=ViewType.ACTIVE_ONLY
  )
  print(f"Found {len(runs)} matching custom param filter. May produce an error.")

except Exception as e:
  print(f"Error during run search with possibly missing parameter: {e}")

#Correct usage assuming there are some runs with my_custom_param, otherwise no runs will be returned
#This can also be handled by creating more robust checks before the mlflow run search.
try:
  runs = client.search_runs(
    experiment_ids=experiment_ids,
    filter="params.my_custom_param > 0.5",
    view_type=ViewType.ACTIVE_ONLY,
  )
  if len(runs)>0:
    print(f"Found {len(runs)} runs with my_custom_param, using correct filter.")
  else:
    print("No runs were found with the specified filter")
except Exception as e:
    print(f"Error during run search with possibly missing parameter: {e}")
```

In short, these errors usually arise from mismatches in data types and query syntax. Always double-check your filter strings, verify data types, and make sure that all the parameters used in the filter are present in the run metadata.

For a deeper dive into this area, I recommend consulting the official mlflow documentation, particularly the sections on `mlflow.search_runs()` and its query language, which can be found online. Additionally, the book *Machine Learning Engineering* by Andriy Burkov offers some excellent insights on practical aspects of managing machine learning experiments using tools such as mlflow. Another valuable resource, specifically when understanding how to build and structure scalable machine learning pipelines, is *Designing Machine Learning Systems* by Chip Huyen, which provides a lot of practical tips.

Hopefully, this explanation helps you navigate the complexities of mlflow run searches and avoid those pesky bad request errors.
