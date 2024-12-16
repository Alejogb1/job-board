---
title: "Why am I getting a bad request when searching runs in MLflow?"
date: "2024-12-16"
id: "why-am-i-getting-a-bad-request-when-searching-runs-in-mlflow"
---

Alright, let's tackle this one. It's a classic issue and, believe me, I’ve spent my fair share of time debugging those pesky 400 bad request errors when querying mlflow runs. It usually boils down to a handful of culprits, and while the error message itself is rather generic, the root cause is typically more specific. I'm going to walk you through the typical scenarios I’ve encountered, how to diagnose them, and provide some actionable solutions, with a little code to demonstrate what’s happening.

The core of the problem is, simply put, that the server receiving your request doesn't understand the parameters of the request you’re sending. In the context of MLflow, this often revolves around the query filters used when searching for runs. MLflow's search functionality is powerful, but it has rules and expectations for those filters that, if not met, result in a bad request.

First off, a common issue is *syntax errors in the filter string itself*. The filter string uses a specific syntax, not unlike a database query, to select runs based on their associated data. For example, you might want to retrieve runs where `metrics.accuracy > 0.9` and `params.model_type = 'cnn'`. But, a simple typo or incorrect operator can throw off the parser. If you’re missing quotes around string values, using an operator that isn’t supported for a given type, or even have stray characters, this is a guaranteed 400 error.

Let's look at a simple case using the mlflow python api:

```python
import mlflow

# Incorrect filter - missing quotes around 'cnn'
try:
    runs = mlflow.search_runs(filter_string="params.model_type = cnn")
except mlflow.exceptions.MlflowException as e:
    print(f"Error: {e}")
# Correct filter
runs = mlflow.search_runs(filter_string="params.model_type = 'cnn'")
print(f"Successfully retrieved {len(runs)} runs")
```

In this example, the first query throws a `MlflowException` because the string value `cnn` isn't enclosed in quotes. The second attempt, with quotes, works as expected. It’s a small change, but crucial. Keep in mind the proper syntax should match the type of data you're querying. Numerics don't need quotes, strings do, and boolean values should be represented as `true` or `false` (again, without quotation marks).

Another frequent cause is attempting to *filter on non-existent fields*. You might be trying to filter based on a metric or parameter that was never actually logged for the runs in question. This leads to the server not knowing how to interpret your filter because it can't find the requested keys in its data. MLflow does not throw an informative error in such cases; instead, it simply rejects the request.

Let’s look at a case where I'm trying to query against a non-existent metric, and then how to approach it:

```python
import mlflow
import mlflow.pyfunc
import pandas as pd

# Create a dummy run with one metric
with mlflow.start_run() as run:
    mlflow.log_metric("actual_accuracy", 0.95)
    # Create a dataframe and log it as an artifact
    data = {'col1': [1,2], 'col2':[3,4]}
    df = pd.DataFrame(data)
    mlflow.pyfunc.save_model(path="model_save",
                             python_model=df)

try:
    runs = mlflow.search_runs(filter_string="metrics.bogus_metric > 0.8")
except mlflow.exceptions.MlflowException as e:
    print(f"Error: {e}")

# Correct filter, using the existing metric
runs = mlflow.search_runs(filter_string="metrics.actual_accuracy > 0.8")
print(f"Successfully retrieved {len(runs)} runs")
```

The first query fails with a 400 because there's no metric named `bogus_metric` within the run. Always verify the exact names and case of the metrics and parameters that were logged in the experiments. Use the mlflow UI or mlflow's python api to check what you are looking for, to avoid this type of error.

Beyond syntax and nonexistent fields, *incorrect data types in the filter* can cause problems. When you try to compare a string value to a numeric value, or a numeric to a boolean, MLflow will respond with a bad request. It's important to know what data type each logged value is and to use matching types in your query. Be aware that MLflow handles logging of parameters as strings; you should be filtering string types or cast them to the appropriate type.

Finally, consider issues with the *search limit*. Although this isn’t the typical cause of a 400 error for me, if you are requesting an enormous number of runs without paging it can be a problem as well. MLflow allows you to set the `max_results` parameter and, if you are requesting more than the default limit of 1000, then you must also consider using `order_by` to guarantee a consistent return of runs.

Let’s see what happens when I go above the default limit:

```python
import mlflow

# Creating 1100 dummy runs
for i in range(1100):
  with mlflow.start_run():
    mlflow.log_metric("test_metric", i)

try:
    runs = mlflow.search_runs(filter_string="metrics.test_metric > -1")

    print(f"Retrieved {len(runs)} runs")

    # Requesting with limit
    runs = mlflow.search_runs(filter_string="metrics.test_metric > -1", max_results=1100, order_by=["metrics.test_metric desc"])
    print(f"Retrieved {len(runs)} runs")
except mlflow.exceptions.MlflowException as e:
  print(f"Error: {e}")
```

The first attempt requests all runs without specifying `max_results` or `order_by` and results in 1000 runs being returned since the default behavior of MLflow limits results to 1000. The second request returns 1100 results, the expected amount when both the limit and ordering are provided.

When encountering a 400 bad request with MLflow, methodical debugging is crucial. First, double-check the syntax of your filter string. Refer to the official MLflow documentation for the correct operators, functions, and type compatibility. The "MLflow Tracking API" documentation section details the filter syntax specifics. For a deep dive, I would recommend looking into the *"Building Machine Learning Pipelines"* book by Hannes Hapke and Catherine Nelson. This will clarify the concepts around artifact and metadata handling in ML pipelines, providing crucial context for why these queries matter. Next, verify that your filter fields actually exist, and are correctly cased. Finally, make sure your data types in the filters match the logged types, and always specify `max_results` and `order_by` if you're expecting a very large result set. This step-by-step approach and the example code I provided should help you isolate and resolve your problem. I’ve found these issues to be the most common causes. By addressing these areas, you should be able to effectively search for runs in MLflow without the frustration of bad request errors.
