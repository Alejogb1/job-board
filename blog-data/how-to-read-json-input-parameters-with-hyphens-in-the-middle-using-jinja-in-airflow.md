---
title: "How to read JSON input parameters with hyphens in the middle using Jinja in Airflow?"
date: "2024-12-23"
id: "how-to-read-json-input-parameters-with-hyphens-in-the-middle-using-jinja-in-airflow"
---

Okay, let's tackle this. It's a common frustration when dealing with api payloads or configuration data and trying to parse it elegantly using Jinja within Airflow. The challenge stems from the fact that Jinja, by default, treats hyphens as subtraction operators, not as part of a property name. I’ve certainly run into this a few times, notably when we were integrating a legacy system that insisted on hyphenated field names in its responses. We had a pipeline trying to pull data from that, and it quickly became apparent that `{{ params.api-response-key }}` wasn't going to work. We had to find a more robust solution than just renaming keys. So, let's break down the approaches that work and why they work.

First and foremost, direct access via dot notation or bracket notation with simple string keys in Jinja won't function properly when encountering hyphens. Jinja attempts to interpret `params.api-response-key` as `params.api` minus `response` minus `key`. So, we need to bypass this interpretation.

The most straightforward method, and generally my preferred approach, is to use the `get()` method of dictionaries within Jinja. Here’s how it functions: instead of directly referencing the key with `params.my-key`, we call `params.get('my-key')`. This forces Jinja to interpret the argument as a string literal representing the key, avoiding the subtraction interpretation.

Let's look at a simple code example illustrating this:

```python
from airflow.models.dag import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def print_param(params):
    print(f"Parameter via get(): {params.get('api-response-key')}")
    # attempt to use the dot notation for contrast and show that it fails
    try:
        print(f"Parameter via dot notation: {params.api-response-key}")
    except Exception as e:
        print(f"Parameter via dot notation failed: {e}")


with DAG(
    dag_id="jinja_hyphen_params",
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False,
) as dag:
    print_task = PythonOperator(
        task_id="print_param_task",
        python_callable=print_param,
        op_kwargs={"params": {"api-response-key": "some_value"}}
    )
```

In this snippet, the `print_param` function demonstrates how `params.get('api-response-key')` successfully retrieves the value associated with the key, while the attempt to use `params.api-response-key` will result in an error during Jinja templating. This method is generally the most readable and my go-to method for simple cases. It directly tackles the problem without introducing complex workarounds.

Another potent technique involves using bracket notation with a string, which works analogously to the `get()` method. Instead of `params.api-response-key`, we would use `params['api-response-key']`. This also forces Jinja to treat `api-response-key` as a literal string key. This approach is often useful when you have a mix of keys that do and don't have hyphens, or if you prefer the syntax. Here's a Python example showing this:

```python
from airflow.models.dag import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def print_param_bracket(params):
    print(f"Parameter via bracket: {params['api-response-key']}")

with DAG(
    dag_id="jinja_hyphen_params_bracket",
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False,
) as dag:
    print_task = PythonOperator(
        task_id="print_param_bracket_task",
        python_callable=print_param_bracket,
        op_kwargs={"params": {"api-response-key": "another_value"}}
    )
```

Here the `print_param_bracket` function will successfully retrieve and print the value, thanks to the usage of bracket notation. While it achieves the same result as `get()`, it can sometimes be more concise in certain contexts. The choice often comes down to personal preference and readability in a given situation. Both techniques avoid Jinja's problematic interpretation of hyphens as subtraction operators.

A third, less common but still valuable solution, especially if you're dealing with a large number of hyphenated keys, is to pre-process the parameter dictionary using a jinja macro or function to convert all keys containing hyphens to something that Jinja can interpret directly. I typically use this when I need to use the dot notation and I absolutely can’t refactor the upstream api. For example, you could convert them to keys that use underscores. In this case, you would have to write the jinja macro yourself or provide a custom jinja environment that defines the macro.
```python
from airflow.models.dag import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from airflow.utils.context import Context
from jinja2 import Environment, FileSystemLoader, select_autoescape
from airflow.configuration import conf
from airflow.providers.common.sql.hooks.sql import DbApiHook
from typing import Any

def replace_hyphens_with_underscores(params: dict[str, Any]) -> dict[str, Any]:
    """Replaces hyphens with underscores in the keys of a dictionary."""
    new_params = {}
    for key, value in params.items():
        if isinstance(key, str):
            new_key = key.replace('-', '_')
        else:
            new_key = key
        new_params[new_key] = value
    return new_params

def print_param_preprocessed(params, **kwargs):
    context= kwargs
    new_params = replace_hyphens_with_underscores(params)
    print(f"Parameter via dot notation after preprocessing: {new_params.api_response_key}")



with DAG(
    dag_id="jinja_hyphen_params_preprocessing",
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False,
) as dag:

    print_task = PythonOperator(
        task_id="print_param_preprocessed_task",
        python_callable=print_param_preprocessed,
        op_kwargs={"params": {"api-response-key": "preprocessed_value"}},
        provide_context = True
    )
```

In this example, the `replace_hyphens_with_underscores` function is called to convert keys and then we can access the value using standard dot notation within the `print_param_preprocessed` function. You could further extend this method by including a Jinja environment that calls the python function as part of the template. While more involved, this method is beneficial if you have to standardize this behavior across many tasks.

In terms of further reading, I recommend looking at the official Jinja documentation. The specifics around filters, tests, and the templating engine itself are detailed there. Also, reading the 'Programming in Lua' by Roberto Ierusalimschy would give a deeper insight into how dynamic languages handle these sorts of key access patterns, since Jinja has strong influences from Lua. The 'Fluent Python' book by Luciano Ramalho can also help understanding data structure manipulation which can be useful when preprocessing your data.

In summary, the `get()` method and bracket notation are usually the most direct and clearest approaches for handling JSON input with hyphenated keys in Jinja within Airflow. Preprocessing is an option when the structure cannot be changed, but the goal is to allow for standard dot notation. Each approach offers a viable and effective solution, with the most suitable choice generally depending on the specifics of your project and preferred style. Remember to always prioritize clarity and maintainability when implementing these kinds of solutions in your workflows.
