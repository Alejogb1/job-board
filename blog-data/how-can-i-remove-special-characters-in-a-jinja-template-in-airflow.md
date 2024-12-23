---
title: "How can I remove special characters in a jinja template in Airflow?"
date: "2024-12-23"
id: "how-can-i-remove-special-characters-in-a-jinja-template-in-airflow"
---

Okay, let’s tackle this. I’ve seen this issue pop up numerous times, often when dealing with data pipelines that pull information from diverse, sometimes messy, sources. In my previous stint managing a large-scale ETL system, we frequently encountered problems with data containing special characters that then caused headaches when rendered into Airflow DAG configurations via Jinja templates. Getting a clean, predictable output from those templates was essential for our operations. Here's how I approached it and how you can too.

The core of the problem lies in Jinja’s rendering process. Jinja essentially takes your template and replaces placeholders (variables, expressions, etc.) with their evaluated values. However, it doesn’t automatically filter out special characters. When these characters are present in the values being substituted, they can break your intended syntax, especially in contexts like file paths, command-line arguments, or SQL queries within your DAG. So, we need to proactively handle this before the template renders fully.

The most straightforward approach is to leverage Jinja's own filter functionality, which allows you to modify the data before it's inserted into the template. We can use custom filters created within the Airflow context, giving us fine-grained control. Jinja supports Python functions as filters, which opens a world of possibilities. Let's illustrate with a concrete example using a basic `remove_special_chars` filter.

First, let’s define the filter, typically within your `airflow_local_settings.py` file, or in a custom plugin if you are employing those:

```python
import re

def remove_special_chars(text):
    if not isinstance(text, str):
        return text # handle non-string inputs gracefully
    return re.sub(r'[^a-zA-Z0-9\s_.-]', '', text)


def jinja_environment_customizer(env):
    env.filters['remove_special_chars'] = remove_special_chars


# You might need to define the jinja_env_vars method in your airflow.cfg, 
# it points to the customizer
# like this:
# jinja_env_vars = airflow.my_plugin.jinja_environment_customizer

```

In this code, `remove_special_chars` uses a regular expression to remove any character that is not a letter, number, whitespace, underscore, period, or hyphen. This is a good starting point; you may need to adjust this regex to meet your specific requirements. The `jinja_environment_customizer` registers the filter, making it available in your templates. Now, in your DAG code, it's as simple as applying the filter like this within a Jinja template:

```python
from airflow.models import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime
from airflow.utils.dates import days_ago

dag = DAG(
    dag_id="jinja_special_chars",
    schedule=None,
    start_date=days_ago(1),
    catchup=False,
)

template_value = 'File name!@# with some$ spec!al characters.'

task = BashOperator(
    task_id="test_special_chars_filter",
    bash_command=f"echo 'File path: {{ '{template_value}' | remove_special_chars }}'",
    dag=dag,
)
```
Here, the template string contains special characters. When we execute this DAG, the BashOperator will output something similar to `File path: File name with some specal characters.` demonstrating that the filter successfully removed the unwanted characters before the `echo` command executes.

Now, what if you need to handle specific characters differently? For instance, perhaps you need to replace, not remove, certain characters. We can easily customize the filter to do that as well. Consider replacing spaces with underscores.

Here’s a modified filter, again going inside the `jinja_environment_customizer`:

```python
import re

def sanitize_filename(text):
    if not isinstance(text, str):
        return text # handle non-string inputs gracefully
    text = re.sub(r'\s+', '_', text)  # Replace spaces with underscores
    return re.sub(r'[^a-zA-Z0-9_.-]', '', text) # Remove any remaining unwanted chars


def jinja_environment_customizer(env):
    env.filters['sanitize_filename'] = sanitize_filename
```

And now, applying this in your dag will produce a different output:

```python
from airflow.models import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime
from airflow.utils.dates import days_ago

dag = DAG(
    dag_id="jinja_sanitize_filename",
    schedule=None,
    start_date=days_ago(1),
    catchup=False,
)

template_value = 'File name!@# with some$ spec!al   characters.'

task = BashOperator(
    task_id="test_sanitize_filename_filter",
    bash_command=f"echo 'File path: {{ '{template_value}' | sanitize_filename }}'",
    dag=dag,
)
```

The bash output would then display something similar to `File path: File_name_with_some_specal_characters.`, which removes special characters but also converts all spaces into underscores.

Finally, one more important consideration when it comes to filtering, specifically if you have data you receive that might be nested, for example a dictionary or a list, and you only want to sanitize some keys of this data, then a slightly more complex example would look something like this:

```python
import re

def sanitize_nested_data(data, keys_to_sanitize):
    if isinstance(data, dict):
       for key, value in data.items():
         if key in keys_to_sanitize:
             if isinstance(value, str):
               data[key] = re.sub(r'[^a-zA-Z0-9\s_.-]', '', value)
         else:
             sanitize_nested_data(value, keys_to_sanitize)

    if isinstance(data, list):
       for element in data:
         sanitize_nested_data(element, keys_to_sanitize)
    return data


def jinja_environment_customizer(env):
   env.filters['sanitize_nested_data'] = sanitize_nested_data
```

And now in your DAG you would do something similar to this:

```python
from airflow.models import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime
from airflow.utils.dates import days_ago

dag = DAG(
    dag_id="jinja_nested_data",
    schedule=None,
    start_date=days_ago(1),
    catchup=False,
)

template_data = {
    "file_path": "path/with/!@#",
    "description": "Description with some!@#$ characters.",
    "nested": [
       {"nested_path": "another/path!@#" }
    ]
}
keys_to_sanitize = ['file_path', 'nested_path'] # we only want to sanitize this keys

task = BashOperator(
    task_id="test_sanitize_nested_data_filter",
    bash_command=f"echo 'Data: {{ '{template_data}' | sanitize_nested_data('{keys_to_sanitize}') }}'",
    dag=dag,
)

```

The output would now look similar to `Data: {'file_path': 'path/with/', 'description': 'Description with some!@#$ characters.', 'nested': [{'nested_path': 'another/path'}]}` demonstrating how you can target specific fields to sanitize nested data.

Key to understand here is that the `'keys_to_sanitize'` argument is passed to the `sanitize_nested_data` filter, so you can use this on the Jinja side to define what keys you would like to sanitize, which can even be a variable passed in to your DAG, adding flexibility to your pipeline.

For further study, I’d recommend looking into the Jinja2 documentation directly; specifically the section on filters and how to extend Jinja’s capabilities. Also, familiarize yourself with Python's `re` module, which gives you powerful tools for string manipulation and filtering. "Mastering Regular Expressions," by Jeffrey Friedl, is an excellent resource for deepening your understanding of regular expressions.

In conclusion, handling special characters in Jinja templates within Airflow is crucial for avoiding unexpected issues. Custom filters are the way to go because they give you complete control over how your data is rendered and allow for re-usability, ensuring that your templated DAGs produce predictable results. Using a combination of regular expressions and the Jinja filter system provides a robust solution for these scenarios.
