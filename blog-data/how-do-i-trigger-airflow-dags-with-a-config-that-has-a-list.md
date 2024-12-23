---
title: "How do I trigger Airflow DAGs with a config that has a list?"
date: "2024-12-23"
id: "how-do-i-trigger-airflow-dags-with-a-config-that-has-a-list"
---

Alright, let's dive into triggering Airflow DAGs with configurations that include lists. I’ve tackled this particular scenario quite a few times in past projects, and it usually crops up when dealing with dynamically generated workflows or situations where you need to parameterize processing steps. You're not alone in hitting this, and there are definitely some elegant approaches we can use.

The core challenge is that Airflow, by default, prefers simple key-value pairs for configurations. Passing a list, while seemingly straightforward, requires a bit more finesse to ensure the values are correctly parsed and accessible within the DAG’s context. We need to think about how we're going to pass this data during the trigger process and, more importantly, how to handle it within the DAG’s Python code.

My experience suggests the most practical ways of doing this involve passing the list as a JSON string or leveraging specific Airflow features that are more amenable to complex data structures, such as dag run confs or templating. I'll walk you through these approaches and provide some code examples.

First, let’s consider the JSON string method. It’s very common and generally quite reliable. When triggering your DAG (whether through the CLI, API, or via the web UI), you can serialize your list into a JSON string and pass it as part of the 'conf'. This makes it easy to handle since JSON is inherently text based and doesn't introduce special characters in the query that might cause issues with Airflow. Within your DAG, you’ll need to deserialize it back into a Python list using the `json` module. This is how that would look:

```python
import json
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def process_list(dag_run, **kwargs):
    conf_str = dag_run.conf.get('my_list')
    if conf_str:
        my_list = json.loads(conf_str)
        for item in my_list:
            print(f"Processing item: {item}")
    else:
        print("No list provided in the configuration.")


with DAG(
    dag_id="json_list_example",
    schedule=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=["example"],
) as dag:
    process_task = PythonOperator(
        task_id="process_list_task",
        python_callable=process_list,
        provide_context=True,
    )
```

In this example, `my_list` is the key within the configuration, holding your serialized list. When triggering via the CLI, the command might look something like this:

`airflow dags trigger json_list_example --conf '{"my_list": "[\\"item1\\", \\"item2\\", \\"item3\\"]"}'`

Notice the double escaping and the use of a string representation of the json. These are important. You may find that you have to single-escape these characters depending on your shell environment.

The key takeaway here is the `json.loads()` function within your PythonOperator, responsible for converting the incoming string back to a Python list you can iterate over. This is pretty straightforward and doesn't have too much extra overhead.

Now, let's look at the second approach: direct list encoding as DAG run confs when triggering. This requires a slightly more structured approach when you define your DAG. Airflow allows you to access DAG run configuration parameters through the `dag_run.conf` attribute.  While it doesn't directly handle list input, we can structure our conf slightly differently.

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def process_list_structured(list_items, **kwargs):
    for item in list_items:
        print(f"Processing item: {item}")

with DAG(
    dag_id="structured_conf_example",
    schedule=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=["example"],
) as dag:
    process_task = PythonOperator(
        task_id="process_list_task",
        python_callable=process_list_structured,
        op_kwargs={"list_items": "{{ dag_run.conf['my_list'] }}"},
        provide_context=True,
    )
```

Here, instead of directly passing the entire string as a JSON string, we're leveraging Jinja templating within `op_kwargs`. We're indicating that the `list_items` argument in the Python callable should be populated with the content stored under the key 'my_list' from the dag run confs. When triggering, we pass:

`airflow dags trigger structured_conf_example --conf '{"my_list": ["item1", "item2", "item3"]}'`

This approach avoids extra deserialization logic in your callable. It's generally more succinct and easier to read for simple list parameters, although it's important to note that the way Airflow handles templating does mean there may be limitations as to how complex the conf values you pass can be.

Finally, there's a third approach worth exploring: using Airflow's templating capabilities in combination with a parameter that is serialized into a string for processing further in the DAG.  This involves a slightly different technique, more akin to how you might handle configuration values that need to be accessible across multiple operators.

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import json

def process_list_parameter(param_string, **kwargs):
    if param_string:
        my_list = json.loads(param_string)
        for item in my_list:
            print(f"Processing item: {item}")
    else:
        print("No list provided in the configuration.")

with DAG(
    dag_id="parameter_example",
    schedule=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=["example"],
) as dag:
    process_task = PythonOperator(
        task_id="process_list_task",
        python_callable=process_list_parameter,
        op_kwargs={'param_string': '{{ params.my_list }}' } ,
    )
```

And here’s the trigger with slightly different conf:

`airflow dags trigger parameter_example --conf '{"my_list": "[\\"item1\\", \\"item2\\", \\"item3\\"]"}' --params '{"my_list": "[\\"item1\\", \\"item2\\", \\"item3\\"]"}'`

This time, we are using `params`, a special argument for the `airflow dags trigger` cli, and accessing that through the Jinja template `params.my_list` passed as an `op_kwargs`. This method mirrors the first one in many ways but instead of passing the data in the `conf`, we are passing it in the `params`, making it easier to distinguish between other parameters you might pass in the `conf`, and is more closely aligned with the intent of templated parameters. Just like the first example, we must parse the JSON in the Python callable using `json.loads()`.

From my experience, the first JSON string method provides flexibility, while the second method with structured confs is more elegant and easier to manage for single lists. The third method leverages the params and template language of Airflow in a way that is both powerful, but also requires careful handling to ensure that no unexpected issues arise due to incorrect string parsing. Each has its own advantages depending on your requirements.

For further reading, I'd strongly recommend exploring "Programming Apache Airflow" by Bas P. Harenslak and Julian B. de Ruiter. They delve deeper into templating, configurations, and how Airflow manages runtime parameters, which will give you a more solid footing. Additionally, a general understanding of Jinja templating will help to ensure that any additional dynamic parameterizations you may need can be created with the least amount of pain. Furthermore, the official Airflow documentation has a section on templating, which is worth consulting. Reading the source code on Github of the `airflow.utils.dag_parsing` and `airflow.utils.cli` modules also reveals key concepts and capabilities you may not find directly in the documentation. Finally, taking a look at some of the Airflow Improvement Proposals, AIPs, may also help you to grasp both the capabilities and potential pitfalls of some of the methods detailed above.

Remember, it’s all about choosing the method that best fits your specific scenario and maintaining a readable and maintainable DAG code base. I hope this has been helpful and you can see a path forward now to implement this in your own projects.
