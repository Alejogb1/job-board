---
title: "How can I dynamically create Airflow 2.0 TaskFlow DAGs?"
date: "2024-12-23"
id: "how-can-i-dynamically-create-airflow-20-taskflow-dags"
---

Alright, let's talk about dynamically generating Airflow 2.0 TaskFlow DAGs. This is something I've spent a good bit of time on in a previous role, where we were moving away from manually written DAGs toward a more programmatic approach. The key challenge, as you've likely encountered, is managing the complexity inherent in orchestrating a variable number of tasks without ending up with a tangled mess of code. The move to TaskFlow in Airflow 2.0 provides an elegant solution through decorators, but dynamic creation requires a more structured methodology than what you might initially expect.

The most common approach, and frankly the most maintainable, revolves around generating your DAG definitions through a function that accepts parameters defining the shape and content of the DAG. We are not going to be generating a whole DAG from scratch on every run or trying to edit the DAG file while it's active. Instead, think of this as a recipe for constructing different DAGs based on variable inputs at definition time.

Let's break this down into steps, and I'll illustrate with code snippets.

**Core Principles:**

1.  **Function-Based Definition:** Encapsulate your DAG creation logic within a function. This function will accept configuration parameters that define the dynamic aspects, such as the number of tasks, their dependencies, and the logic each task executes.

2.  **Parameterized Tasks:** Utilize Airflow TaskFlow's decorator-based task definitions with parameter passing. This enables us to reuse task logic across a variety of contexts within the dynamically generated DAG.

3.  **Structured Configuration:** Avoid hardcoding anything within your DAG definition code. Instead, use external configuration sources, like dictionaries, JSON files, or databases, to define the parameters for your function, allowing flexibility and easy updates.

4.  **Modularity:** Keep your task functions small and focused. This practice promotes reuse and reduces the complexity of debugging any specific task. It also makes your code easier to read and follow.

Let's start with a simple example, illustrating these concepts. This example will generate a DAG with a variable number of tasks, each running the same basic python function.

```python
from airflow import DAG
from airflow.decorators import task
from datetime import datetime

def create_dynamic_dag(dag_id, num_tasks, start_date):
    @dag(dag_id=dag_id, start_date=start_date, catchup=False)
    def dynamic_dag():
        @task
        def sample_task(task_number):
             print(f"Executing task {task_number}")
             return f"Task {task_number} completed."

        tasks = [sample_task.override(task_id=f"task_{i}")(task_number=i)
                    for i in range(num_tasks)]

    return dynamic_dag()

dag1 = create_dynamic_dag(dag_id="dynamic_dag_1", num_tasks=3, start_date=datetime(2024, 1, 1))
```

In this code snippet, `create_dynamic_dag` takes parameters like `dag_id`, `num_tasks`, and `start_date` to create the DAG. The TaskFlow decorated `sample_task` is defined within the function, and we loop to create instances of this task, overriding the task ID for clarity.  This produces a dag with a varying number of sequentially executed tasks based on the provided number.

Now, let’s extend this example to a scenario where the task’s function is itself dynamic, based on configuration:

```python
from airflow import DAG
from airflow.decorators import task
from datetime import datetime
import json

def create_dynamic_dag_with_config(dag_id, config_path, start_date):
    with open(config_path, 'r') as f:
        config = json.load(f)

    @dag(dag_id=dag_id, start_date=start_date, catchup=False)
    def dynamic_dag():
        @task
        def dynamic_task(task_config):
           print(f"Executing task with config: {task_config}")
           if task_config["task_type"] == "square":
              return task_config["value"] ** 2
           elif task_config["task_type"] == "double":
              return task_config["value"] * 2
           else:
               return "unknown task type"
        tasks = [dynamic_task.override(task_id=task['task_id'])(task_config=task)
                   for task in config["tasks"]]

    return dynamic_dag()

# sample config.json file
# {
#   "tasks":[
#      {"task_id":"task_1","task_type":"square","value":5},
#      {"task_id":"task_2","task_type":"double","value":7}
#   ]
# }

dag2 = create_dynamic_dag_with_config(dag_id="dynamic_dag_2", config_path="config.json", start_date=datetime(2024, 1, 1))
```

This code uses an external `config.json` to define each task's ID, type, and value. The `dynamic_task` function is now a conditional execution, performing different logic based on the `task_type` value from the configuration. This demonstrates how task logic and task-specific parameters can be configured entirely through external sources.

A third illustrative example might incorporate dynamic dependency management based on the config, and show an example of grouping different tasks together:

```python
from airflow import DAG
from airflow.decorators import task
from datetime import datetime
from airflow.utils.trigger_rule import TriggerRule
import json

def create_dynamic_dag_with_deps(dag_id, config_path, start_date):
    with open(config_path, 'r') as f:
       config = json.load(f)

    @dag(dag_id=dag_id, start_date=start_date, catchup=False)
    def dynamic_dag():
        @task
        def task_function(task_config):
           print(f"Executing {task_config['task_id']} with: {task_config}")
           return f"{task_config['task_id']} completed"

        task_map = {}
        for group_config in config["task_groups"]:
             group_tasks = []
             for task_config in group_config["tasks"]:
                  task_instance = task_function.override(task_id=task_config["task_id"])(task_config=task_config)
                  task_map[task_config["task_id"]] = task_instance
                  group_tasks.append(task_instance)
             if group_config["dependencies"]:
                  for dep_task_id in group_config["dependencies"]:
                      if dep_task_id in task_map:
                          for t in group_tasks:
                             task_map[dep_task_id] >> t


    return dynamic_dag()

# sample config_with_deps.json file
#{
#    "task_groups":[
#        {
#            "tasks":[
#                {"task_id":"initial_task1","data":"data1"},
#                {"task_id":"initial_task2","data":"data2"}
#             ],
#             "dependencies":[]
#         },
#         {
#            "tasks":[
#                {"task_id":"dependent_task1","data":"data3"},
#                {"task_id":"dependent_task2","data":"data4"}
#             ],
#              "dependencies":["initial_task1","initial_task2"]
#         }
#    ]
#}

dag3 = create_dynamic_dag_with_deps(dag_id="dynamic_dag_3", config_path="config_with_deps.json", start_date=datetime(2024, 1, 1))

```

This final example introduces task groups, where tasks are defined inside of groups which are then linked together using a simple dependency specification. Here, `task_map` holds our tasks, and the code reads group configurations from `config_with_deps.json` which also includes a dependency declaration for groups. If dependencies are specified for a given group, the tasks in that group will be linked to those from the dependencies field, showing a more complex workflow.

**Resource Recommendations:**

For a deeper dive into the concepts, I highly recommend looking at the following resources:

1.  **The official Apache Airflow documentation:** Start with the Airflow documentation site, specifically the sections on TaskFlow and DAG definition. This resource is indispensable.
2.  **"Data Pipelines with Apache Airflow" by Bas P. Harenslak and Julian J. de Ruiter:** This book provides a comprehensive guide to best practices with Airflow, including techniques for creating dynamic DAGs. It's a useful reference.
3. **"Designing Data-Intensive Applications" by Martin Kleppmann:** While not directly about Airflow, this book offers excellent insight into building robust and scalable systems which is valuable when working on complex pipeline systems.
4. **Blog posts by the Airflow contributors and community members:** You can often find helpful articles about cutting edge techniques and best practices. These are often very practical.

Remember, dynamic DAG generation is a powerful tool, but with power comes responsibility. Keep your designs modular, well-documented, and test your solutions rigorously before moving into production. Over-engineering can lead to more problems. Start with a simple use case, work with incremental improvements, and always strive for simplicity where possible. It is always better to be clear about the structure than to generate something so complex it is unmaintainable.
