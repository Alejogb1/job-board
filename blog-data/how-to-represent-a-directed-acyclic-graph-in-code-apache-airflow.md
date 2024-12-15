---
title: "How to Represent a directed acyclic graph in code (Apache airflow)?"
date: "2024-12-15"
id: "how-to-represent-a-directed-acyclic-graph-in-code-apache-airflow"
---

alright, let's talk about representing directed acyclic graphs, or dags, in code, particularly in the context of something like apache airflow. i've been around the block a few times with this kind of thing, and i've seen it done both gracefully and… not so gracefully. so, here's the lowdown on how i approach it.

first off, when we say "dag," we're talking about a set of nodes (or tasks, in airflow terms) and directional edges between them. think of it like a recipe where some steps must happen before others. in code, the core challenge is capturing these dependencies: which task needs to finish before another can begin?

there are a few ways to tackle this. at the most basic level, you might think of just storing the information in a simple way with the use of dictionaries or lists. for example using dictionaries where the keys are the nodes and values are the list of nodes that depends on. this works but as your graph gets more complex, it can quickly turn into a nightmare to manage, specially if the relations get complex or you need to trace some dependencies.

for airflow, which is what you’re after, things are thankfully more straightforward. airflow, internally, also uses a version of the following concepts, but it provides you with abstractions to make it less painful and more declarative.

i remember a particularly hairy project back in '08, where we tried to manually track a complicated workflow for an image processing pipeline. the project's graph was something like read image -> enhance image -> apply filters -> detect faces -> extract data -> save results. initially we just used nested lists. things were manageable at the start, but as requirements shifted and new tasks were needed, we ended up with a monstrous, unreadable mess. it was a classic case of trying to get too fancy and too early for the type of data we had. we spent most of our time debugging the graph representation rather than the actual image processing stuff and spent a great deal of time fixing our own mistakes. lets say we learned the hard way to plan ahead.

the takeaway from that disaster was clear: you want a structure that's easy to understand, easy to modify, and doesn't require you to remember the intricate details of how every task depends on another task.

so, instead of rolling your own, when you’re using airflow, you define dags using a python script, specifically using the `dag` and `task` classes from the airflow libraries. it’s not purely a declarative language but it’s very close to that. you essentially define task objects, and then you specify the relationships between them. it’s also worth mentioning that internally airflow will create something that is similar to the examples described before.

here's a small example to illustrate what i mean, and how i normally do it:

```python
from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

with DAG(
    dag_id='simple_dag',
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False
) as dag:
    task_a = BashOperator(task_id='task_a', bash_command='echo "running task a"')
    task_b = BashOperator(task_id='task_b', bash_command='echo "running task b"')
    task_c = BashOperator(task_id='task_c', bash_command='echo "running task c"')

    task_a >> task_b
    task_a >> task_c
```

in this example, you have three bash operator tasks: `task_a`, `task_b`, and `task_c`. the `>>` operator defines dependencies. so, in this dag, both `task_b` and `task_c` depend on `task_a` completing successfully. airflow internally takes care of scheduling and running tasks based on these dependencies. no need to keep track of lists or dictionaries by hand!

if you are using more complex operations or you need more control or have to do more work you can use a different operator called `pythonoperator`. this also creates a more complex dag with more tasks to manage, but that will come at the cost of more maintenance. here is an example of how to use it.

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def my_task_function(task_instance, value):
  ti = task_instance
  print(f"processing the value: {value}")
  ti.xcom_push(key='return_value', value=value+1)


with DAG(
    dag_id='python_dag',
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False
) as dag:
    task_a = PythonOperator(
        task_id='task_a',
        python_callable=my_task_function,
        op_kwargs={'value': 5}
    )
    task_b = PythonOperator(
        task_id='task_b',
        python_callable=my_task_function,
        op_kwargs={'value': 10}
    )
    task_c = PythonOperator(
        task_id='task_c',
        python_callable=my_task_function,
        op_kwargs={'value': 15}
    )
    task_a >> task_b
    task_a >> task_c
```

in this case we are executing python code for each of the tasks. and doing some basic passing of variables between them.

the beauty of this approach, aside from being less error-prone, is that airflow visualizes these dags in its web interface. it provides you with a graph view where you can clearly see how tasks are connected and what is currently running. this is a lifesaver when debugging complex workflows because it gives a clear representation of the structure of the dag.

now if you need to build your own solution, or create a proof of concept from scratch without using airflow there are still some tricks that can be used, you can model it as a dictionary of nodes, where each node’s value is its dependencies. for instance:

```python
dag_representation = {
    'a': [],
    'b': ['a'],
    'c': ['a'],
    'd': ['b', 'c'],
    'e': ['d']
}

def get_dependencies(node):
  return dag_representation.get(node, [])

def is_acyclic(graph):
    visited = set()
    visiting = set()

    def dfs(node):
      visited.add(node)
      visiting.add(node)
      for neighbor in get_dependencies(node):
          if neighbor not in visited:
            if dfs(neighbor):
              return True
          elif neighbor in visiting:
              return True
      visiting.remove(node)
      return False

    for node in graph:
        if node not in visited:
            if dfs(node):
                return True
    return False
#this will print "false"
print(is_acyclic(dag_representation.keys()))

dag_representation_cycle = {
    'a': ['c'],
    'b': ['a'],
    'c': ['b'],
}
#this will print "true"
print(is_acyclic(dag_representation_cycle.keys()))
```
this snippet of code simulates the basic data structure you may need to represent a graph and also adds a method to identify if a graph is cyclic which is key to identify problematic cases.

for those who want to dive deeper into the theoretical foundations of graph algorithms, i would recommend checking out "introduction to algorithms" by thomas h. cormen et al. it’s a bible in the field and covers a broad range of topics but if you need the theory, it’s a great place to go. if you’re after something focused purely on graph structures and algorithms, “graph algorithms” by shimon even is another great source. these are dense books, but they provide a very strong grasp of the underlying math and algorithms behind these techniques.

so that is my approach. you can use airflow or you can use something built from scratch, the important thing is to think about the proper data structure to manage it and the complexities that might arise from that choice. i hope it helps, and let me know if you have any other questions!
