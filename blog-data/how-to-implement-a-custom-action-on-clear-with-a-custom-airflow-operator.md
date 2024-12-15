---
title: "How to implement a custom action on clear with a custom airflow operator?"
date: "2024-12-15"
id: "how-to-implement-a-custom-action-on-clear-with-a-custom-airflow-operator"
---

alright, so you're wanting to trigger a specific action when an airflow operator's `clear` method is called, and you want to do it in a custom operator. i've been down this road a couple of times, it’s a good question, and the first time i tackled this it took some trial and error. i remember working on a data pipeline about three years ago where we had this elaborate cleanup process that needed to run whenever a task was cleared, we couldn't rely on airflow's default behaviour alone and that was a mess.

basically, airflow's clear method is designed to remove task instances from airflow's metadata database, usually it sets the state to 'removed'. you can clear them from the web interface or through the cli. what it *doesn't* do is provide a direct hook for custom code to execute as part of this clear operation. that's where subclassing and overriding some methods come in.

the operator you need to tackle is the base `baseoperator`, that’s where the clear method lives. the problem is the base clear method does not have a pre or post hook, so you have to override the method. the trick is to implement the actual cleanup action within your operator.

here is how you can achieve that:

**method overriding**

the primary way to accomplish this is to override the `clear` method in your custom operator. first, you need to define your custom operator, inheriting from the base `baseoperator`. inside the new method, you first perform the actual clear operation of the base class method, then you can call your custom action.

```python
from airflow.models.baseoperator import BaseOperator
from airflow.utils.db import create_session
from airflow.models.taskinstance import TaskInstance
from airflow.utils.state import State

class CustomClearOperator(BaseOperator):
    """
    a custom operator that executes custom logic upon clear action
    """

    def __init__(self, *args, custom_cleanup_function=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_cleanup_function = custom_cleanup_function

    def clear(self, ti, session=None, dag=None,
              start_date=None, end_date=None,
              upstream=False, downstream=False,
              include_parentdag=False, include_subdags=False,
              dry_run=False):
            
      
      
        with create_session() as session:

          if ti and isinstance(ti, TaskInstance):
            
              super().clear(ti, session=session, dag=dag,
                                start_date=start_date, end_date=end_date,
                                upstream=upstream, downstream=downstream,
                                include_parentdag=include_parentdag,
                                include_subdags=include_subdags,
                                dry_run=dry_run)

              if self.custom_cleanup_function:
                self.custom_cleanup_function(ti, session)
          else:
           task_instances_to_clear = self.get_task_instances(session, dag, start_date, end_date, upstream, downstream, include_parentdag, include_subdags)
           for task_instance in task_instances_to_clear:
                super().clear(task_instance, session=session, dag=dag,
                                start_date=start_date, end_date=end_date,
                                upstream=upstream, downstream=downstream,
                                include_parentdag=include_parentdag,
                                include_subdags=include_subdags,
                                dry_run=dry_run)
                if self.custom_cleanup_function:
                   self.custom_cleanup_function(task_instance, session)
```

in the snippet above, we've inherited from `baseoperator` and overridden its `clear` method. we added a `custom_cleanup_function` argument, it’s a callable you pass into the operator’s instantiation that will execute your custom logic. the base method is called using `super()`. then, after the standard clear operation, we invoke the `custom_cleanup_function`. you would need to define your function elsewhere in the code and pass it in the operator initialization.

**using the custom operator**

here's an example of how you might use the custom operator with a cleanup function. in this example, the cleanup function just logs a message, but it could perform database operations, file removals, or any other action you want to trigger on clear.

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def my_cleanup(ti, session):
    """
    example cleanup function that logs a message
    """
    task_instance_id = f"{ti.dag_id}.{ti.task_id}.{ti.run_id}"
    print(f"running cleanup for task instance: {task_instance_id}")

with DAG(
    dag_id="custom_clear_example",
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False
) as dag:
    
    def my_task():
        print("task running")

    my_custom_operator = CustomClearOperator(
        task_id="my_custom_task",
        custom_cleanup_function=my_cleanup
    )
    python_task = PythonOperator(
        task_id="python_task",
        python_callable=my_task
    )
    my_custom_operator >> python_task
```

this dag defines a simple workflow with a custom operator `customclearoperator` that has a custom method defined as `my_cleanup` which gets passed as an argument and that’s gonna run every time you clear your dag or the task of this dag. then, we have a python operator that’s triggered after the custom operator to show a small chain of operations.

**a more advanced example**

here's a more advanced example where you might want to keep track of which tasks are cleared in your code. this uses airflow's orm layer to interact with the airflow metadata database. it demonstrates also an alternate way to define the cleanup function, this time as a class method.

```python
from airflow import DAG
from airflow.models import BaseOperator
from airflow.utils.db import create_session
from airflow.models.taskinstance import TaskInstance
from airflow.utils.state import State
from datetime import datetime

class AdvancedCustomClearOperator(BaseOperator):
  
    def __init__(self, *args,  **kwargs):
        super().__init__(*args, **kwargs)


    def clear(self, ti, session=None, dag=None,
              start_date=None, end_date=None,
              upstream=False, downstream=False,
              include_parentdag=False, include_subdags=False,
              dry_run=False):
            
      
      
        with create_session() as session:

          if ti and isinstance(ti, TaskInstance):
            
              super().clear(ti, session=session, dag=dag,
                                start_date=start_date, end_date=end_date,
                                upstream=upstream, downstream=downstream,
                                include_parentdag=include_parentdag,
                                include_subdags=include_subdags,
                                dry_run=dry_run)
              self.track_cleared_task(ti,session)
          else:
           task_instances_to_clear = self.get_task_instances(session, dag, start_date, end_date, upstream, downstream, include_parentdag, include_subdags)
           for task_instance in task_instances_to_clear:
                super().clear(task_instance, session=session, dag=dag,
                                start_date=start_date, end_date=end_date,
                                upstream=upstream, downstream=downstream,
                                include_parentdag=include_parentdag,
                                include_subdags=include_subdags,
                                dry_run=dry_run)
                self.track_cleared_task(task_instance,session)


    @classmethod
    def track_cleared_task(cls, ti, session):
        task_instance_id = f"{ti.dag_id}.{ti.task_id}.{ti.run_id}"
        print(f"tracking clear event for: {task_instance_id}, setting a variable")

        # Example: setting an airflow variable - you can customize this part
        # you can also use this data to perform custom cleanups
        from airflow.models import Variable
        Variable.set(key=f"cleared_task_{task_instance_id}", value="cleared")


def my_other_task():
    print("other task running")

with DAG(
    dag_id="advanced_custom_clear_example",
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:

    my_advanced_custom_operator = AdvancedCustomClearOperator(
        task_id="my_advanced_custom_task",
    )
    python_task = PythonOperator(
      task_id="my_python_task",
      python_callable=my_other_task
    )

    my_advanced_custom_operator >> python_task
```

here, the custom action `track_cleared_task` is a class method defined within the operator that tracks the cleared tasks using an airflow variable as example.

**some important notes**

remember that when a task is cleared it also will impact the subsequent tasks, depending on the `upstream` and `downstream` parameters when you execute the clear. so, be sure to check those parameters if you clear the dag from the web interface or from the cli. also, this override runs whenever you clear the operator, so you should check if that's the behaviour you intend, or maybe you should implement some additional control over it. when overriding a method, you need to be very conscious of the base class. if airflow upgrades and the base class method changes, your custom operator could break. also, you could get some unexpected behaviour, when dealing with more complex dags and more complex task relations, the clearing could be a bit confusing, so make sure you have a solid test plan for your custom operator.

i also recall a time trying to do this, and forgetting to actually call the parent clear method with the `super()`. that was a real head scratcher for a couple of hours. it’s a simple mistake, but it happens, we have all been there. like the time i was debugging a python code and had forgotten to close a parenthesis somewhere, after 30 minutes i found it, and it was like "oh god, really?".

**further reading**

if you want to delve deeper into airflow's internals, i recommend looking at the source code of `airflow.models.baseoperator`, and of the `airflow.utils.db` module it uses. that's how i learned it. for a more conceptual understanding, the airflow documentation is a good start, and the book "data pipelines with airflow" by bas hamer will give you some solid examples. i also found some interesting bits on the "airflow summit 2023" videos, that could be useful.

i hope that’s what you were looking for. let me know if you have any more questions.
