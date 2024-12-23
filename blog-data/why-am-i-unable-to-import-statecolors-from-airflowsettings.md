---
title: "Why am I unable to import `STATE_COLORS` from `airflow.settings`?"
date: "2024-12-23"
id: "why-am-i-unable-to-import-statecolors-from-airflowsettings"
---

Okay, let's unpack this. It’s not uncommon to run into these kinds of import issues when working with a complex system like Apache Airflow. I’ve seen similar scenarios multiple times, usually stemming from a misunderstanding of how Airflow’s internals are structured and intended to be used. It's crucial to understand that, while Airflow exposes several settings through the `airflow.settings` module, not everything is considered part of the public API that you, as a user, should be directly accessing for operational use. Specifically, the `STATE_COLORS` constant you're trying to import from `airflow.settings` is not part of the public, stable API. It's an internal implementation detail. This isn't to gatekeep access, but to emphasize that accessing such internal attributes might break in future Airflow versions without warning, as they aren't bound by the same backwards compatibility contracts as the public API elements.

From what I recall, a few years back when I was managing a rather large Airflow deployment across several AWS environments, we encountered a situation where a custom visualization script was pulling in several constants directly from the `airflow.settings` module. It worked initially, but when we upgraded Airflow, things went south. Quite a few of those internal attributes had changed location or were no longer available. This became a lesson learned about the dangers of relying on what I’d call "unsupported" access patterns.

The underlying issue here is that `STATE_COLORS` is indeed defined internally within Airflow, likely for rendering task state visualizations in the UI. However, its inclusion in the `airflow.settings` module is not to provide direct access to users, but rather it is made available for internal modules. Attempting to import it from `airflow.settings` is not the standard method of accessing such information, nor is it recommended.

Instead of directly importing this constant, you should consider the intended use case. If your aim is to style or interpret the state of tasks in your code, consider leveraging other available Airflow utilities for such purposes. Let's explore some alternative approaches to address what I presume to be your use case of mapping task states to specific colors.

**First Approach: Using Airflow's Task Instance Object**

When handling task states, instead of trying to access state colors directly, the task instance object is the appropriate vehicle. The task instance, available as part of Airflow's execution context, contains status information. From there, one can implement a custom color mapping function. Here’s how you might implement a custom mapping using a task instance object, within your python script:

```python
from airflow.models import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from airflow.utils.state import State

def colorize_state(task_instance):
    task_state = task_instance.current_state()
    color_mapping = {
        State.SUCCESS: "green",
        State.FAILED: "red",
        State.RUNNING: "yellow",
        State.SCHEDULED: "blue",
        State.QUEUED: "orange",
        State.UP_FOR_RETRY: "purple",
    }
    return color_mapping.get(task_state, "gray")

def my_task(**kwargs):
    ti = kwargs['ti']
    task_color = colorize_state(ti)
    print(f"Current task state: {ti.current_state()}, color: {task_color}")


with DAG(
    dag_id='task_state_example',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:
    task = PythonOperator(
        task_id='my_task_with_state',
        python_callable=my_task,
    )
```

Here, we define a `colorize_state` function that uses an explicit mapping between `State` enum values (which you _can_ import and use, since these are part of the public API) and colors. This eliminates the need to try to extract color data from internal sources, making it more resilient to Airflow version changes. The `my_task` method demonstrates how you can access the task instance (`ti`) to obtain current state information.

**Second Approach: A Static Color Mapping**

If, after investigating, you discover you only need a standard color map and don't want to deal with TaskInstance objects, or are working in an environment where they are not directly accessible, you can hardcode or define your own mapping. Again this is advantageous as it doesn't rely on undocumented attributes. Here’s a straightforward static color mapping:

```python
from airflow.models import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from airflow.utils.state import State

COLOR_MAP = {
    State.SUCCESS: "green",
    State.FAILED: "red",
    State.RUNNING: "yellow",
    State.SCHEDULED: "blue",
    State.QUEUED: "orange",
    State.UP_FOR_RETRY: "purple",
    State.SKIPPED: "gray",
}

def my_task(**kwargs):
    print(f"State colors: {COLOR_MAP}")

with DAG(
    dag_id='static_color_example',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:
    task = PythonOperator(
        task_id='my_static_color_task',
        python_callable=my_task,
    )
```

In this case, `COLOR_MAP` is defined statically and used as a simple lookup table to map state to color. This is the most basic approach and might suit scenarios where detailed access to task instances isn't readily available.

**Third Approach: Using the Public API for State Details**

For tasks where you need more advanced detail and might be working in a context where you have access to a task run, you can use the TaskInstanceState enum. While it doesn't directly map to colors, it provides the same context as the states do. It’s still using a public API, which makes it much more stable.

```python
from airflow.models import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from airflow.utils.state import State
from airflow.utils.state import TaskInstanceState

STATE_MAP = {
    State.SUCCESS: TaskInstanceState.SUCCESS,
    State.FAILED: TaskInstanceState.FAILED,
    State.RUNNING: TaskInstanceState.RUNNING,
    State.SCHEDULED: TaskInstanceState.SCHEDULED,
    State.QUEUED: TaskInstanceState.QUEUED,
    State.UP_FOR_RETRY: TaskInstanceState.UP_FOR_RETRY,
}

def my_task(**kwargs):
    ti = kwargs['ti']
    state = ti.current_state()
    state_details = STATE_MAP[state]
    print(f"Current task state: {state}, details: {state_details}")


with DAG(
    dag_id='public_state_example',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:
    task = PythonOperator(
        task_id='my_task_state_details',
        python_callable=my_task,
    )
```

In this final example, I am mapping the state enum to the `TaskInstanceState` enum. This approach makes more use of Airflow's public interface, while still giving you detailed state information that you can subsequently map to colors should you wish.

As for resources, I recommend focusing on two: “Apache Airflow: The Definitive Guide” by Kaxil Naidoo, and the official Airflow documentation. Also, keep an eye on the Apache Airflow release notes, which detail API changes with each version. These resources will deepen your understanding of the public API and highlight the importance of respecting API boundaries for maintainable code.

In summary, while `STATE_COLORS` may seem readily accessible from `airflow.settings`, it's not a supported public API, and direct imports are not advisable. The strategies I've outlined are more resilient ways to manage task states within your Airflow workflows, ensuring that you avoid unexpected breakage during future Airflow upgrades. When in doubt, always prefer the public API and avoid internal implementation details.
