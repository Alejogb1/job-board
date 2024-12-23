---
title: "How to run Airflow DAGs twice daily?"
date: "2024-12-23"
id: "how-to-run-airflow-dags-twice-daily"
---

Alright, let's tackle scheduling Airflow DAGs to execute twice a day. It's a common requirement, and while seemingly straightforward, there are a few nuances to get it precisely right. I've encountered this myself several times across different projects, each with slightly varying constraints, and it always comes back to mastering the scheduler's intricacies. The goal, of course, is to avoid unnecessary runs and ensure timely execution.

First, the core concept: Airflow uses cron expressions or a combination of preset schedules to define when a dag should trigger. A cron expression gives you fine-grained control, but it can be confusing if you're not used to the syntax. Alternatively, preset schedules like `@hourly` or `@daily` simplify things, but might be too rigid for this particular requirement. For running twice daily, a custom cron string or a combination of two different schedules are usually the most effective solutions. I'll explain both approaches, focusing on practical implications.

Before I jump into examples, let's set some ground rules. We should ensure that the start date of our DAG is set appropriately. In Airflow, the `start_date` is crucial and determines the first execution of the dag. Without a proper start date, things can go haywire with backfilling and unexpected execution times.

, let’s look at my first approach, using cron:

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def some_task():
    print("This is a scheduled run.")

with DAG(
    dag_id="twice_daily_cron",
    start_date=datetime(2024, 1, 1),
    schedule="0 8,20 * * *",
    catchup=False,
    tags=["example"],
) as dag:
    task_example = PythonOperator(
        task_id="run_python_task",
        python_callable=some_task,
    )
```

In this example, the `schedule` argument is set to `"0 8,20 * * *"`. This cron expression means: 'At minute 0 of hour 8 and hour 20, every day of every month, every day of the week'. So this dag will run precisely at 8:00 AM and 8:00 PM UTC (or the timezone configured in your Airflow instance). Setting `catchup=False` is generally a good idea, especially for production, because it prevents the scheduler from running old intervals if the scheduler was down or if the DAG was added later.

The critical element here is the cron string itself. It specifies the minute, hour, day of the month, month, and day of the week. A good source for learning more about cron is the official documentation for your operating system's cron implementation, as well as section 2 of the *UNIX Programming Environment* by Brian Kernighan and Rob Pike, which covers the basics of the shell and related utilities like cron. It's foundational to understanding such systems.

Now, if you prefer a more readable approach and don't want to get into the intricacies of a potentially complex cron expression, here’s a way using the timedelta for scheduling and the `schedule` variable:

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

def some_task():
    print("This is scheduled using timedelta.")

with DAG(
    dag_id="twice_daily_timedelta",
    start_date=datetime(2024, 1, 1),
    schedule=timedelta(hours=12),
    catchup=False,
    tags=["example"],
) as dag:
    task_example = PythonOperator(
        task_id="run_python_task",
        python_callable=some_task,
    )
```

In this second approach, we used `schedule=timedelta(hours=12)`. While this technically schedules the DAG to execute every 12 hours, it only accomplishes what we want if the `start_date` falls neatly into those slots. If we wanted it specifically at, say, 8 am and 8 pm, and our `start_date` wasn’t perfectly aligned with that, we might end up with runs at odd times. For this specific two-a-day requirement, it's best paired with a specified start time, but it's less flexible if you decide to alter your desired run times later. For exploring schedule options in Airflow, chapter 5 from *Data Pipelines with Apache Airflow* by Bas Harenslak and Julian Rutger is a must-read. It delves into all these scheduling options quite thoroughly.

Finally, if you want more precise control and also some flexibility, a viable third option involves using multiple `schedule` definitions, each specifying a different time. While Airflow technically accepts a list of cron expressions, it does not execute based on that list as of its current stable version. However, we can make sure the scheduler treats these schedule arguments separately by using the 'none' keyword:

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def some_task():
    print("This is running from multi-schedule definition.")


schedule_1 = "0 8 * * *"
schedule_2 = "0 20 * * *"
with DAG(
    dag_id="multiple_schedule_defs",
    start_date=datetime(2024, 1, 1),
    schedule=schedule_1,
    catchup=False,
    tags=["example"],
) as dag:
    task_example1 = PythonOperator(
        task_id="run_8_am",
        python_callable=some_task
        )
    dag.schedule = schedule_2
    task_example2 = PythonOperator(
    task_id = "run_8_pm",
    python_callable = some_task
    )
```

This example demonstrates how to run a DAG at two different times per day by reassigning the `schedule` property of the DAG object. Here, each `task_example` will be scheduled at distinct times. This method provides an elegant way to achieve this, especially if you need to separate different processes executed in those two daily runs.

In my experience, using the initial approach with a specific cron expression is often the most robust and predictable for these common daily scheduling needs. The other techniques are useful to understand and use in different situations, but cron offers precision and makes reasoning about the DAG’s execution easy. As a general rule, stick to the simplest approach that fulfills your requirements to keep your pipelines maintainable. Don’t overcomplicate things with multiple definitions unless absolutely necessary. When problems arise, it becomes simpler to trace the execution flow when the setup is kept concise and clear.

In closing, scheduling DAGs twice daily is very common, and Airflow gives you several ways to achieve it. Choosing the method that aligns with your specific needs and maintains clarity will save a lot of time in debugging. Pay close attention to start dates and time zones, as this is where most headaches happen. Reading up on crontab syntaxes in the documentation of your operating system or Unix book will help. Airflow scheduling is powerful, but mastering its core mechanisms is critical for successful pipeline management.
