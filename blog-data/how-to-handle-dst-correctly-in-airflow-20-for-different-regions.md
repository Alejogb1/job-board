---
title: "How to handle DST correctly in Airflow 2.0+ for different regions?"
date: "2024-12-14"
id: "how-to-handle-dst-correctly-in-airflow-20-for-different-regions"
---

alright, so handling dst in airflow, especially when you're juggling different regions, yeah, that's a classic headache. been there, done that, got the t-shirt with the slightly confusing timezone abbreviations. let me break down how i’ve tackled this beast, specifically within the context of airflow 2.0 and above.

first off, the core issue boils down to how airflow stores and interprets datetime objects. by default, airflow uses utc internally. this is good, because it gives us a single source of truth. but the problem arises when you start introducing dags and tasks that need to be aware of local timezones, especially those that observe daylight saving time. if not done correctly this will completely mess up your scheduling.

for instance, i recall a project a while back where we were processing data for a client in central europe. i initially thought it would be easy, just specify the timezone in the dag and be done with it. oh boy, was i wrong. during the march switch to summer time our pipelines were off by an hour, everything was delayed, and we looked like a bunch of amateurs. that mistake taught me a lesson and i’ve never looked back. the solution is not to rely on system timezone settings, that is a recipe for disaster.

the key thing to grasp here is that airflow's scheduler does all its calculations in utc time, it’s the display and the task executions that potentially need to be localized. therefore, when specifying a dag schedule, or dealing with timedeltas, it’s important to remember that these are interpreted in relation to the utc start time and time of execution of a specific run not in the local time of a given region.

so, how do we wrangle this effectively? well, there are several strategies, and i'll lay out the ones i've found most reliable.

**1. explicitly using timezone-aware datetime objects:**

this is a big one. instead of using naive datetime objects (those without timezone info), always work with timezone-aware ones. this involves using the `pendulum` library. pendulum is now the default timezone library used in airflow since version 2.0 so chances are you have it installed already. it's a fork of the `arrow` library. it provides explicit timezone handling and makes dealing with date and times much saner.

here's how you might define a dag schedule using pendulum, ensuring a london timezone:

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
import pendulum
from datetime import timedelta

def my_python_task(**kwargs):
  london_now = pendulum.now('europe/london')
  print(f"task running in london time: {london_now}")


with DAG(
    dag_id="timezone_aware_dag_london",
    schedule=timedelta(minutes=5),
    start_date=pendulum.datetime(2023, 1, 1, tz="europe/london"),
    catchup=False,
    tags=["timezones"],
) as dag:
    
    run_task_london = PythonOperator(
        task_id='python_task_london',
        python_callable=my_python_task,
    )
```

in this example, the `start_date` is explicitly defined with london timezone. the scheduler internally stores the converted utc time, but the display in the ui and task execution times will respect the defined timezone. in the task you can also use pendulum to handle timezone conversions as you see fit.

**2. converting datetimes to local time within the dags and tasks**

this approach is about handling the time conversions within the specific task itself. this is useful when your task needs to interact with systems that expect time in a specific local timezone. for instance, when uploading logs to systems that expects the time stamp in a local timezone.

let's say you have a task where a log needs to be timestamped in the 'america/new_york' timezone:

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
import pendulum
from datetime import timedelta

def process_log(**kwargs):
    utc_time = pendulum.now('utc')
    new_york_time = utc_time.in_timezone('america/new_york')
    print(f"log created at {new_york_time} (new york)")


with DAG(
    dag_id="timezone_conversion_dag",
    schedule=timedelta(minutes=5),
    start_date=pendulum.datetime(2023, 1, 1, tz="utc"),
    catchup=False,
    tags=["timezones"],
) as dag:

    log_task = PythonOperator(
        task_id='process_log',
        python_callable=process_log,
    )
```
here, we get the current utc time and convert it to 'america/new_york' time within the task using pendulum's `.in_timezone()` method. this way your log time will be formatted according to the new york timezone.

**3. configuration approach:**

sometimes, hardcoding timezones within the dags isn't ideal. a better approach is to read timezone information from a configuration file or environment variables. this allows you to easily adjust timezones without modifying the dag code itself. it makes your code more portable and manageable.

here's a simple example of how you might read a timezone from an environment variable:

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
import pendulum
import os
from datetime import timedelta


def get_timezone_aware_time(**kwargs):
  timezone_name = os.environ.get("APP_TIMEZONE", "utc")
  current_time = pendulum.now(timezone_name)
  print(f"current time in {timezone_name} is: {current_time}")


with DAG(
    dag_id="timezone_env_var_dag",
    schedule=timedelta(minutes=5),
    start_date=pendulum.datetime(2023, 1, 1, tz="utc"),
    catchup=False,
    tags=["timezones"],
) as dag:

    process_env_var_timezone_task = PythonOperator(
        task_id='process_env_var_timezone',
        python_callable=get_timezone_aware_time,
    )
```

in this setup, the python operator is going to get the timezone configured in the os environment variable `app_timezone`. if this environment variable is not set the default timezone would be 'utc'. you could configure the environment variable in your airflow container or use a configuration file. this allows for a more flexible and modular system. i know one company that handles many clients across the globe uses this configuration approach.

**some additional points:**

*   **be consistent:** pick one of these strategies and stick to it. avoid mixing and matching, as that leads to confusion and subtle bugs. once you set up your system use it consistently through the dag and task definitions.
*   **testing:** rigorously test your dag's execution, especially when daylight saving time changes occur. use pendulum's library methods to simulate different dates and times to test the pipeline. it could save you from having some very long and frustrating debugging sessions.
*   **logging:** log datetime objects with their timezones in the tasks outputs. this makes debugging much easier. always print out the timezone in the logs so you can keep track of what’s happening.
*   **ui:** note that the airflow ui displays time in your configured webserver’s timezone by default. keep that in mind when interpreting times, it can be confusing for a beginner. there are settings for changing the timezone used by the ui as well.
*   **documentation:** if you are in a team environment document all timezone policies. it's a good idea to have a shared place for time-related info.

i would recommend reading the pendulum documentation thoroughly it will help you gain a deep understanding of how timezones and conversions work. it's also worth checking out the “timezones and scheduling” section of the airflow documentation. it contains some good practical examples on how to implement the concepts described here. finally, for a comprehensive understanding of datetime handling in general, the book “time and date” by edward m. reingold and nachum dershowitz is a great resource. it covers all the basics and the more intricate details of time and date calculations. that book is probably the most advanced and deep diving book on this topic i've ever read. and you know what? it’s a great read to be used when you want to fall asleep as well, i mean, who does not like timekeeping algorithms to fall asleep?

handling dst in airflow isn't rocket science, but it does require attention to detail. by consistently using timezone-aware datetime objects, performing necessary conversions, and leveraging configuration approaches, you can avoid those frustrating time-related bugs. good luck and may your pipelines always run on time.
