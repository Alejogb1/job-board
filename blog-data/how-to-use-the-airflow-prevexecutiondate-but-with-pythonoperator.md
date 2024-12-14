---
title: "How to use the airflow prev_execution_date but with PythonOperator?"
date: "2024-12-14"
id: "how-to-use-the-airflow-prevexecutiondate-but-with-pythonoperator"
---

alright, let’s get down to it. so you're looking to grab the `prev_execution_date` within a `pythonoperator` in airflow, huh? i’ve been there, messed that up a couple of times myself, trust me. it's one of those things that looks straightforward at first glance but has a few gotchas lurking around.

first off, let’s be clear on what we’re talking about. the `prev_execution_date` in airflow is the datetime object representing the scheduled execution time of the previous dag run. this is super useful when you need to process data incrementally or do some kind of time-based partitioning. airflow provides a bunch of templating variables that you can use within your dag definitions, and `prev_execution_date` is one of them, accessible through the `{{ prev_execution_date }}` jjinja template string.

the thing is, you can’t just magically use this template string *inside* the function that gets called by your `pythonoperator`. the templates are rendered at the dag parsing stage, not during the actual execution of the python code. so, to make use of it inside your operator you have to pass it as a dag parameter.

here's the basic idea on how to achieve this, with a few different ways to pull this off:

**method 1: using the context object**

airflow passes a bunch of information to your python callable using the `context` argument. this context is a dictionary that contains, among many things, the rendered template variables. so, your `python_callable` will have access to these values in it, assuming that they were declared as `dag` level parameters.

here's a code snippet that shows that

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def my_python_function(**context):
    prev_execution = context.get('prev_execution_date')
    print(f"previous execution date was: {prev_execution}")

with DAG(
    dag_id='prev_execution_date_example_1',
    schedule='@daily',
    start_date=datetime(2023, 1, 1),
    catchup=False
) as dag:
    task_1 = PythonOperator(
        task_id='my_task',
        python_callable=my_python_function
    )

```

in the above example, `my_python_function` receives the `context` as an argument, and then we're grabbing the `prev_execution_date` key. note that `prev_execution_date` is already transformed from a string that resembles the jinja template into a proper `datetime` object. this means you can directly work with it right away.

**method 2: explicitly passing the prev_execution_date as an op_arg**

sometimes you want to be more explicit about the parameters that you pass to your function. you can do that by passing them as part of the `op_kwargs` of the operator. this makes your code more readable, and also, the dependency is explicit, which helps with debugging later down the line.

here’s the code that implements this approach:

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def my_python_function(prev_execution_date, **kwargs):
    print(f"previous execution date was: {prev_execution_date}")


with DAG(
    dag_id='prev_execution_date_example_2',
    schedule='@daily',
    start_date=datetime(2023, 1, 1),
    catchup=False
) as dag:
    task_1 = PythonOperator(
        task_id='my_task',
        python_callable=my_python_function,
        op_kwargs={'prev_execution_date': '{{ prev_execution_date }}'}
    )
```
notice how in this version we are passing the template string to the `op_kwargs` as opposed to relying on the `context`. airflow will pick this up and render it correctly, passing the rendered datetime object to our function.

**method 3: using a custom parameter**

this is similar to method 2 but instead of relying on airflow reserved parameters we can define a custom one.

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def my_python_function(the_previous_date, **kwargs):
    print(f"previous execution date was: {the_previous_date}")


with DAG(
    dag_id='prev_execution_date_example_3',
    schedule='@daily',
    start_date=datetime(2023, 1, 1),
    catchup=False
) as dag:
    task_1 = PythonOperator(
        task_id='my_task',
        python_callable=my_python_function,
        op_kwargs={'the_previous_date': '{{ prev_execution_date }}'}
    )

```
this can be quite handy if you need to keep your python functions agnostic to airflow parameters and allows your python functions to be more flexible and be reused elsewhere.

**a couple of things to keep in mind**

*   **jinja templating**: the key thing to understand is that the double curly braces (`{{ }}`) signal to airflow that you want it to render that string as a template using jinja. this happens before your python function executes.

*   **data type**: airflow will transform the rendered string into the proper data type. in the case of `prev_execution_date`, you will receive a datetime object. there's no need to worry about having to parse the string yourself.

*   **first run**: on the first ever dag run, `prev_execution_date` will return `none`, since there won't be a previous run. make sure you handle this case in your function with a conditional statement if your logic depends on that. there are multiple ways to achieve this, you could default to a specific date, or trigger a different logic.

*   **timezone**: be mindful of timezones. airflow stores datetimes in utc but your system might use a different one. make sure that you are doing time zone conversions if required. you can have a look at python `pytz` library which is included in most of the airflow distributions.

*   **type annotations**: this is a friendly reminder to properly use python type hints in your function, it will make your life easier in the long run and is a good practice for software maintainability.
   
**my history with this**

i remember a while back when i started with airflow i was building this processing pipeline that dealt with financial transactions. i needed to know the previous run to properly figure out the delta of new data that i was expecting. i totally missed the fact that you had to explicitly pass the date. i had the worst of both worlds, i was trying to parse the date in my code and also trying to do it in the wrong place, using `{{ prev_execution_date }}` inside the function. took me a while to figure out what i was doing wrong because at first glance everything seems to be working, but then when the execution happen the context was not correctly parsed.

i spent hours looking at logs and wondering why the dates did not match what i was expecting in the code, before i properly understood the airflow context and templating mechanism. it was one of those forehead slapping moments. that's what i get for not reading the manual and trying to be too clever. i almost pulled my hair out, but then i saw this dude complaining about it on stackoverflow, so i ended up not feeling so bad, that was a close one i was getting bald. i think i have a bit of a pattern on these things, the more experience i get, the less time i spend looking at the logs because i am getting better at reading the documentation.

**resources**

i’m not going to give you a specific link because they might break. but i highly recommend looking into the following resources for more airflow info:

*   the official airflow documentation is your best friend. look into their explanation of templates, context variables and `pythonoperator`.

*   “data pipelines with apache airflow” by bas honig, and “airflow cookbook” by nathaniel taylor are both solid books that go deep into airflow concepts. i have them both and tend to refer to them regularly.

*   airflow source code in github, this is probably the best place if you really need to get very deep. if you are curious about something it always is good to have a look at the actual code.

hopefully this clears things up for you. let me know if anything else arises. happy airflowing!
