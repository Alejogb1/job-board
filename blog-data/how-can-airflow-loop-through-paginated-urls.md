---
title: "How can Airflow loop through paginated URLs?"
date: "2024-12-23"
id: "how-can-airflow-loop-through-paginated-urls"
---

Alright, let's talk about paginated urls and Airflow. I've dealt with this thorny problem more times than I care to recall, often late on a Friday, and it's something that requires a bit of forethought to get functioning correctly. The challenge, of course, lies in how to dynamically generate the necessary url calls based on what the API returns indicating there’s more data available. It's not something Airflow handles natively, we need to orchestrate a solution.

The fundamental issue is that Airflow tasks, by design, are generally static in their definition. They execute a fixed piece of code or command. With paginated urls, we need to iteratively alter the parameters of our url calls based on the response data, typically a "next page" indicator. We can't predict in advance how many pages there might be. Thus, we’re dealing with a process requiring a loop and this is where a blend of Airflow's task structure and specific Python logic comes into play.

My experience has usually involved external APIs, and I've found a few robust ways to manage this situation, all revolving around leveraging Airflow's programmable nature. Let's break down my preferred methodology involving a combination of custom operators and xcoms for communication between tasks. In essence, I tend to think about this in terms of generating a list of tasks dynamically and then allowing each task to process its part of the data.

First, the cornerstone is a function which will determine whether there are additional pages, and if so, return the url of the next page. This is the key, really. We'll need to parse the response from the API to see if a next-page link or indicator exists. It might be in the `Link` header, within the json payload itself, or it could be specified via a simple integer index. Let's consider an example where the API returns the next page url in a json key called ‘next’.

Here is an initial function that we will use in an Airflow task:

```python
import requests
import json
from airflow.models import Variable
def fetch_paginated_data(current_url, api_key_var, page_number, **context):

    api_key = Variable.get(api_key_var)
    headers = {"Authorization": f"Bearer {api_key}"}

    try:
      response = requests.get(current_url, headers=headers)
      response.raise_for_status()
      data = response.json()

      next_page_url = data.get('next')

      context['ti'].xcom_push(key=f'page_{page_number}_data', value=data) #Push the data retrieved from this page
      if next_page_url:
          context['ti'].xcom_push(key='next_page_url', value=next_page_url)
      else:
          context['ti'].xcom_push(key='next_page_url', value=None)
      return next_page_url
    except requests.exceptions.RequestException as e:
        print(f"Error fetching url {current_url}: {e}")
        context['ti'].xcom_push(key='next_page_url', value=None)
        return None
```

This function handles making an http call, checking if we have an error condition, and extracting the 'next' url from the returned json. It then pushes this information, as well as the data that was retrieved to an xcom. The pushing to Xcoms provides the mechanism for us to pass data between tasks in the workflow. I’ve used an airflow variable to handle the api_key, as that is good practice.

Now, let’s look at how we might use this in an Airflow dag. I’ll show this with a basic python operator implementation. For more complex scenarios, consider creating custom operators for better maintainability, but the core logic remains the same.

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from airflow.models import Variable

def create_tasks_for_pages(dag, api_key_var, start_url):
    tasks = []
    page_number = 0
    current_url = start_url
    while current_url:
        task_id = f"fetch_page_{page_number}"
        task = PythonOperator(
            task_id=task_id,
            python_callable=fetch_paginated_data,
            op_kwargs={'current_url': current_url,
                        'api_key_var': api_key_var,
                         'page_number': page_number},
            dag=dag,
        )
        tasks.append(task)
        if page_number > 0:
             tasks[-2] >> tasks[-1]
        current_url = "{{ ti.xcom_pull(task_ids='" + task_id + "', key='next_page_url') }}"
        page_number += 1
    return tasks

with DAG(
    dag_id="paginated_api_fetch",
    schedule=None,
    start_date=datetime(2023, 1, 1),
    catchup=False
) as dag:

    api_key_variable = 'my_api_key_var'
    start_url_variable = 'my_api_start_url'

    start_url = Variable.get(start_url_variable)


    fetch_tasks = create_tasks_for_pages(dag, api_key_variable, start_url)

```
In this Dag definition, I'm dynamically generating `PythonOperator` tasks using the `create_tasks_for_pages` function, continuing the loop until `next_page_url` is None. We use the jinja templating feature to dynamically pull the next page url from the xcoms.

Finally, the last part of the puzzle is to aggregate our data. After retrieving all of the pages, we likely need to do some sort of further processing. A simple example, would involve simply collecting the data from all the xcoms into a list.

```python
from airflow.operators.python import PythonOperator

def aggregate_data_from_xcoms(**context):
    all_data = []
    for task in context['dag'].tasks:
        if task.task_id.startswith('fetch_page_'):
            page_data = context['ti'].xcom_pull(task_ids=task.task_id, key=f'{task.task_id.replace("fetch_page_", "page_")}_data')
            if page_data:
                all_data.extend(page_data)
    # Do something with all_data here
    print(f"Total number of data elements processed: {len(all_data)}")
    context['ti'].xcom_push(key='all_data', value = all_data)


with DAG(
    dag_id="paginated_api_fetch",
    schedule=None,
    start_date=datetime(2023, 1, 1),
    catchup=False
) as dag:

    api_key_variable = 'my_api_key_var'
    start_url_variable = 'my_api_start_url'

    start_url = Variable.get(start_url_variable)


    fetch_tasks = create_tasks_for_pages(dag, api_key_variable, start_url)
    aggregate_data = PythonOperator(
          task_id='aggregate_data',
          python_callable=aggregate_data_from_xcoms,
          dag=dag
    )

    fetch_tasks >> aggregate_data
```

Here, after our fetch tasks, we loop through the previous tasks using the 'dag' object and the 'tasks' property to access the specific page data stored in the xcoms, and then we aggregate the data into a single xcom value.

This workflow is quite flexible. The `fetch_paginated_data` function can handle different types of next page indicators with a bit of modification. Similarly, the aggregation can be adapted to the specific needs of the application. You would not include the above Dag code in the same file as the function, but I’ve done it here for clarity.

For further learning, I highly recommend diving into the Airflow documentation (especially the section on XComs and operators), and familiarize yourself with best practices for dynamic task generation. You can find these details on the official Apache Airflow website. Also, for understanding the nuances of pagination within APIs, I would point you towards reading the specific API documentation and researching best practices in API design in general. That should help you recognize the various approaches they might use. Finally, working through more complex examples involving real-world APIs with varying types of pagination is incredibly beneficial for hands-on understanding. I hope this helps, and if you run into any more issues, feel free to ask!
