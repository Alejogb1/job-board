---
title: "How can Airflow best send emails containing small database tables?"
date: "2024-12-23"
id: "how-can-airflow-best-send-emails-containing-small-database-tables"
---

, let’s tackle this. Generating and sending emails with small database tables within Airflow workflows is a common need, but it can be surprisingly nuanced. I've personally been down this road numerous times, from straightforward daily reports to more intricate data validation notifications. The key is to understand Airflow's strengths and limitations, and how to best leverage Python's capabilities within its context. I'll walk you through some practical approaches, explain why they work, and provide code snippets.

The fundamental approach here is to use Python operators within your Airflow DAGs to: a) query your database, b) format the retrieved data into a user-friendly table, and c) then compose and send that data via email. It's tempting to think Airflow itself will handle the formatting, but that's not its role; Airflow is the orchestrator, not the transformer. We have to explicitly handle that part within our tasks.

Let’s start with the query. Airflow supports many database operators (e.g., `PostgresOperator`, `MySqlOperator`), and you should select the one specific to your database system. For this example, let’s assume we are working with a postgres database and are using the `PostgresOperator`, and then move onto the Python logic.

**Example 1: Basic Table Formatting with Jinja**

Here's how you might approach a simple use case, where you just want the table in an HTML email and you want to keep it manageable. You'll want to retrieve your data using the appropriate Airflow database operator and then use a Python operator, that will use Jinja2 templating.

```python
from airflow import DAG
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from jinja2 import Environment, FileSystemLoader
import smtplib
from email.message import EmailMessage
import os

default_args = {
    'owner': 'airflow',
    'start_date': days_ago(1),
}

def format_table(task_instance, template_path="email_template.html"):
    # fetch results from previous task via xcom
    results = task_instance.xcom_pull(task_ids="get_data_from_db")
    if not results:
        print("no results retrieved")
        return None

    # prepare context for jinja
    context = {
      "data" : results
    }

    env = Environment(loader=FileSystemLoader(searchpath="./"))
    template = env.get_template(template_path)

    html_table = template.render(context)
    return html_table

def send_email(task_instance, smtp_server="smtp.example.com", port=587, sender_email="sender@example.com", receiver_email="receiver@example.com", password="smtp_password"):
    html_table = task_instance.xcom_pull(task_ids="format_table_task")

    if not html_table:
        print("No HTML table to send")
        return

    msg = EmailMessage()
    msg.set_content(html_table, subtype='html')

    msg['Subject'] = 'Database table from Airflow'
    msg['From'] = sender_email
    msg['To'] = receiver_email


    with smtplib.SMTP(smtp_server, port) as server:
        server.starttls()
        server.login(sender_email, password)
        server.send_message(msg)
    print(f"email sent successfully to: {receiver_email}")
    return

with DAG(
    dag_id='email_with_table',
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
) as dag:
    get_data_from_db = PostgresOperator(
        task_id='get_data_from_db',
        postgres_conn_id='your_postgres_connection', # replace with your connection
        sql="SELECT * FROM your_table LIMIT 10;", # replace with your actual query
    )

    format_table_task = PythonOperator(
      task_id="format_table_task",
      python_callable=format_table
    )

    send_email_task = PythonOperator(
        task_id="send_email_task",
        python_callable=send_email,
    )

    get_data_from_db >> format_table_task >> send_email_task
```

**And the template located at "./email_template.html":**

```html
<!DOCTYPE html>
<html>
<head>
<style>
table {
  border-collapse: collapse;
  width: 100%;
}

th, td {
  border: 1px solid black;
  padding: 8px;
  text-align: left;
}
</style>
</head>
<body>

  <table>
      <thead>
          <tr>
            {% for key in data[0].keys() %}
            <th>{{ key }}</th>
            {% endfor %}
          </tr>
        </thead>
      <tbody>
          {% for row in data %}
          <tr>
            {% for col in row.values() %}
             <td> {{ col }}</td>
            {% endfor %}
          </tr>
          {% endfor %}
      </tbody>
  </table>

</body>
</html>
```

*Explanation:* This code first executes the `PostgresOperator` which executes the SQL query, then pulls this data using xcom to the `PythonOperator` task called `format_table_task`. The `format_table_task` loads the HTML file with Jinja2 and dynamically renders it by iterating over the query result. It stores this formatted HTML string to `xcom`, and it is pulled by the `send_email_task` and used to send the email via `smtplib`.

**Example 2: Using pandas for Complex Formatting**

For more sophisticated formatting, or if you need to do data transformations *before* the table appears in the email, using `pandas` is often the most efficient approach. Here's a modified example:

```python
from airflow import DAG
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
import pandas as pd
import smtplib
from email.message import EmailMessage

default_args = {
    'owner': 'airflow',
    'start_date': days_ago(1),
}

def create_table_from_df(task_instance):
    results = task_instance.xcom_pull(task_ids="get_data_from_db")
    if not results:
      print("no results retrieved")
      return None

    # results is a list of tuples (or similar)
    df = pd.DataFrame(results)
    if df.empty:
      print("empty dataframe")
      return None

    # if you have column names in your query use this instead
    #df = pd.DataFrame.from_records(results, columns=results[0].keys())
    html_table = df.to_html(classes='table table-striped')
    return html_table

def send_email(task_instance, smtp_server="smtp.example.com", port=587, sender_email="sender@example.com", receiver_email="receiver@example.com", password="smtp_password"):
    html_table = task_instance.xcom_pull(task_ids="create_table_task")

    if not html_table:
      print("No HTML table to send")
      return
    msg = EmailMessage()
    msg.set_content(html_table, subtype='html')

    msg['Subject'] = 'Database table from Airflow'
    msg['From'] = sender_email
    msg['To'] = receiver_email


    with smtplib.SMTP(smtp_server, port) as server:
        server.starttls()
        server.login(sender_email, password)
        server.send_message(msg)
    print(f"email sent successfully to: {receiver_email}")
    return

with DAG(
    dag_id='email_with_pandas',
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
) as dag:
    get_data_from_db = PostgresOperator(
        task_id='get_data_from_db',
        postgres_conn_id='your_postgres_connection', # replace with your connection
        sql="SELECT * FROM your_table LIMIT 10;", # replace with your actual query
    )

    create_table_task = PythonOperator(
        task_id="create_table_task",
        python_callable=create_table_from_df,
    )

    send_email_task = PythonOperator(
        task_id="send_email_task",
        python_callable=send_email,
    )

    get_data_from_db >> create_table_task >> send_email_task
```

*Explanation:* Here, we pull the query results as a list of tuples (or records). Then, we use `pd.DataFrame.from_records` to create a pandas DataFrame (remember that this assumes the first result from your database contains the headers). This provides powerful manipulation tools – filtering, sorting, aggregations and then `df.to_html()` creates the table for the email. We also use `df.to_html` with css classes to style it. The rest of the email sending logic remains the same.

**Example 3: Handling Sensitive Information**

It’s essential to treat sensitive data carefully. If the tables contain sensitive data and it must be included, make sure that the emails are not stored, or available to anyone except for the intended recipients.

```python
from airflow import DAG
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
import pandas as pd
import smtplib
from email.message import EmailMessage

default_args = {
    'owner': 'airflow',
    'start_date': days_ago(1),
}

def mask_and_format_data(task_instance):
    results = task_instance.xcom_pull(task_ids="get_data_from_db")
    if not results:
      print("no results retrieved")
      return None
    df = pd.DataFrame(results)
    if df.empty:
        print("empty dataframe")
        return None

    # assuming the column name with sensitive info is named 'sensitive_data'
    df['sensitive_data'] = df['sensitive_data'].apply(lambda x: '*****' if x else ' ')
    html_table = df.to_html(classes='table table-striped')
    return html_table


def send_email(task_instance, smtp_server="smtp.example.com", port=587, sender_email="sender@example.com", receiver_email="receiver@example.com", password="smtp_password"):
    html_table = task_instance.xcom_pull(task_ids="mask_and_format_task")

    if not html_table:
      print("No HTML table to send")
      return

    msg = EmailMessage()
    msg.set_content(html_table, subtype='html')
    msg['Subject'] = 'Database table from Airflow'
    msg['From'] = sender_email
    msg['To'] = receiver_email


    with smtplib.SMTP(smtp_server, port) as server:
        server.starttls()
        server.login(sender_email, password)
        server.send_message(msg)
    print(f"email sent successfully to: {receiver_email}")
    return


with DAG(
    dag_id='email_with_masking',
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
) as dag:
    get_data_from_db = PostgresOperator(
        task_id='get_data_from_db',
        postgres_conn_id='your_postgres_connection',  # replace with your connection
        sql="SELECT id, some_other_data, sensitive_data FROM your_sensitive_table LIMIT 10;", # replace with your actual query
    )

    mask_and_format_task = PythonOperator(
        task_id="mask_and_format_task",
        python_callable=mask_and_format_data,
    )

    send_email_task = PythonOperator(
        task_id="send_email_task",
        python_callable=send_email,
    )

    get_data_from_db >> mask_and_format_task >> send_email_task
```

*Explanation:* We introduce a data transformation, where we mask out sensitive information from the query before sending it in the email. This is done using `apply` and a lambda function within pandas to mask the `sensitive_data` column.

**Key Considerations**

*   **Error Handling:** Make sure all the steps here have adequate error handling. Network errors and server errors can happen while sending emails. Your database operator could also fail to retrieve the data. It's important to add try/except blocks and use logging statements.
*   **Security:** Your SMTP credentials should be managed securely using Airflow's connection variables and secret backends.
*   **Performance:** If you're dealing with large tables (although the question specifies *small* tables), you might want to implement pagination when querying the database. The examples provided will not scale to larger datasets.
*  **Airflow Version:** Double check your Airflow version and its compatibility with the Python libraries you use.
*   **Alternative Libraries:** While `pandas` and Jinja2 are very useful for these types of tasks, there are other options. If you want to generate very complex excel files, for example, using `openpyxl` might be preferable. Or, if you have complex text formatting requirements, use `textwrap`.

**Resources**

*   For Jinja2, I'd recommend starting with the official [Jinja2 documentation](https://jinja.palletsprojects.com/). It provides comprehensive explanations and examples.
*   For pandas, "[Python for Data Analysis](https://wesmckinney.com/book/)" by Wes McKinney is the definitive resource.
*   For working with SMTP in Python, review the [official documentation for the smtplib module](https://docs.python.org/3/library/smtplib.html).
* For general Airflow documentation, refer to the official documentation: [https://airflow.apache.org/docs/](https://airflow.apache.org/docs/).

In short, while Airflow isn't directly a data processing tool, its Python operator allows us to use Python libraries to prepare, process, and transform data effectively for emails. The above examples should give you a solid starting point to build upon. Focus on good error handling, data privacy, and scalability, and you should be well-equipped to handle a variety of table formatting and email delivery tasks with your Airflow workflows.
