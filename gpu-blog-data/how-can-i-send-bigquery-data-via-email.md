---
title: "How can I send BigQuery data via email using Airflow?"
date: "2025-01-30"
id: "how-can-i-send-bigquery-data-via-email"
---
The core challenge in sending BigQuery data via email using Airflow lies in efficiently handling the data extraction and formatting before integrating with an email sending service.  My experience working on large-scale data pipelines highlighted the critical need for robust error handling and scalability when dealing with potentially large BigQuery result sets.  Ignoring these aspects often leads to performance bottlenecks and unreliable email delivery.

**1. Clear Explanation:**

The process involves three primary stages: data extraction from BigQuery, data transformation (often for formatting and size reduction to meet email constraints), and email delivery via an Airflow operator.  BigQuery integration typically uses the `BigQueryHook` to execute SQL queries.  The extracted data then needs conversion into a suitable format for email inclusion, such as a CSV or HTML table.  Finally, the `EmailOperator` (or a similar operator depending on your Airflow environment) facilitates sending the email.  The most efficient approach minimizes the data transferred – sending only necessary summary data rather than the entire dataset.

Consider the constraints imposed by email providers.  Many limit message size and attachment size.  Exceeding these limits results in delivery failures.  Therefore, careful data preparation and potentially summarization are crucial.  Error handling needs to be comprehensive, gracefully managing situations like empty datasets, query failures, and email sending errors.  Logging at each step is essential for debugging and monitoring.

**2. Code Examples with Commentary:**

**Example 1:  Sending a summary using a simple CSV attachment:**

```python
from airflow import DAG
from airflow.providers.google.cloud.operators.bigquery import BigQueryOperator
from airflow.operators.email import EmailOperator
from airflow.utils.dates import days_ago

with DAG(
    dag_id='bigquery_email_summary',
    start_date=days_ago(1),
    schedule_interval=None,
    catchup=False
) as dag:
    get_summary = BigQueryOperator(
        task_id='get_summary_data',
        bql="""
            SELECT COUNT(*) AS total_rows, SUM(amount) AS total_amount
            FROM `your_project.your_dataset.your_table`
        """,
        destination_dataset_table=None, # We don't need to save to BigQuery
        use_legacy_sql=False
    )

    format_summary = PythonOperator(
        task_id='format_summary',
        python_callable=lambda ti: _format_csv(ti.xcom_pull(task_ids='get_summary_data')),
    )


    send_email = EmailOperator(
        task_id='send_email',
        to=['recipient@example.com'],
        subject='BigQuery Data Summary',
        html_content='See attached CSV for summary.',
        files=['/tmp/summary.csv'] # Temporary file location
    )

    get_summary >> format_summary >> send_email

def _format_csv(data):
    import csv
    with open('/tmp/summary.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Total Rows', 'Total Amount'])
        writer.writerow([row['total_rows'], row['total_amount'] for row in data][0])  # assuming only one row
    return True

```

This example demonstrates a basic summary report. The `BigQueryOperator` retrieves aggregated data. The custom `_format_csv` function creates a CSV.  The `EmailOperator` sends the file. Error handling is minimal for brevity but would ideally include try-except blocks around data retrieval and file writing.  Note the use of `xcom_pull` to pass data between tasks.


**Example 2:  Sending a smaller subset of data as an HTML table:**

```python
# ... (Import statements as in Example 1) ...

with DAG(
    dag_id='bigquery_email_subset',
    start_date=days_ago(1),
    schedule_interval=None,
    catchup=False
) as dag:
    get_subset = BigQueryOperator(
        task_id='get_subset_data',
        bql="""
            SELECT date, product, amount
            FROM `your_project.your_dataset.your_table`
            LIMIT 100
        """,
        use_legacy_sql=False
    )

    format_html = PythonOperator(
        task_id='format_html_table',
        python_callable=lambda ti: _format_html(ti.xcom_pull(task_ids='get_subset_data'))
    )

    send_email = EmailOperator(
        task_id='send_email_html',
        to=['recipient@example.com'],
        subject='BigQuery Data Subset',
        html_content='''
        <h1>BigQuery Data</h1>
        {{ti.xcom_pull(task_ids='format_html_table')}}
        ''',
        files=[] # No file attachment needed for HTML content
    )

    get_subset >> format_html >> send_email

def _format_html(data):
    html = "<table><tr><th>Date</th><th>Product</th><th>Amount</th></tr>"
    for row in data:
        html += f"<tr><td>{row['date']}</td><td>{row['product']}</td><td>{row['amount']}</td></tr>"
    html += "</table>"
    return html
```

Here, a limited dataset is fetched.  The `_format_html` function generates an HTML table directly embedded within the email, avoiding attachment issues.  This is suitable for smaller datasets; large ones would still require summarization.


**Example 3: Implementing robust error handling:**

```python
# ... (Import statements as in Example 1, plus logging) ...
import logging

# ... (DAG definition as before, choose either query from example 1 or 2) ...

    send_email = EmailOperator(
        task_id='send_email_with_error_handling',
        to=['recipient@example.com'],
        subject='BigQuery Data Summary',
        html_content='''
        <h1>BigQuery Data</h1>
        {{ ti.xcom_pull(task_ids='format_output') }}
        ''',
        files=[],
        on_failure_callback=lambda context: _send_error_email(context),
        retry_delay=timedelta(minutes=15)
    )

    # ... (task dependencies as before) ...

def _send_error_email(context):
    logging.error(f"Email sending failed: {context}")
    # Send error email using another EmailOperator or custom function
    return

def _format_output(data): #Combines formatting and error handling
    try:
        # either _format_csv or _format_html depending on the query chosen
        return _format_html(data)  
    except Exception as e:
        logging.exception(f"Data formatting failed: {e}")
        return f"<p>Error formatting data: {e}</p>"
```

This adds error handling with `on_failure_callback` for email sending and exception handling during data formatting.  The `_send_error_email` function is a placeholder for sending an email notification about failures. This improves reliability and offers better insight into potential issues.


**3. Resource Recommendations:**

*   Airflow documentation:  Crucial for understanding operators and best practices.
*   BigQuery documentation: Covers query optimization and data export methods.
*   Python's `csv` and `html` modules:  Essential for data formatting.  Additional libraries may be necessary for more complex formatting.



By implementing these strategies and choosing the method that best suits your data volume and recipient requirements, you can create a reliable and robust solution for sending BigQuery data via email using Airflow.  Remember, scalability and error handling are paramount for production deployments.
