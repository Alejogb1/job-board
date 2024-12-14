---
title: "How to Convert an xcom_pull list in the "in" clause of an SQL query?"
date: "2024-12-14"
id: "how-to-convert-an-xcompull-list-in-the-in-clause-of-an-sql-query"
---

alright, so you're looking at pulling a list from xcom in airflow and using that list to filter results in an sql query, sounds familiar. been there, done that, got the t-shirt, and probably a few grey hairs. let me break down how i usually tackle this, and some pitfalls i've stumbled into along the way.

first off, xcom, for those not deeply ingrained in airflow, is essentially a message passing system within your dags. it lets tasks communicate data, and in our case, we're talking about a list of something you need in your sql. a common scenario for me was pulling a list of customer ids after some initial data processing and then using that to query specific customer details in a subsequent task. initially, i made some mistakes assuming that xcom_pull just hands me a perfectly formatted sql-ready string, oh boy was i wrong.

the core issue you're facing is transforming that python list into a format that the sql 'in' clause understands. the 'in' clause expects a comma-separated list of values, enclosed in parentheses, which isn't exactly the default output of `xcom_pull`.

let's start with a basic scenario, assuming the xcom value is a list of integers. here's how i'd typically pull that list, process it, and use it in a query:

```python
from airflow.decorators import dag, task
from airflow.providers.postgres.hooks.postgres import PostgresHook
from datetime import datetime

@dag(start_date=datetime(2023, 1, 1), catchup=False, schedule=None)
def xcom_to_sql_example():

    @task()
    def generate_ids():
        #imagine this is a more complex task retrieving values, for this example a list
        return [1, 2, 3, 4, 5]

    @task()
    def query_database(ids):
       
        postgres_hook = PostgresHook(postgres_conn_id="your_postgres_connection")
        
        # transform list into a string suitable for the sql in clause
        sql_values = ", ".join(str(id) for id in ids)
        sql_query = f"SELECT * FROM your_table WHERE id IN ({sql_values})"
        
        records = postgres_hook.get_records(sql_query)
        print(f"query results are: {records}")

    id_list = generate_ids()
    query_database(id_list)
    
xcom_sql_dag = xcom_to_sql_example()

```
that's the basic approach when dealing with integers. the key part is `", ".join(str(id) for id in ids)`. this converts each integer to a string and joins them together with a comma, creating our sql-ready list.

but what if you're working with strings, for instance, customer names or product codes? strings require special handling because you need to wrap each value in single quotes in the sql query. if not, you'll get sql errors, i had that problem quite often, it's like the sql server is trying to say 'i know you are not sending me the right stuff'.

here’s how i would adapt that for string values:

```python
from airflow.decorators import dag, task
from airflow.providers.postgres.hooks.postgres import PostgresHook
from datetime import datetime

@dag(start_date=datetime(2023, 1, 1), catchup=False, schedule=None)
def xcom_to_sql_example_strings():

    @task()
    def generate_names():
        # imagine this is a more complex task retrieving values
        return ['alice', 'bob', 'charlie']

    @task()
    def query_database_strings(names):
       
        postgres_hook = PostgresHook(postgres_conn_id="your_postgres_connection")

        # transform strings to sql format
        sql_values = ", ".join(f"'{name}'" for name in names)
        sql_query = f"SELECT * FROM your_table WHERE name IN ({sql_values})"

        records = postgres_hook.get_records(sql_query)
        print(f"query results are: {records}")

    name_list = generate_names()
    query_database_strings(name_list)
    
xcom_sql_string_dag = xcom_to_sql_example_strings()
```
the only change is in the line `sql_values = ", ".join(f"'{name}'" for name in names)`. the `f"'{name}'"` part encloses each name in single quotes, making the sql happy, i had issues on the past with those kind of details.

one thing to absolutely watch out for is sql injection. if your xcom values come from an untrusted source, directly injecting them into your sql queries like this is a huge security risk, please be aware. never assume that data is clean, always sanitize your inputs. if you deal with user-submitted values, consider using parameterized queries or prepared statements.

i've found using parameterized queries to be much safer specially when you have unpredictable inputs. here is an approach using psycopg2 that i use for a production environment instead of using the postgres hook get_records (just to be clear, if you don't need this you might be better sticking with the hook for simplicity)

```python
from airflow.decorators import dag, task
from airflow.providers.postgres.hooks.postgres import PostgresHook
import psycopg2
from datetime import datetime

@dag(start_date=datetime(2023, 1, 1), catchup=False, schedule=None)
def xcom_to_sql_parameterized():

    @task()
    def generate_items():
        #imagine this is a more complex task retrieving values, for this example a list
        return ["item_1", "item_2", "item_3"]

    @task()
    def query_database_parameterized(items):
        
        postgres_hook = PostgresHook(postgres_conn_id="your_postgres_connection")
        conn = postgres_hook.get_conn()
        cur = conn.cursor()
        
        placeholders = ','.join('%s' for _ in items)
        sql_query = f"SELECT * FROM your_table WHERE item_name IN ({placeholders})"
        
        cur.execute(sql_query, items)
        records = cur.fetchall()
        print(f"query results are: {records}")
        cur.close()
        conn.close()

    items_list = generate_items()
    query_database_parameterized(items_list)
    
xcom_sql_parameterized_dag = xcom_to_sql_parameterized()
```
in this case, i generate placeholders '%s' for each item in the list, then execute the query with the items as parameters using the cursor execute method. this prevents any malicious sql from being injected. this is the way to go specially if the values come from a user, or an external application where you don't fully have the control.

now, for more advanced scenarios, you might find your xcom values are not simple lists, maybe you have nested json structures, or complex datatypes. in that case, you'll need to parse them appropriately before passing it to the sql query. the specific methods you need will depend completely on your data structures. usually for complex json structure the library json in python works like a charm to extract the desired values.

for learning more about sql and working with databases, i'd recommend the book "sql for dummies" (not the official name obviously, but you probably get the idea) this is a good starting point. and the official documentation for your specific database system, such as postgresql documentation, is invaluable if you are using postgresql like in my examples. also consider taking a look at the pep 249 (python database api specification) for a good standard on how to programmatically interact with databases, there is a lot of good information there. and as a bonus "the pragmatic programmer" provides good principles to have solid foundations on writing software in general.

one funny thing about sql is that it's like trying to teach a computer to speak human, but only with very strict grammar and no room for creativity. anyways, always keep an eye on the types and make sure your sql is correctly formatted to avoid headaches.
hope that helps you out with your issue, remember always sanitize the data and prefer parameterized queries if possible.
