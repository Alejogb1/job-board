---
title: "Why am I getting 'Error DPY-3011' with python-oracledb and Oracle DB 12.1?"
date: "2024-12-16"
id: "why-am-i-getting-error-dpy-3011-with-python-oracledb-and-oracle-db-121"
---

Okay, let's tackle this dpy-3011 error. I've definitely seen this one rear its head more times than I’d prefer, especially when dealing with older Oracle database versions like 12.1 and trying to wrangle it with `python-oracledb`. The error, specifically `DPY-3011: Connection pool timeout`, usually indicates that your application is struggling to obtain a connection from the connection pool within the configured timeout period. It isn't a problem with your oracle database itself, but rather an issue on the application side in how it is handling the connections to the database. The root causes are often quite varied, but they generally boil down to a few main culprits, which I'll explain.

From my experience, the first place to inspect is your connection pooling configuration. With `python-oracledb`, the library uses a connection pooling mechanism to optimize database access by reusing established connections rather than constantly creating and destroying them. This improves application performance but can also cause issues if not configured properly. Think of it like a limited number of doors leading into a building. If too many people try to enter at once, and the doors are limited in number, some will inevitably be stuck waiting outside. That's how a connection pool works at its simplest.

A common reason for this error is that the maximum size of the pool (`pool_max`) is too small for the demand your application is placing on the database. When your application needs a connection but the pool is full, it has to wait. If it has to wait too long (as defined by the connection pool timeout), it'll throw the dpy-3011 error. This can happen suddenly if there is a surge in users or transactions hitting your application. I once dealt with a sudden burst in overnight batch processing that overwhelmed a database client, causing a cascade of dpy-3011 errors.

Another contributing factor is the connection timeout itself. The `pool_timeout` parameter determines how long the application will wait for a connection before giving up and throwing the dpy-3011 error. Sometimes, this timeout is set too aggressively. For instance, network latency or underlying database performance issues can sometimes slow down connection requests, especially with older systems. If the timeout is too short, even minor delays can trigger this error.

Furthermore, it’s worth looking at connection leaks. If an application does not correctly close or return connections back to the pool after they're used (a classic programming error), then the pool will progressively become exhausted as connections get stuck in an ‘in-use’ state. Over time, all available connections are held and thus the application is left unable to obtain a connection from the pool.

Lastly, a long running transaction from the client side can also block other clients from obtaining a connection. If an application executes a long running query or transaction and does not return the connection to the pool, then that connection is out of the pool, reducing the available resource, and potentially causing connection exhaustion.

Let’s look at some examples to demonstrate these concepts. First, a basic example of a connection pool setup:

```python
import oracledb

pool = oracledb.create_pool(
    user="my_user",
    password="my_password",
    dsn="my_dsn",
    min=2,
    max=10,
    increment=1,
    timeout=60 # timeout in seconds
)


def execute_query(query):
  with pool.acquire() as connection:
    with connection.cursor() as cursor:
      cursor.execute(query)
      results = cursor.fetchall()
    return results

# Example Usage
try:
    data = execute_query("SELECT * FROM my_table")
    print(data)
except oracledb.DatabaseError as e:
    print(f"An error occurred: {e}")


```

In this example, `min=2` and `max=10` set the minimum and maximum connections allowed within the pool respectively. The `timeout=60` parameter dictates how long the application waits for a connection before throwing the `DPY-3011` error. If the application’s load exceeds 10 concurrent queries, it is likely to trigger the connection pool timeout.

To illustrate the problem of a too-small pool with an increased workload:

```python
import oracledb
import time
import threading

pool = oracledb.create_pool(
    user="my_user",
    password="my_password",
    dsn="my_dsn",
    min=2,
    max=5,  # Reduced max to demonstrate a small pool issue
    increment=1,
    timeout=5  # Reduced timeout to demonstrate timeout issue
)

def execute_query_with_delay(query):
  try:
    with pool.acquire() as connection:
      with connection.cursor() as cursor:
        time.sleep(2)  # Introduce a small delay to simulate workload
        cursor.execute(query)
        results = cursor.fetchall()
      return results
  except oracledb.DatabaseError as e:
    print(f"An error occurred: {e}")

def start_thread_query(query):
    threading.Thread(target=execute_query_with_delay, args=(query,)).start()


threads = []
for i in range(7):
    start_thread_query("SELECT * FROM my_table")

```

Here, we’ve intentionally reduced the maximum number of connections to 5, while attempting to execute seven concurrent threads. Because the timeout is only 5 seconds and there aren't enough available connections in the pool, we'll likely start seeing the `DPY-3011` errors appear with some of the requests.

Finally, let's look at an example of a connection leak. Notice that the connection is not returned to the pool because the `with` statement is absent in the first example of obtaining the connection, leaving an acquired connection lingering:

```python
import oracledb

pool = oracledb.create_pool(
    user="my_user",
    password="my_password",
    dsn="my_dsn",
    min=2,
    max=5,
    increment=1,
    timeout=10
)

def execute_query_leak(query):
  try:
    connection = pool.acquire()
    cursor = connection.cursor()
    cursor.execute(query)
    results = cursor.fetchall()
    # Intentional absence of connection.close() or with statement
  except oracledb.DatabaseError as e:
     print(f"An error occurred: {e}")
  return results


# Simulate multiple calls, leaking connections
for i in range(10):
    execute_query_leak("SELECT * FROM my_table")
```

In this third example, each call to `execute_query_leak` obtains a connection from the pool but never returns it. Over time this will exhaust the connection pool. Running code like this will undoubtedly lead to `DPY-3011` errors very quickly.

To mitigate these issues, I’d recommend reviewing the `python-oracledb` documentation thoroughly and specifically the connection pooling parameters. The official documentation will provide a full explanation of every available connection pool parameter. Adjust the `min`, `max`, `increment`, and `timeout` according to your application's expected workload. For deeper understanding of connection pooling, I suggest *“Patterns of Enterprise Application Architecture”* by Martin Fowler for general design patterns, or *“Java Concurrency in Practice”* by Brian Goetz for specific concurrency concerns which translate well into any programming language’s general design concerns including connection pool management. Also, always use context managers (`with` statement) to automatically release database connections back into the pool after use to avoid connection leaks. Thorough logging and monitoring can also help in tracking the number of active connections and identifying the source of these errors within the application. If none of those work you should also check the health of the Oracle database itself.
