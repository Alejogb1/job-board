---
title: "What data adapter supports NumPy arrays and integers?"
date: "2025-01-30"
id: "what-data-adapter-supports-numpy-arrays-and-integers"
---
A common requirement in data analysis and scientific computing pipelines involves seamlessly integrating data stored in NumPy arrays with relational database systems. Traditional database connectors often struggle with NumPy's multi-dimensional structure and non-standard integer types. While many drivers offer rudimentary support, the psycopg2 library, particularly when used in conjunction with its extensions, provides robust and efficient handling of NumPy arrays and various integer types within a PostgreSQL database. My own experience migrating a large image processing pipeline from flat files to a PostgreSQL backend emphasized the necessity of a reliable adapter like psycopg2 for NumPy integration.

The core challenge stems from the mismatch between native Python data types, particularly NumPy arrays, and the data representations utilized within SQL databases. Standard database APIs typically expect scalar values and cannot directly interpret or store the complex data structures representing n-dimensional arrays. Furthermore, integer types in NumPy can vary in bit width (int8, int16, int32, int64) and signedness, creating potential type conversion issues if not handled correctly by the database adapter. Psycopg2 addresses this through several mechanisms. First, it provides a direct mapping for Python integer types to appropriate PostgreSQL types. Second, for NumPy arrays, it relies on type adaptation routines allowing for serialization into byte streams acceptable by PostgreSQL's binary data representation, which are then reconstructed into NumPy arrays during retrieval.

This type adaptation is essential for avoiding manual type casting and tedious serialization routines, making data persistence and retrieval efficient. When interacting with a database, data is typically passed as SQL query parameters, represented in Python as placeholders using `%(...)s`. Psycopg2 automatically detects the type of provided Python values and, if configured correctly, maps these types to their corresponding PostgreSQL equivalents, or in the case of NumPy arrays, transforms them into compatible binary representations. This automated mapping greatly simplifies the process of reading from and writing to a PostgreSQL database when dealing with numerical data residing in NumPy arrays.

Below are examples that highlight how `psycopg2` and its extensions handle different integer and NumPy data. Each example presents a basic scenario illustrating a common data operation:

**Example 1: Inserting and Retrieving Integer Data**

```python
import psycopg2
import numpy as np

# Connection parameters
conn_params = {
    'dbname': 'your_database',
    'user': 'your_user',
    'password': 'your_password',
    'host': 'your_host',
    'port': 'your_port'
}


try:
    conn = psycopg2.connect(**conn_params)
    cursor = conn.cursor()

    # Create a table with different integer types
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS integer_data (
            id SERIAL PRIMARY KEY,
            small_int SMALLINT,
            normal_int INTEGER,
            big_int BIGINT
        );
    """)

    conn.commit()

    # Insert Python integers
    small_int_value = 100
    normal_int_value = 10000
    big_int_value = 10000000000

    cursor.execute("INSERT INTO integer_data (small_int, normal_int, big_int) VALUES (%s, %s, %s);",
                   (small_int_value, normal_int_value, big_int_value))

    conn.commit()

    # Retrieve inserted values
    cursor.execute("SELECT small_int, normal_int, big_int FROM integer_data;")
    retrieved_values = cursor.fetchone()

    print("Retrieved Integer Values:", retrieved_values)

except psycopg2.Error as e:
    print(f"Error: {e}")

finally:
    if conn:
        cursor.close()
        conn.close()
```

In this example, we establish a connection to the database and create a table containing different integer types (SMALLINT, INTEGER, and BIGINT) present in PostgreSQL. Subsequently, we insert Python integers into the table, utilizing parameter placeholders. Psycopg2 automatically handles the type mappings between Python’s built-in integer types and the corresponding PostgreSQL types. During the SELECT operation, the stored integer data is retrieved and returned by psycopg2 as Python integers. There is no manual type handling or data conversion involved. This direct mapping contributes to simplified coding and efficient operations when dealing with numerical data in the database.

**Example 2: Handling Simple NumPy Array Insertion and Retrieval**

```python
import psycopg2
import numpy as np
from psycopg2.extensions import register_adapter, AsIs

# Connection parameters (same as before)

def adapt_numpy_array(arr):
    return AsIs(arr.tobytes())

register_adapter(np.ndarray, adapt_numpy_array)

try:
    conn = psycopg2.connect(**conn_params)
    cursor = conn.cursor()

    # Create table with bytea column to store array
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS numpy_data (
            id SERIAL PRIMARY KEY,
            data BYTEA
        );
    """)
    conn.commit()

    # Create a NumPy array
    numpy_array = np.array([1, 2, 3, 4, 5], dtype=np.int32)

    # Insert array
    cursor.execute("INSERT INTO numpy_data (data) VALUES (%s);", (numpy_array,))
    conn.commit()

    # Retrieve the array
    cursor.execute("SELECT data FROM numpy_data;")
    retrieved_data = cursor.fetchone()[0]

    # Transform byte data to NumPy array
    retrieved_array = np.frombuffer(retrieved_data, dtype=np.int32)
    print("Retrieved NumPy Array:", retrieved_array)

except psycopg2.Error as e:
    print(f"Error: {e}")
finally:
    if conn:
      cursor.close()
      conn.close()
```

This second example demonstrates the integration of NumPy arrays with PostgreSQL. A critical component here is the `adapt_numpy_array` function coupled with `register_adapter`, which instructs psycopg2 on how to interpret and convert `np.ndarray` types before insertion. The provided function serializes the NumPy array using the `tobytes()` method, storing it as a byte stream compatible with PostgreSQL’s `BYTEA` data type. On retrieval, we extract the byte stream from the database and employ `np.frombuffer()` to reconstruct the NumPy array. The explicit adapter ensures a seamless transition between the NumPy array and database storage, encapsulating the serialization logic and maintaining data fidelity. Note, in a real application, you would likely store the datatype of the array as a column in the database, allowing you to know how to de-serialize on read.

**Example 3: Inserting and Retrieving Multi-dimensional NumPy arrays**

```python
import psycopg2
import numpy as np
from psycopg2.extensions import register_adapter, AsIs

# Connection parameters (same as before)
def adapt_numpy_array(arr):
    return AsIs(arr.tobytes())

register_adapter(np.ndarray, adapt_numpy_array)

try:
    conn = psycopg2.connect(**conn_params)
    cursor = conn.cursor()

    # Create table with bytea column
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS multidim_numpy_data (
            id SERIAL PRIMARY KEY,
            data BYTEA
        );
    """)
    conn.commit()

    # Create a 2D NumPy array
    multi_dim_array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)

    # Insert the array
    cursor.execute("INSERT INTO multidim_numpy_data (data) VALUES (%s);", (multi_dim_array,))
    conn.commit()

    # Retrieve the array
    cursor.execute("SELECT data FROM multidim_numpy_data;")
    retrieved_data = cursor.fetchone()[0]

    # Transform byte data to NumPy array
    retrieved_array = np.frombuffer(retrieved_data, dtype=np.float64).reshape(2,3)
    print("Retrieved Multi-Dimensional NumPy Array:", retrieved_array)

except psycopg2.Error as e:
    print(f"Error: {e}")
finally:
   if conn:
        cursor.close()
        conn.close()
```
This final example extends the previous one to demonstrate the handling of multi-dimensional NumPy arrays. The array is again converted into a byte stream for storage via the adapter. Upon retrieval, a crucial step involves reshaping the reconstructed array using NumPy's `reshape` method. This underscores the need to preserve the dimensionality information alongside the data. If one did not store the shape and datatype, a de-serialization like the one presented is not possible. The database driver alone cannot automatically restore the array’s original shape and type. Hence, while `psycopg2` facilitates the storage of NumPy arrays, you should augment your schema to also store shape and datatype information when handling multi-dimensional arrays.

For further investigation, I would recommend reviewing the official documentation for `psycopg2` as well as publications that discuss database interaction patterns with NumPy. The documentation for PostgreSQL itself is an invaluable resource to understand the capabilities and characteristics of different data types including numeric and the binary bytea format. Research focused on Python database drivers and type adaptation strategies can also provide deeper context. Furthermore, examples and tutorials related to scientific data management with SQL systems can be beneficial in formulating more robust pipelines.
