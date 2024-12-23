---
title: "Should a CQRS application's query side interact directly with the database?"
date: "2024-12-23"
id: "should-a-cqrs-applications-query-side-interact-directly-with-the-database"
---

Alright, let’s unpack this. I've seen this debated countless times, and the answer isn't as cut-and-dry as some might think. Based on my experience building and maintaining several systems with varying levels of CQRS implementation, I can tell you there's a nuanced approach needed here, not a hard ‘yes’ or ‘no.’ The short answer, is ‘it depends’. Let’s dive into the details, focusing on practical concerns and trade-offs.

First, let's re-establish a fundamental principle: the core concept of Command Query Responsibility Segregation (CQRS) is to separate operations that modify data (commands) from those that read data (queries). The idea is to optimize each side independently. Now, the question about the query side interacting directly with the database raises a critical architectural decision point. A naive implementation might suggest that the query side reads directly from the write database, the very same database being modified by the command side. This *can* work, but it often leads to performance bottlenecks and scaling issues, especially when dealing with high read loads. I've witnessed this firsthand in an e-commerce platform where read requests dramatically outpaced write requests, causing the single database to crumble under the pressure.

On the other hand, the argument for having a dedicated read database, sometimes referred to as the "read model" or "query model," is that it allows for tailored optimization. We’re not talking about a simple replicated version of the command-side database here; we mean a structure optimized specifically for queries. This might involve using a different data storage technology, such as a denormalized database optimized for fast reads, or even an in-memory cache. Think about scenarios where your reads are always based on specific filters or aggregates – a standard relational database might struggle, whereas a document store or a specialized indexing system would excel. I recall a system I worked on involving geospatial data, where a traditional RDBMS proved incredibly slow; migrating the read side to a dedicated geospatial database resulted in a massive performance boost.

The problem with a direct-to-write-database read approach is that it can also introduce transactional coupling. Consider a scenario where complex write operations occur, requiring locking or expensive computations. These can directly impact the query performance if both sides are hitting the same database instance. The read side should ideally be decoupled from these concerns as much as possible, ensuring that query performance is consistent, predictable, and less affected by write side activities. If the write database is a standard RDBMS, for instance, the need for database indexes for both read and write operations creates a challenge, where optimisations for one might hinder the performance of the other.

Now, let me illustrate these points with some practical code examples. I'll use Python here for readability, but the concepts apply regardless of the chosen technology.

**Example 1: Direct Querying from the Command Database (Anti-Pattern in many situations)**

```python
import sqlite3

# Assume this is our database connection, being used for both writes and reads
conn = sqlite3.connect('mydatabase.db')
cursor = conn.cursor()

def create_user(user_id, username, email):
    cursor.execute("INSERT INTO users (user_id, username, email) VALUES (?, ?, ?)", (user_id, username, email))
    conn.commit()

def get_user(user_id):
  cursor.execute("SELECT username, email FROM users WHERE user_id = ?", (user_id,))
  user_data = cursor.fetchone()
  if user_data:
    return {"username": user_data[0], "email": user_data[1]}
  return None

#Example usage:
create_user(1, "john_doe", "john.doe@example.com")
user_info = get_user(1)
print(user_info)
```

This illustrates a naive CQRS setup where both the create operation (command) and the retrieval operation (query) hit the same database. This might seem straightforward, but as you can see, `get_user()` is coupled to the write database schema. Imagine if we needed to add a "last_login" field that isn't immediately critical for the command side and has different indexing needs. We'd need to modify both the command and query functions, breaking the separation of concerns.

**Example 2: Utilizing a Separate Read Database (Improved Approach)**

```python
import sqlite3

# Database for writing
write_conn = sqlite3.connect('write_db.db')
write_cursor = write_conn.cursor()

# Database for reading (Optimised for reads)
read_conn = sqlite3.connect('read_db.db')
read_cursor = read_conn.cursor()

def create_user(user_id, username, email):
    write_cursor.execute("INSERT INTO users (user_id, username, email) VALUES (?, ?, ?)", (user_id, username, email))
    write_conn.commit()

def update_read_model(user_id, username, email):
    read_cursor.execute("INSERT INTO users_read (user_id, username, email) VALUES (?, ?, ?)", (user_id, username, email))
    read_conn.commit()

def get_user(user_id):
    read_cursor.execute("SELECT username, email FROM users_read WHERE user_id = ?", (user_id,))
    user_data = read_cursor.fetchone()
    if user_data:
        return {"username": user_data[0], "email": user_data[1]}
    return None

#Example Usage
create_user(1, "john_doe", "john.doe@example.com")
# Note: In real scenario, a message queue would be used to update the read database
update_read_model(1, "john_doe", "john.doe@example.com")
user_info = get_user(1)
print(user_info)
```

Here, we have two separate databases: `write_db.db` for commands, and `read_db.db` for queries. The `get_user` function now only interacts with the read database. Crucially, we've separated the databases; this lets us optimize them independently. The read database table `users_read` could use different indexes or even different storage if needed, without affecting the write side. There's still the matter of syncing the read database (simulated by `update_read_model`), which in a real application would likely involve an event bus or queue, but the separation is clearer now.

**Example 3: A More Realistic Read Model Setup (Using a Document Store)**

```python
from pymongo import MongoClient

# Writing to SQL Database
write_conn = sqlite3.connect('write_db.db')
write_cursor = write_conn.cursor()

# Reading from MongoDB as a document store
client = MongoClient('localhost', 27017)
db = client.mydb

def create_user(user_id, username, email):
    write_cursor.execute("INSERT INTO users (user_id, username, email) VALUES (?, ?, ?)", (user_id, username, email))
    write_conn.commit()

def update_read_model(user_id, username, email):
    user_doc = {
      "user_id": user_id,
      "username": username,
      "email": email
    }
    db.users_read.insert_one(user_doc)

def get_user(user_id):
    user_doc = db.users_read.find_one({"user_id": user_id})
    if user_doc:
        return {"username": user_doc["username"], "email": user_doc["email"]}
    return None

#Example Usage
create_user(1, "john_doe", "john.doe@example.com")
update_read_model(1, "john_doe", "john.doe@example.com")
user_info = get_user(1)
print(user_info)
```

This example shows a slightly more advanced setup. The write side still uses the SQL database, but now the read side uses MongoDB, a document store. The `get_user` method can take advantage of the document-oriented nature of MongoDB, making it very fast to fetch a user, especially if you were doing searches based on other criteria as well. This is a more realistic example of what a CQRS system might look like in a modern architecture, leveraging different datastores for specific needs. The read database and its data model are now completely separate from the write database.

In summary, while direct interaction of the query side with the command database is *possible*, it's often not the optimal choice for scalable, maintainable systems. I've found that having a dedicated read model and read database is generally the better approach.

If you’re interested in diving deeper into this area, I highly recommend checking out *Domain-Driven Design: Tackling Complexity in the Heart of Software* by Eric Evans for the theoretical underpinnings of domain modeling. *Implementing Domain-Driven Design* by Vaughn Vernon provides a more practical approach. For understanding the underlying distributed systems concepts, *Designing Data-Intensive Applications* by Martin Kleppmann is invaluable. Also, papers on materialised views and event sourcing can offer practical insights on keeping the read database up-to-date. These should keep you going for a while, and give you the depth to navigate these challenges effectively.
