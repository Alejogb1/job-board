---
title: "How to Update the records along with duplicates using upsert_all?"
date: "2024-12-15"
id: "how-to-update-the-records-along-with-duplicates-using-upsertall"
---

so, you're dealing with updating records, including handling duplicates, when using `upsert_all`, huh? i've been there. it's that sticky situation where you’re staring at your database and thinking, "there *has* to be a less clunky way."  i remember one project back in '08 - we were building a content management system for a now-defunct dot-com. i had a table full of product descriptions that needed constant updating, and, of course, the data we were getting was riddled with duplicates. we originally tried doing individual updates, but that crashed and burned the moment the dataset hit something like 10,000 entries. those were dark times, the server was weeping and my hair was thinning (more than it is now). anyway, using `upsert_all` became the obvious solution to get us out of that mess. let's walk through the best approach on how to make that happen smoothly.

the key with `upsert_all` is its ability to either insert new records or update existing ones based on a conflict clause. now, when you're talking duplicates, that conflict clause becomes super important. essentially, you need to specify which columns uniquely identify a record. if you have a composite unique constraint (multiple columns that together define uniqueness), that's what the upsert will check for, in case it exists update them instead.

first off, let’s start with a simple case and see how to handle this: say you have a table called `products` with columns `id`, `name`, and `description`. also lets say the `name` is what you want to consider to be unique to update the description column. the following code would be the way to do this:

```python
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import insert

# setup your connection
engine = create_engine("postgresql://user:password@host:port/database")
metadata = MetaData()

products_table = Table(
    "products",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("name", String, unique=True),
    Column("description", String),
)

metadata.create_all(engine)

session = sessionmaker(bind=engine)()

# our data, notice the duplicates based on name
data_to_upsert = [
    {"name": "laptop", "description": "a portable computer"},
    {"name": "mouse", "description": "a pointing device"},
    {"name": "keyboard", "description": "an input device"},
    {"name": "laptop", "description": "updated laptop description"},
    {"name": "monitor", "description": "a display device"},
    {"name": "keyboard", "description": "updated keyboard description"},
]

# create an insert statement on conflict do update
insert_stmt = insert(products_table).values(data_to_upsert)
upsert_stmt = insert_stmt.on_conflict_do_update(
    index_elements=[products_table.c.name],
    set_={
       "description": insert_stmt.excluded.description
    }
)

session.execute(upsert_stmt)
session.commit()

session.close()
```

what this code does is setup a connection, defines a table using sqlalchemy and then generates an insert statement and specifies the conflict action to update on conflict of the `name` column. what is important to notice is the `excluded` attribute from the insert statement. this allows you to refer to the columns of the insert clause inside the update clause for the update part.

now let's say you have a more complex case where you have multiple fields that identify uniqueness. imagine you have a table for `users` with columns `email`, `username`, and `last_login`. you decide that the user's record should be uniquely identified by `email` and `username`. the `last_login` column is the value you want to update if a user that matches that record, already exists. so it means when there is a duplicate on `email` and `username` update the `last_login` column of that record.

```python
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import insert
from datetime import datetime

engine = create_engine("postgresql://user:password@host:port/database")
metadata = MetaData()

users_table = Table(
    "users",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("email", String),
    Column("username", String),
    Column("last_login", DateTime),
    UniqueConstraint("email", "username", name="email_username_unique"),
)

metadata.create_all(engine)

session = sessionmaker(bind=engine)()

data_to_upsert = [
    {"email": "john.doe@example.com", "username": "johndoe", "last_login": datetime.now()},
    {"email": "jane.doe@example.com", "username": "janedoe", "last_login": datetime.now()},
    {"email": "john.doe@example.com", "username": "johndoe", "last_login": datetime.now()},
    {"email": "peter.pan@example.com", "username": "peters", "last_login": datetime.now()},
    {"email": "jane.doe@example.com", "username": "janedoe", "last_login": datetime.now()},

]


insert_stmt = insert(users_table).values(data_to_upsert)
upsert_stmt = insert_stmt.on_conflict_do_update(
    index_elements=[users_table.c.email, users_table.c.username],
    set_={
        "last_login": insert_stmt.excluded.last_login
    }
)

session.execute(upsert_stmt)
session.commit()
session.close()
```

the important thing to notice here is that we specify more than one column on the `index_elements` argument. this represents the composite unique key of our table and the conflict resolution is done based on those parameters. in this case, it would insert the new entries and update the record last\_login column if a record exists with that email/username combination. that means that if there are duplicated rows based on email and username, the `last_login` will be updated with the new value for that row. it's a really elegant way to deal with duplicates, and its way better than looping. by the way, i once spent an entire weekend debugging a similar issue, and it turned out that the database server's clock was out of sync, so the records were updating but with the wrong times, man, what a facepalm moment.

and what happens if you dont want to update any of the record fields if it exists? for instance, you could be importing records in batch, and a primary key could not be specified at insert time, but be generated on the database. this can happen frequently if you need to import records from other sources, or you can be doing a batch insert and your primary key is `autoincrement`. in such case, you would like to insert records if they do not exist. let me show you a simplified version of the previous example that covers this case:

```python
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, DateTime, UniqueConstraint
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import insert
from datetime import datetime

engine = create_engine("postgresql://user:password@host:port/database")
metadata = MetaData()

users_table = Table(
    "users",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("email", String),
    Column("username", String),
    Column("last_login", DateTime),
    UniqueConstraint("email", "username", name="email_username_unique"),
)

metadata.create_all(engine)

session = sessionmaker(bind=engine)()

data_to_upsert = [
    {"email": "john.doe@example.com", "username": "johndoe", "last_login": datetime.now()},
    {"email": "jane.doe@example.com", "username": "janedoe", "last_login": datetime.now()},
    {"email": "john.doe@example.com", "username": "johndoe", "last_login": datetime.now()},
    {"email": "peter.pan@example.com", "username": "peters", "last_login": datetime.now()},
    {"email": "jane.doe@example.com", "username": "janedoe", "last_login": datetime.now()},

]


insert_stmt = insert(users_table).values(data_to_upsert)
upsert_stmt = insert_stmt.on_conflict_do_nothing(
    index_elements=[users_table.c.email, users_table.c.username]
)

session.execute(upsert_stmt)
session.commit()
session.close()

```

here the `on_conflict_do_nothing` method takes the index elements, and if it finds a record that matches them, it skips the upsert action for that row. this is very useful when doing idempotent batch inserts.

when working with this feature, it is super important that you test your code before deploying it in production. as in, write unit tests, integration tests, and end to end tests if needed, specially on conflict resolution parts of your implementation.

the key things to remember is that `upsert_all` is a powerhouse for handling database updates with duplicates if you understand the concept of a conflict clause properly. by specifying unique columns, you can insert or update records accurately, and it’s way more efficient than doing individual updates or selects followed by updates. and for more detailed information, i recommend checking out “database internals: a deep dive into how distributed data systems work" by alex petrov. this book provides not only a great understanding on the underlying database structures but it also covers different concepts about transactional systems that are also helpful, or if you need more concrete examples check out the "postgresql documentation" there is a section dedicated to this topic that is well worth to study in detail.
