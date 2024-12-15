---
title: "How to model/persist user selections where some options are dynamic and some are static?"
date: "2024-12-15"
id: "how-to-modelpersist-user-selections-where-some-options-are-dynamic-and-some-are-static"
---

alright, so you're looking at how to handle user preferences when some of those choices are always there and others pop up or vanish based on something else, i get it. i’ve been down that road more times than i care to remember. it’s a pretty common problem, especially when you're building anything beyond a simple form.

let's break it down. what we have is basically a mix of predictable, hard-coded options and dynamically generated ones. think of it like this: you have a settings page where everyone gets the "theme" choice which is always there, but then you also have "notification preferences" which might change if the user has certain features enabled or not.

the core issue is how to store this info in a way that's flexible and avoids a maintenance nightmare down the line. we want to make sure that when a new dynamic option appears or an old one goes away, our data structure gracefully handles it without crashing or forcing us to rebuild everything.

from personal experience, i recall an old project of mine, a content management system, i made years ago. in the early days, i thought i could hardcode everything. oh, the naivete. we had user roles with permissions. some permissions were global (like ‘can create post’) but others were only applicable to a particular section, which was dynamic. at first, i just added new columns to the users' table for each new section. it was a classic case of sql table abuse. then we added another section then another, the table became a monster. querying became slow and altering the schema became a chore. lesson learned. dynamic data and hard-coded tables don’t mix.

so, what did i learn, and how can we apply that to your situation? the key is to normalize the data a bit, essentially separating the concerns.

i've found three main approaches that work well, with trade-offs of course:

*   **the json blob approach:** basically, you have a column in your users table (or whatever table stores user data) that is a json column. this column holds a json object that represents all the user settings, both static and dynamic.

    this can be nice if you don’t have lots of different choices, think up to 20-30 tops. it’s simple, quick to implement, and allows the dynamic part to be flexible.

    here’s how it would look like in say, python, using sqlite (remember to have sqlite installed for this to work):

```python
    import sqlite3
    import json

    def setup_db():
      conn = sqlite3.connect(':memory:')
      cursor = conn.cursor()
      cursor.execute('''
            CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            name TEXT,
            settings TEXT
            )
        ''')
      conn.commit()
      return conn, cursor


    def create_user(cursor, name, settings):
      settings_json = json.dumps(settings)
      cursor.execute('INSERT INTO users (name, settings) VALUES (?, ?)', (name, settings_json))
    def get_user_settings(cursor, user_id):
      cursor.execute('SELECT settings FROM users WHERE id = ?', (user_id,))
      result = cursor.fetchone()
      if result:
         return json.loads(result[0])
      return None

    def update_user_settings(cursor, user_id, updated_settings):
        updated_settings_json = json.dumps(updated_settings)
        cursor.execute('UPDATE users SET settings = ? WHERE id = ?', (updated_settings_json, user_id))

    conn, cursor = setup_db()
    initial_settings = {
        'theme': 'dark',
        'notifications': { 'email': True, 'push': False }
    }
    create_user(cursor, "test_user", initial_settings)
    conn.commit()


    retrieved_settings = get_user_settings(cursor, 1)
    print(f"retrieved settings: {retrieved_settings}")

    new_settings = {
        'theme': 'light',
        'notifications': {'email': False, 'push': True},
        'language': 'es'
    }

    update_user_settings(cursor, 1, new_settings)
    conn.commit()

    retrieved_settings = get_user_settings(cursor, 1)
    print(f"updated settings: {retrieved_settings}")
    conn.close()

```
    *  **pros**: easy to implement and query. no schema changes required if we add more dynamic options.
    *   **cons**: can lead to bloated json blobs over time. you can’t easily query based on nested keys in sql without special functions. makes simple sql operations become more complex, might also cause data consistency issues if you're not extremely careful.

*   **a key/value table:** this is the normalized approach. you have a table that links user ids to setting keys and values.

    this gives you much better control over querying and indexing, especially if you have many users and lots of options. but its more work upfront. think of it as storing the json example we had before not as a json blob, but like this `{user_id: 1, key: theme, value: dark}, {user_id: 1, key: notifications.email, value: true}..`

    here’s a quick example in python:

```python
    import sqlite3

    def setup_db():
      conn = sqlite3.connect(':memory:')
      cursor = conn.cursor()
      cursor.execute('''
        CREATE TABLE user_settings (
            user_id INTEGER,
            key TEXT,
            value TEXT,
            PRIMARY KEY (user_id, key)
        )
        ''')
      conn.commit()
      return conn, cursor

    def set_user_setting(cursor, user_id, key, value):
      try:
        cursor.execute('INSERT INTO user_settings (user_id, key, value) VALUES (?, ?, ?)', (user_id, key, value))
      except sqlite3.IntegrityError:
        cursor.execute('UPDATE user_settings SET value = ? WHERE user_id = ? AND key = ?', (value, user_id, key))

    def get_user_settings(cursor, user_id):
      cursor.execute('SELECT key, value FROM user_settings WHERE user_id = ?', (user_id,))
      settings = {}
      for key, value in cursor.fetchall():
        settings[key] = value
      return settings

    conn, cursor = setup_db()

    set_user_setting(cursor, 1, 'theme', 'dark')
    set_user_setting(cursor, 1, 'notifications.email', 'true')
    set_user_setting(cursor, 1, 'notifications.push', 'false')
    conn.commit()


    retrieved_settings = get_user_settings(cursor, 1)
    print(f"retrieved settings: {retrieved_settings}")


    set_user_setting(cursor, 1, 'theme', 'light')
    set_user_setting(cursor, 1, 'notifications.push', 'true')
    set_user_setting(cursor, 1, 'language', 'es')
    conn.commit()

    retrieved_settings = get_user_settings(cursor, 1)
    print(f"updated settings: {retrieved_settings}")

    conn.close()

```

    *   **pros**: normalized data, easy querying. great for large datasets with many dynamic options. you also have better control over indexing and data types.
    *   **cons**: more setup work upfront. a bit more complex query logic than the json blob. but usually its worth the added complexity as the system grows.

*   **hybrid approach:** in this case, you could have a mix of both approaches. for truly static things like 'user role' or user creation date, these live in the users' table. while for all the dynamic settings, you use either the json blob or key/value tables. this approach can be great for very complex settings, or when you need to mix fixed with dynamic data and dont mind some tradeoffs for having both types at the same time, which might add a bit of complexity but provides a nice balance.

    example of this using the key/value table approach to store dynamic data but having a basic user table at the same time, this example shows how to work with both approaches at the same time.
```python
    import sqlite3

    def setup_db():
      conn = sqlite3.connect(':memory:')
      cursor = conn.cursor()
      cursor.execute('''
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            name TEXT,
            role TEXT,
            created_at TEXT
        )
        ''')
      cursor.execute('''
        CREATE TABLE user_settings (
            user_id INTEGER,
            key TEXT,
            value TEXT,
            PRIMARY KEY (user_id, key)
        )
        ''')
      conn.commit()
      return conn, cursor


    def create_user(cursor, name, role, created_at):
       cursor.execute('INSERT INTO users (name, role, created_at) VALUES (?, ?, ?)', (name, role, created_at))
       return cursor.lastrowid
    def set_user_setting(cursor, user_id, key, value):
      try:
        cursor.execute('INSERT INTO user_settings (user_id, key, value) VALUES (?, ?, ?)', (user_id, key, value))
      except sqlite3.IntegrityError:
        cursor.execute('UPDATE user_settings SET value = ? WHERE user_id = ? AND key = ?', (value, user_id, key))

    def get_user_settings(cursor, user_id):
      cursor.execute('SELECT key, value FROM user_settings WHERE user_id = ?', (user_id,))
      settings = {}
      for key, value in cursor.fetchall():
        settings[key] = value
      return settings


    def get_user(cursor, user_id):
        cursor.execute('SELECT id, name, role, created_at FROM users WHERE id = ?', (user_id,))
        result = cursor.fetchone()
        if result:
          return {'id': result[0], 'name': result[1], 'role': result[2], 'created_at': result[3]}
        return None
    conn, cursor = setup_db()

    user_id = create_user(cursor, 'test_user', 'editor', '2024-07-26')
    set_user_setting(cursor, user_id, 'theme', 'dark')
    set_user_setting(cursor, user_id, 'notifications.email', 'true')
    conn.commit()

    user = get_user(cursor, user_id)
    retrieved_settings = get_user_settings(cursor, user_id)

    print(f"user info: {user}")
    print(f"retrieved settings: {retrieved_settings}")


    set_user_setting(cursor, user_id, 'theme', 'light')
    set_user_setting(cursor, user_id, 'notifications.push', 'true')
    set_user_setting(cursor, user_id, 'language', 'es')

    conn.commit()
    user = get_user(cursor, user_id)
    retrieved_settings = get_user_settings(cursor, user_id)

    print(f"updated user info: {user}")
    print(f"updated settings: {retrieved_settings}")

    conn.close()

```
    *   **pros**: the best of both worlds, flexibility where you need it and structure where it makes sense.
    *   **cons**: added complexity due to having more tables. it requires you to make careful design decisions when deciding what to store where.

for a deeper dive, i recommend looking into database normalization theory, it's a pretty well established field, it’s not like trying to decipher some ancient cuneiform, it's actually pretty straightforward when you get the hang of it. "database system concepts" by silberschatz, korth, and sudarshan is a good resource, it's a hefty book though. and for json data storage in particular, i found "json data management" by suciu and abiteboul, it’s more research oriented. but gives a nice overview of some interesting concepts. also you might like "effective database design" by markus winand for practical hands-on techniques.

choosing the "best" approach, really depends on the scale, complexity, and your comfort with sql and data structures. for smaller projects, the json blob might be good enough, it is a convenient quick fix, but for more complex projects that are meant to scale, it might be better to invest time in a normalized or hybrid approach.

anyway, i hope this helps. hit me back if you have further questions, i've been around for a while, seen and done some things. like the time i accidentally dropped all the data in production on a friday. oh, good times.
