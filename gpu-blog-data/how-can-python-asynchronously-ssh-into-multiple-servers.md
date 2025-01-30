---
title: "How can Python asynchronously SSH into multiple servers, execute commands, and store results in a database?"
date: "2025-01-30"
id: "how-can-python-asynchronously-ssh-into-multiple-servers"
---
Asynchronous SSH operations using Python, while seemingly complex, greatly enhance the efficiency of managing multiple remote servers. Traditional synchronous approaches, where each connection and command execution waits for the preceding one to complete, are unsuitable for large-scale deployments. Instead, harnessing the power of libraries designed for concurrency allows for non-blocking execution, drastically reducing the overall time required for such tasks. My experience working with large server farms has consistently shown that implementing asynchronous SSH is pivotal for maintaining acceptable operational speeds.

Specifically, the core challenge is managing the latency associated with network I/O and remote command processing. When initiating an SSH session and executing commands, a significant amount of time is spent waiting for responses. If performed sequentially, this results in a cumulative delay that scales linearly with the number of target servers. Asynchronous programming, in contrast, allows us to initiate a multitude of these operations simultaneously, effectively overlapping their wait times.

Here’s how one might approach this problem using Python:

Firstly, we require a library that supports asynchronous SSH operations. `asyncssh` is an excellent candidate, providing a robust framework for building asynchronous SSH clients. Its underlying asyncio framework makes it ideal for implementing concurrent interactions. Also, we will need a database connector that supports asyncio, `asyncpg` or `aiosqlite` are suitable choices, depending on the desired database system. To keep this example self-contained, let’s choose `aiosqlite`. Finally, `asyncio` is, of course, the heart of asynchronous execution.

The process involves three key stages: 1) Establishing asynchronous SSH connections to multiple servers. 2) Executing commands on these servers concurrently. 3) Storing the results, including command output and server details, into a database, again asynchronously.

Below are code examples that illustrate these steps.

**Example 1: Asynchronous SSH Connection and Command Execution**

```python
import asyncio
import asyncssh
import logging

logging.basicConfig(level=logging.INFO)


async def execute_command(host, username, password, command):
    try:
        async with asyncssh.connect(host, username=username, password=password) as conn:
            logging.info(f"Connected to {host}")
            result = await conn.run(command, check=True)
            logging.info(f"Command '{command}' executed on {host}, output: {result.stdout}")
            return host, result.stdout
    except (asyncssh.Error, OSError) as e:
        logging.error(f"Error connecting to {host}: {e}")
        return host, None


async def main():
    hosts = [
        {"host": "server1.example.com", "username": "user1", "password": "password1"},
        {"host": "server2.example.com", "username": "user2", "password": "password2"},
        {"host": "server3.example.com", "username": "user3", "password": "password3"},
        ]
    command_to_run = "uname -a"
    tasks = [
        execute_command(
            h["host"], h["username"], h["password"], command_to_run
        )
        for h in hosts
    ]
    results = await asyncio.gather(*tasks)
    for host, output in results:
        if output:
            logging.info(f"Successfully executed on {host}")
        else:
            logging.warning(f"Failed to execute on {host}")


if __name__ == "__main__":
    asyncio.run(main())
```

In this example, `execute_command` attempts to establish an SSH connection and execute a specified command. It returns both the host and the command's standard output if successful or host and None if unsuccessful. The `main` function constructs a list of asynchronous tasks and uses `asyncio.gather` to run them concurrently. The function then iterates through the results, logging success or failure. Note, real-world scenarios will typically retrieve host information and credentials from a centralized location such as a configuration file or environment variables. The `try...except` block is critical for handling network or SSH errors gracefully, preventing the entire process from crashing.

**Example 2: Database Insertion (using aiosqlite)**

```python
import asyncio
import asyncssh
import aiosqlite
import logging

logging.basicConfig(level=logging.INFO)


async def execute_command(host, username, password, command, db_conn):
    try:
        async with asyncssh.connect(host, username=username, password=password) as conn:
            result = await conn.run(command, check=True)
            await db_conn.execute(
                "INSERT INTO command_results (host, command, output) VALUES (?, ?, ?)",
                (host, command, result.stdout),
            )
            logging.info(f"Inserted result for {host} into database")
            return host, result.stdout
    except (asyncssh.Error, OSError) as e:
        logging.error(f"Error connecting to {host}: {e}")
        return host, None


async def main():
    hosts = [
        {"host": "server1.example.com", "username": "user1", "password": "password1"},
        {"host": "server2.example.com", "username": "user2", "password": "password2"},
        {"host": "server3.example.com", "username": "user3", "password": "password3"},
    ]
    command_to_run = "date"

    async with aiosqlite.connect("command_results.db") as db_conn:
        await db_conn.execute(
            """
            CREATE TABLE IF NOT EXISTS command_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                host TEXT,
                command TEXT,
                output TEXT
            )
            """
        )

        tasks = [
            execute_command(h["host"], h["username"], h["password"], command_to_run, db_conn)
            for h in hosts
        ]
        results = await asyncio.gather(*tasks)

        await db_conn.commit()

        for host, output in results:
            if output:
                logging.info(f"Successfully executed on {host}")
            else:
                logging.warning(f"Failed to execute on {host}")


if __name__ == "__main__":
    asyncio.run(main())
```

This example extends the first by adding database interaction. Before establishing SSH connections, the code creates a SQLite database table (if it doesn't exist). Within `execute_command`, after command execution, the output is stored in the database. The `db_conn` object is passed as an argument, allowing each connection to write to the same database asynchronously. Crucially, the database commit is handled after all tasks complete using `db_conn.commit()`. It is essential to manage database transactions effectively and commit data after all modifications to preserve the consistency of the database.

**Example 3: Advanced Error Handling and Logging**

```python
import asyncio
import asyncssh
import aiosqlite
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
RETRY_DELAY = 2 # seconds

async def execute_command(host, username, password, command, db_conn, max_retries=3):
    retries = 0
    while retries < max_retries:
        try:
            async with asyncssh.connect(host, username=username, password=password) as conn:
                 result = await conn.run(command, check=True)
                 await db_conn.execute("INSERT INTO command_results (host, command, output) VALUES (?, ?, ?)", (host, command, result.stdout))
                 logging.info(f"Inserted result for {host} into database")
                 return host, result.stdout
        except (asyncssh.Error, OSError) as e:
            retries += 1
            logging.error(f"Error connecting to {host} (retry {retries}/{max_retries}): {e}")
            if retries < max_retries:
                await asyncio.sleep(RETRY_DELAY)
            else:
                 logging.error(f"Max retries reached for {host}. Operation failed.")
                 return host, None

async def main():
    hosts = [
       {"host": "server1.example.com", "username": "user1", "password": "password1"},
        {"host": "server2.example.com", "username": "user2", "password": "password2"},
        {"host": "server3.example.com", "username": "user3", "password": "password3"},
    ]
    command_to_run = "whoami"

    async with aiosqlite.connect("command_results.db") as db_conn:
        await db_conn.execute("""
                CREATE TABLE IF NOT EXISTS command_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                    host TEXT,
                    command TEXT,
                    output TEXT
                )
                """
            )
        tasks = [
           execute_command(h["host"], h["username"], h["password"], command_to_run, db_conn)
           for h in hosts
        ]
        results = await asyncio.gather(*tasks)
        await db_conn.commit()

        for host, output in results:
            if output:
                logging.info(f"Successfully executed on {host}")
            else:
                logging.warning(f"Failed to execute on {host}")


if __name__ == "__main__":
    asyncio.run(main())
```
This final example introduces retry logic with a delay and improved logging. The `execute_command` function now includes a retry mechanism, which attempts to reconnect a specified number of times before giving up. This is useful for handling intermittent network issues. The logging has been enhanced to include timestamps and level information, improving its effectiveness during debugging.

For furthering one's understanding, consulting the `asyncssh` documentation is crucial for understanding its full capabilities. The `asyncio` standard library documentation offers the foundation for understanding asynchronous operations. Finally, for database interaction, `aiosqlite`’s or `asyncpg`'s resources will be valuable, depending on which database system is targeted. Thoroughly familiarizing oneself with these resources will allow for crafting increasingly sophisticated and reliable systems for asynchronous SSH operations.
