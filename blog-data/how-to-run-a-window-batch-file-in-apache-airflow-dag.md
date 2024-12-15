---
title: "How to run a window batch file in apache airflow DAG?"
date: "2024-12-15"
id: "how-to-run-a-window-batch-file-in-apache-airflow-dag"
---

alright, so you're looking to trigger a windows batch file from within an apache airflow dag. i get it, been there, done that, bought the t-shirt… several, actually. it's one of those things that seems straightforward at first glance but then you hit a wall when you realize airflow is primarily designed for *nix environments.

i've had my fair share of frustrations with this, particularly early in my career when i was working on a project where we had a bizarre mix of legacy windows systems and newer linux-based servers. the need to run a batch file in a data pipeline was an odd requirement, almost a weird outlier but, it was non-negotiable. back then, airflow wasn't as mature as it is today and resources were sparse (pre-stackoverflow day).

first off, let's be clear: airflow itself doesn't directly execute windows batch files in a native sense. it relies on the underlying operating system where its workers run. so, if your airflow workers are on linux, you can't just magically run a `.bat` file there. the solution involves a small workaround and some careful configuration.

the basic idea is to use the `bashoperator` and have that bash command execute the batch file through a remote windows system. this means you need a windows machine accessible on your network, where you have permission to execute files. the communication between the airflow worker and the remote windows system is handled using `winrs` (windows remote shell) or something similar.

here's how i typically approach it. the core principle is to use a remote shell to execute the command on the windows machine which has to be already configured to accept such requests. first, we need to make sure all the necessary dependencies exist in the target windows machine, that is, to execute a command by using the `winrs` tool. this needs to be configured ahead of time with some windows commands and user credentials that will allow the remote execution.

here's a simple python airflow dag snippet showing this:

```python
from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

with DAG(
    dag_id='execute_windows_batch_file',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False
) as dag:
    execute_batch = BashOperator(
        task_id='execute_remote_batch',
        bash_command="""
            winrs -r:windows_server_ip -u:username -p:password  "cmd /c c:\\path\\to\\your\\batchfile.bat arg1 arg2"
        """
    )
```

in this example:

*   `windows_server_ip` is the ip address or hostname of the windows server where the batch file resides.
*   `username` and `password` are credentials of a user account on the windows server that has permissions to execute the batch file.
*   `c:\\path\\to\\your\\batchfile.bat` is the full path to your batch file on the windows server.
*   `arg1 arg2` are any arguments you might need to pass to your batch file, space separated, which are then passed to the file.

it's critical to remember that the `winrs` tool must be installed and configured correctly on both the airflow worker machine and the target windows server, or the whole process will fail silently and you'll spend hours wondering why. if i could go back, i'd make sure i had this part down pat before attempting any automation.

another thing, the command `cmd /c` executes the batch file and then exits. if you need the command window to remain open for debugging purposes, you can use `cmd /k` instead. for production environments, it's recommended to use `/c` to release resources quickly.

now, let's say your batch file generates output which you need to ingest into your airflow pipeline. you might be tempted to pipe the output of `winrs` directly, but that's not usually a good approach. it's best to have the batch file save its output to a file accessible by the airflow workers.

here's a slightly more involved example, with output redirection to a file:

```python
from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

with DAG(
    dag_id='execute_windows_batch_file_with_output',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False
) as dag:
    execute_batch = BashOperator(
        task_id='execute_remote_batch',
        bash_command="""
            winrs -r:windows_server_ip -u:username -p:password  "cmd /c c:\\path\\to\\your\\batchfile.bat > c:\\path\\to\\output.txt"
        """
    )
    process_output = BashOperator(
        task_id='process_batch_output',
        bash_command="""
            # assuming a mount point for the windows share is present in airflow worker machine
            cat /mnt/windows_share/path/to/output.txt
            # here do something with the output file.
        """
    )
    execute_batch >> process_output
```

in this second example, the batch file output is redirected into a file called `output.txt`, the batch file should be setup to do that, this means that `c:\\path\\to\\output.txt` should be accessible to write into. after the batch file runs, the airflow worker reads the output file from the shared directory, and then can use the contents of the file using the cat command. we are assuming that the output file was configured to be in a shared mount between the worker and windows server and its accessible through a location `/mnt/windows_share/`. this way the output is accessible in the airflow worker itself.

keep in mind, dealing with windows paths in bash commands can be a headache. backslashes need to be escaped or you can use forward slashes, since windows itself understands forward slashes just fine. also, proper quoting is crucial to avoid unexpected command behavior.

now, if you are working with very sensitive data like passwords, it is a really, *really* bad idea to put the credentials directly into the dag file. i mean, seriously, it's a horrible idea. i learned that the hard way once, during a security audit where my code was not only scrutinised to the extreme, but also publicly ridiculed, not my best day at the office.

instead, you should use airflow's secrets backend to handle these kinds of sensitive parameters. you can store the username and password in a secret manager (like hashicorp vault or aws secrets manager) and then use jinja templating in the bash operator to access them at runtime.

here's an example showing how you would use an airflow variable in the bash command, similar logic applies for secret variables:

```python
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.models import Variable
from datetime import datetime

with DAG(
    dag_id='execute_windows_batch_file_secrets',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False
) as dag:
    username = Variable.get("windows_username")
    password = Variable.get("windows_password")
    windows_server_ip = Variable.get("windows_server_ip")

    execute_batch = BashOperator(
        task_id='execute_remote_batch',
        bash_command="""
            winrs -r:{{ var.value.windows_server_ip }} -u:{{ var.value.windows_username }} -p:{{ var.value.windows_password }}  "cmd /c c:\\path\\to\\your\\batchfile.bat"
        """,
        env={"windows_username": username, "windows_password": password,"windows_server_ip": windows_server_ip }
    )
```

in this example, the username, password and windows server ip are retrieved using airflow variables, assuming that you have defined them in the airflow ui or programmatically. using jinja templating they can be referenced inside the bash command. you can then do the same using secrets backends. this way, your passwords are not hardcoded into your dag files. it makes everything more secure and more maintainable. i cannot stress the importance of this point enough.

now, if you need to dive deeper into windows remote management and `winrs`, i'd suggest looking at the documentation from microsoft. their docs are generally pretty thorough and up to date, also try to search for `powershell remoting` this will help you understand the background technology at play. as for general airflow best practices, the book "data pipelines with apache airflow" by bas p. hamer and marcel rademacher, is a good start, although it does not go deep into windows related issues. i would also recommend the airflow documentation itself as it is a very mature product and its documentation is well maintained.

finally, always remember to handle errors gracefully. for example, you can implement retry logic and error logging for your tasks. this will prevent silent failures and will make your dags much more robust. remember, debugging a poorly written dag is never fun, trust me on this. i've had days when the errors from the dag were so cryptic that i felt i was trying to decipher ancient hieroglyphs. and it's always something simple when you finally find the issue, like an extra space somewhere.

so that's about it. running batch files from airflow is possible, but it requires some planning and configuration. i hope this helps you avoid some of the pitfalls i encountered. and if it doesn’t, well, just remember to always double check your file paths and command syntax, you'd be surprised how many issues are just typos and missing characters, like that time when i forgot to add the 'r' in a `winrs` command and it took me a whole afternoon to figure it out, i tell ya, sometimes i feel like i should have become a botanist.
