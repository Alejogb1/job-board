---
title: "How can an Airflow SSHOperator's output be passed to a subsequent task?"
date: "2025-01-30"
id: "how-can-an-airflow-sshoperators-output-be-passed"
---
The crucial limitation of Airflow's `SSHOperator` lies in its inherent design: it primarily focuses on command execution, not data retrieval.  While it provides a mechanism for executing commands on a remote server,  the standard output and standard error streams are not directly exposed as readily usable XCom values within the Airflow DAG.  My experience working on large-scale data pipelines across diverse infrastructure environments—including geographically distributed systems and cloud-based solutions—has highlighted this precise challenge. Directly capturing and passing the SSHOperator's output necessitates employing indirect methods.

**1. Clear Explanation:**

The solution revolves around two key strategies: modifying the remote command to write its output to a file, and then using a subsequent Airflow operator to fetch that file. The first operator, the `SSHOperator`, will execute a script or command that produces the desired output and saves it to a pre-defined location.  A second operator, such as `SFTPOperator` or a custom operator, then retrieves this file.  Finally, a third operator processes the retrieved data.  This approach effectively decouples the command execution from the data retrieval and handling, thereby enhancing robustness and maintainability.  Error handling, crucial in production systems, should be implemented at each stage, particularly considering the potential for network issues and remote server unavailability.


**2. Code Examples with Commentary:**

**Example 1: Using SFTPOperator for file transfer**

This example utilizes a simple bash script on the remote server to generate output and saves it to a file.  I've consistently found this approach to be efficient and easily scalable.


```python
from airflow import DAG
from airflow.providers.ssh.operators.ssh import SSHOperator
from airflow.providers.sftp.operators.sftp import SFTPOperator
from airflow.operators.python import PythonOperator
from datetime import datetime

with DAG(
    dag_id="ssh_output_to_sftp",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
) as dag:

    generate_output = SSHOperator(
        task_id="generate_output",
        ssh_conn_id="my_ssh_conn",
        command="bash /path/to/remote/script.sh",
    )

    get_output = SFTPOperator(
        task_id="get_output",
        ssh_conn_id="my_ssh_conn",
        local_filepath="/path/to/local/file.txt",
        remote_filepath="/path/to/remote/output.txt",
    )


    def process_output(**context):
        with open("/path/to/local/file.txt", "r") as f:
            output = f.read()
            # Process the output data here
            print(f"Output received: {output}")

    process_data = PythonOperator(
        task_id="process_data",
        python_callable=process_output,
    )

    generate_output >> get_output >> process_data

```

`/path/to/remote/script.sh` would contain a bash script that generates the desired output and writes it to `/path/to/remote/output.txt`.  The `ssh_conn_id` should be configured in your Airflow connections. Error handling (e.g., using `try-except` blocks within the `process_output` function) is essential to manage potential file read failures.


**Example 2:  Custom Operator for Enhanced Flexibility**

For more complex scenarios, especially those needing robust error handling and customized data parsing, I recommend building a custom operator.  This provides superior control over the data transfer and processing steps.

```python
from airflow.models.baseoperator import BaseOperator
from airflow.providers.ssh.hooks.ssh import SSHHook
import paramiko


class SSHOutputOperator(BaseOperator):
    def __init__(self, ssh_conn_id, command, remote_filepath, local_filepath, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ssh_conn_id = ssh_conn_id
        self.command = command
        self.remote_filepath = remote_filepath
        self.local_filepath = local_filepath

    def execute(self, context):
        hook = SSHHook(ssh_conn_id=self.ssh_conn_id)
        with hook.get_conn() as client:
            stdin, stdout, stderr = client.exec_command(self.command)
            #Check for errors during command execution
            err = stderr.read().decode()
            if err:
                raise Exception(f"Error executing command: {err}")
            # Transfer the file
            sftp = client.open_sftp()
            sftp.get(self.remote_filepath, self.local_filepath)
            sftp.close()
            return {"output_filepath": self.local_filepath}


#Usage in DAG similar to example 1
#...
#Note that get_output now uses SSHOutputOperator.
#...
```

This custom operator encapsulates SSH connection, command execution, and file transfer within a single unit, thereby promoting code reusability and easier maintenance.  Remember to handle exceptions appropriately to prevent pipeline failures.  I typically implement comprehensive logging within custom operators for better debugging.


**Example 3: Leveraging `subprocess` within a PythonOperator (Less Recommended)**


This approach utilizes a PythonOperator and the `subprocess` module to directly interact with the SSH connection.  While functional, it's generally less preferred due to reduced maintainability and potential security risks if not handled meticulously. I  generally only use this approach for simple cases or when integrating with existing internal libraries that already handle SSH connections.

```python
from airflow.providers.ssh.hooks.ssh import SSHHook
from airflow.operators.python import PythonOperator
import subprocess
import paramiko


def execute_ssh_command(context):
    hook = SSHHook(ssh_conn_id="my_ssh_conn")
    with hook.get_conn() as client:
        sftp = client.open_sftp()
        sftp.put("/path/to/local/script.sh", "/path/to/remote/script.sh")
        sftp.close()
        stdin, stdout, stderr = client.exec_command("bash /path/to/remote/script.sh")
        out, err = stdout.read(), stderr.read()
        if err:
             raise Exception(f"Error executing command: {err}")
        # Process the output here
        print(f"SSH output: {out.decode()}")

with DAG(...) as dag:
    # ...
    ssh_task = PythonOperator(
        task_id="execute_ssh_command",
        python_callable=execute_ssh_command,
    )
    #...
```

 This method requires careful consideration of security, especially regarding the transfer of the local script to the remote server.  Hardcoding sensitive information within the script should be avoided; using parameterized commands and environment variables is strongly recommended.


**3. Resource Recommendations:**

For in-depth understanding of Airflow operators, consult the official Airflow documentation.  Explore the source code of various Airflow providers for deeper insight into their implementation.  Familiarize yourself with best practices for secure SSH connections and file handling in Python.  A solid understanding of bash scripting will prove invaluable for crafting effective remote commands.  Mastering the use of custom operators, a key skill for advanced Airflow development, is recommended for greater control and maintainability.
