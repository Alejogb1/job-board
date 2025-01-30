---
title: "How can SSH operator output be captured as a variable in Airflow?"
date: "2025-01-30"
id: "how-can-ssh-operator-output-be-captured-as"
---
Capturing SSH operator output directly into an Airflow variable requires careful consideration of the asynchronous nature of SSH execution and Airflow's task orchestration.  My experience working with large-scale data pipelines has shown that relying solely on the SSHOperator's inherent return values is insufficient for robust output handling;  a more structured approach is necessary to reliably capture and manage the potentially voluminous output of remote commands.

The core challenge lies in the fact that the SSHOperator, by design, doesn't inherently block until the remote command completes and returns its standard output.  Instead, it initiates the connection and command execution, then continues with the Airflow DAG execution.  Therefore, attempting to directly assign the operator's return value to a variable will often result in an empty or partially populated variable due to the timing discrepancy.  To circumvent this, a more sophisticated method involving either a custom operator, the use of `xcom`, or a dedicated file-based approach is necessary.

**1.  Custom SSH Operator with Output Capture:**

The most robust solution involves creating a custom SSH operator extending the existing `SSHOperator`. This operator will explicitly manage the SSH connection, capture the standard output and error streams, and then push the output to an Airflow XCom, allowing access from downstream tasks.

```python
from airflow.providers.ssh.hooks.ssh import SSHHook
from airflow.providers.ssh.operators.ssh import SSHOperator
from airflow.decorators import task
from airflow.models import Variable

class CustomSSHOperator(SSHOperator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def execute(self, context):
        hook = SSHHook(ssh_conn_id=self.ssh_conn_id)
        with hook.get_connection() as conn:
            stdin, stdout, stderr = conn.exec_command(self.command)
            output = stdout.read().decode().strip()
            error = stderr.read().decode().strip()

            if error:
                raise Exception(f"SSH command failed: {error}")

            # Store the output in XCom
            context['ti'].xcom_push(key='ssh_output', value=output)


#Example DAG utilization:
with DAG(dag_id='ssh_output_capture', start_date=datetime(2023, 1, 1), schedule=None, catchup=False) as dag:
    get_ssh_output = CustomSSHOperator(
        task_id='get_remote_data',
        ssh_conn_id='my_ssh_connection',
        command='ls -l /tmp'
    )

    # Access the output from downstream tasks
    @task
    def process_output(ti):
        output = ti.xcom_pull(task_ids='get_remote_data', key='ssh_output')
        #process the output string here
        print(f"SSH output: {output}")

    process_output_task = process_output()
    get_ssh_output >> process_output_task

```

This custom operator explicitly reads both `stdout` and `stderr` ensuring error handling and complete output capture. The captured output is then pushed to XCom using the task instance context (`context['ti']`).  This allows downstream tasks to access the result using `ti.xcom_pull`.  The crucial aspect is the explicit reading of the output streams, resolving the asynchronous execution problem.  Error handling is integrated to manage potential remote command failures.

**2. Utilizing Airflow XCom with Standard SSHOperator:**

While less robust than a custom operator, using XCom with the standard `SSHOperator` can be a viable option for simpler scenarios.  This method leverages a post-execution task to retrieve the output from a file written by the remote command.

```python
from airflow.providers.ssh.operators.ssh import SSHOperator
from airflow.decorators import task
from airflow.models import Variable

with DAG(dag_id='ssh_output_xcom_file', start_date=datetime(2023,1,1), schedule=None, catchup=False) as dag:
    write_to_file = SSHOperator(
        task_id='write_remote_file',
        ssh_conn_id='my_ssh_connection',
        command='echo "This is my output" > /tmp/airflow_output.txt'
    )

    @task
    def read_remote_file(ti):
        read_command = SSHOperator(
            task_id='read_output',
            ssh_conn_id='my_ssh_connection',
            command='cat /tmp/airflow_output.txt'
        ).execute(context=ti)
        return read_command

    read_output_task = read_remote_file()
    write_to_file >> read_output_task
```

This approach introduces a two-step process. First, the SSH operator executes a command that writes the desired output to a file on the remote server. A subsequent task then retrieves the contents of that file using another SSH operator. The output is then implicit in the `execute()` command's return. While simpler, it relies on the existence and accessibility of the temporary file.  Error handling needs to be explicitly added to handle file write and read failures.  The cleanup of the temporary file on the remote server should also be considered as part of a more production-ready solution.


**3.  Direct Output Redirection and File Transfer:**

This method redirects the remote command's output to a file, which is then transferred to the Airflow environment.  This circumvents the need for XCom entirely.

```python
from airflow.providers.ssh.operators.ssh import SSHOperator
from airflow.providers.sftp.operators.sftp import SFTPOperator
from airflow.models import Variable
from airflow.utils.dates import days_ago


with DAG(dag_id='ssh_output_file_transfer', start_date=days_ago(2), schedule=None, catchup=False) as dag:
    execute_command = SSHOperator(
        task_id='execute_remote_command',
        ssh_conn_id='my_ssh_connection',
        command='ls -l /tmp > /tmp/remote_output.txt'
    )

    transfer_file = SFTPOperator(
        task_id='transfer_output_file',
        ssh_conn_id='my_ssh_connection',
        local_filepath='/tmp/local_output.txt',
        remote_filepath='/tmp/remote_output.txt'
    )

    # Post processing
    @task
    def process_local_file(ti):
        with open('/tmp/local_output.txt', 'r') as f:
            output = f.read()
            print(f"File Content: {output}")

    process_local = process_local_file()
    execute_command >> transfer_file >> process_local
```


This method uses an SSHOperator to execute the command, redirecting its output to a file on the remote server. Subsequently, an SFTPOperator transfers that file to the Airflow environment's local filesystem.  Finally, a post-processing task reads and processes the local file. This requires configuring SFTP access in addition to SSH, but it offers a clear separation between remote execution and local processing.  Error handling should be implemented for both the SSH and SFTP operations, and the temporary files should be cleaned up.

**Resource Recommendations:**

For a deeper understanding of Airflow operators, refer to the official Airflow documentation. Consult the documentation for your specific SSH library (paramiko, for instance) to familiarize yourself with its intricacies for error handling and robust output management.  Understanding asynchronous programming concepts will significantly aid in grasping the challenges and solutions presented here.  Exploring Python's `subprocess` module might provide further insights into managing external processes.
