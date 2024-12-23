---
title: "How can I create an Airflow DAG on a cloud VM using SSH?"
date: "2024-12-23"
id: "how-can-i-create-an-airflow-dag-on-a-cloud-vm-using-ssh"
---

Alright, let’s tackle this. I remember dealing with a similar setup a few years back when orchestrating ETL pipelines on a series of VMs before we fully transitioned to Kubernetes. It’s a common need, especially when dealing with legacy systems or situations where direct cloud service integration isn't feasible. The key is to manage the remote deployment and execution of the DAG effectively via ssh. It requires a clear understanding of airflow's architecture, remote execution paradigms, and secure shell interactions. I've learned the hard way that skipping steps here can lead to a debugging nightmare, so let's be methodical.

The core challenge is that Airflow, running on a central server or perhaps a cloud managed service, needs to trigger and monitor tasks that execute on a remote VM. This involves three major parts: securely transferring the DAG definition to the remote machine, triggering the DAG to start on the remote machine where airflow is also installed and making it operational, and finally, ensuring the remote execution tasks are correctly initiated.

First, concerning DAG deployment, while you *could* manually copy a dag file, that’s incredibly inefficient. A much better approach is to leverage Git. Have your dag definitions within a git repository. From there you can use an ssh based approach to pull the definitions down onto your vm. Assuming the VM also has airflow installed (and is configured), it will automatically pick up the deployed DAG files once they are in the designated 'dags' directory (usually configured in `airflow.cfg` or similar). Here's how a simple shell script might handle this part within the VM that airflow needs access to:

```bash
#!/bin/bash
# Assumes SSH keys are setup for git operations

cd /path/to/your/airflow/dags/
git pull origin main
```

In this script, `main` is a placeholder for your branch name. You might need to create the directory `/path/to/your/airflow/dags/` or ensure it exists if the airflow installation doesn’t have that already. Automating this with a cron job (or similar scheduler) to regularly check for updates is good practice, as long as it happens *before* a scheduled run time of your DAGs, ensuring your most up-to-date code is being executed.

Now, triggering the DAG. Here, the `ssh` command is crucial. We can use it to execute airflow commands remotely. In this scenario, you would typically have a second airflow instance running locally, and you can then use `ssh` from that locally running Airflow instance to interact with the remote one.
Here’s how you might define a simple dag on your local airflow server which manages the remote airflow instance:

```python
from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

with DAG(
    dag_id='remote_dag_trigger',
    start_date=datetime(2023, 10, 26),
    schedule_interval=None,
    catchup=False,
) as dag:
    trigger_remote_dag = BashOperator(
        task_id='trigger_remote_dag',
        bash_command='ssh -i /path/to/your/private_key -o StrictHostKeyChecking=no user@remote_vm "airflow dags trigger remote_dag_name"',
    )
```

Let's break this down. The `BashOperator` executes a command on the local Airflow server. The command itself is the `ssh` command connecting to `remote_vm`, using a private key for secure authentication (ensure that the permissions on the key are appropriate - typically 600). The `StrictHostKeyChecking=no` disables the host key checking (for convenience in this simplified example, but consider using known_hosts or a proper configuration in a real environment for better security), which is sometimes needed during initial setup if the host keys haven't been previously registered, but for production systems, this should be avoided. The most important part is the command that is executed on the remote server – `airflow dags trigger remote_dag_name`. This is what triggers the *remote* airflow instance to start the DAG with the identifier `remote_dag_name`. The `remote_dag_name` must be previously deployed to that server.

It’s crucial to recognize here that the *remote* airflow instance must have already parsed and loaded `remote_dag_name`. This is why the first script that does the git pull is necessary. Furthermore, the user under which Airflow is operating must have the permission to execute those tasks.

Finally, let’s address remote execution. Within your `remote_dag_name`, the tasks might perform some operation, say on some data files. Airflow’s built-in `SSHOperator` is suitable for this. Here is an example of how the `remote_dag_name` would be defined:

```python
from airflow import DAG
from airflow.providers.ssh.operators.ssh import SSHOperator
from datetime import datetime

with DAG(
    dag_id='remote_dag_name',
    start_date=datetime(2023, 10, 26),
    schedule_interval=None,
    catchup=False,
) as dag:
    remote_execution_task = SSHOperator(
       task_id='remote_task',
       ssh_conn_id='ssh_default', # use an airflow defined connection
       command="echo 'Hello from the remote server!' >> /tmp/remote_output.txt",
    )
```

Here, the `SSHOperator` needs to be configured with a *connection* through the airflow UI (or using environment variables or similar methods) that identifies the remote server that it will execute commands on. The `ssh_conn_id` references that previously created connection. The `command` is what will be executed via `ssh` on the remote host. In this example, it's simply appending a string to a file, but this can be any arbitrary shell script or binary that your system can run. Note, a connection will need to be configured using the `airflow connections` interface (either from the webui, or command line utility) using information about the remote machine.

Now, a few crucial points. Security is paramount when using ssh. Never hardcode passwords or private keys into code. Use Airflow’s connection management or utilize environment variables and secrets management to handle sensitive information. The example above uses a key based approach. Second, the remote airflow installation needs to be properly configured, including the same python environment, as required for your DAGs to run correctly. This is particularly true for dependencies that your DAG relies on, as it is responsible for parsing the DAG file. Lastly, ensure that the host on which the DAG *execution* occurs, has permission to write to the locations indicated in your code, for things like files or data. It can be especially tricky to debug problems related to insufficient write permissions.

For further reading, I recommend focusing on a few areas. For a thorough grounding in secure shell and remote authentication, “SSH: The Secure Shell: The Definitive Guide” by Daniel J. Barrett and Richard E. Silverman is an excellent resource. For a deep dive into Apache Airflow, the official documentation is, of course, the best starting point; make sure to pay attention to documentation of the providers, as you need the ssh provider installed. Then, to understand better the specifics of distributed systems and job scheduling, “Designing Data-Intensive Applications” by Martin Kleppmann covers many of the core principles of these systems. You might also look into the principles of system administration more generally as well.

While setting up such a system can be a bit more work than deploying directly to a cloud-managed service, it remains essential knowledge for many complex workflows. Remember, methodical planning, careful security, and thoroughly testing your solution are the keys to success. Good luck with your setup!
