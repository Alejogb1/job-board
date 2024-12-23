---
title: "How do I run docker images in GCP Compute Engine with Apache Airflow?"
date: "2024-12-16"
id: "how-do-i-run-docker-images-in-gcp-compute-engine-with-apache-airflow"
---

, let’s talk about deploying docker images within google compute engine while orchestrating it all with apache airflow. It's a problem I’ve tackled a few times over the years, most notably when we migrated our data processing pipeline from an aging on-premise setup to a cloud-native architecture back in 2018. The initial attempt, frankly, was a bit of a mess. We quickly realized the manual deployment scripts weren't cutting it, and that's where the combined power of docker, gcp compute engine, and airflow really started to shine. The key is understanding how these pieces interoperate and choosing the appropriate airflow operators.

First and foremost, you’re not directly running a docker container *inside* a compute engine instance (unless you’re using a container-optimized os, which is a slightly different beast). Rather, you’ll be using airflow to instruct a compute engine instance to pull and execute your docker image. This typically involves creating a compute engine instance, and then, within that instance, using commands to execute your docker image using the docker cli.

The general workflow, as I've found it, breaks down into a few steps. You need an image repository, typically google container registry (gcr). You need an airflow environment that can communicate with gcp. You need a compute engine instance configured correctly, ideally with docker already installed, or you need to use startup scripts to handle that automatically when the instance is created. Lastly, you need airflow dag definitions that utilize operators to trigger the necessary commands on your compute engine instance. Let's get into some specific code examples.

For airflow, we are going to use the `sshoperator`. It's the workhorse in this context. The trick lies in how you configure the operator to securely access your compute engine instance and then execute the correct docker commands. To start, you will need to create a connection in Airflow for the target server with the correct credentials. I'll assume that you have that in place and the connection id is `gcp_compute_instance_ssh`. Here’s how you can define a simple airflow dag:

```python
from airflow import DAG
from airflow.providers.ssh.operators.ssh import SSHOperator
from datetime import datetime

with DAG(
    dag_id='docker_on_gce_example',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:
    run_docker_task = SSHOperator(
        task_id='run_docker_container',
        ssh_conn_id='gcp_compute_instance_ssh',
        command="""
            docker run -d \
              --restart always \
              my-gcr-repository/my-docker-image:latest
        """
    )
```

In this first snippet, we have a basic dag, and the ssh operator with a single command. We are directly running the docker container and detaching it, so it runs as a service in the background. Here's the breakdown:

*   `ssh_conn_id`: Specifies the pre-configured ssh connection within airflow. Make sure you've created it with the correct host and authentication method (ssh key is preferred).
*   `command`: This is the series of commands that will be executed on the compute engine instance. In this case, the `docker run` command, where we're specifying our image from our gcr.

A common issue I used to encounter with this was the `docker` executable not being available, usually due to PATH problems or missing installation. I always recommend adding the full path to the docker binary if you're unsure, and ensure that docker is installed.

Now, let’s say we want more control over our container; we might want to pass some environment variables. We can modify our `command` to do this, and also include docker stop/remove commands for a clean start:

```python
from airflow import DAG
from airflow.providers.ssh.operators.ssh import SSHOperator
from datetime import datetime

with DAG(
    dag_id='docker_on_gce_env_vars',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:
    run_docker_with_env_task = SSHOperator(
        task_id='run_docker_container_env',
        ssh_conn_id='gcp_compute_instance_ssh',
        command="""
            docker stop my-container || true
            docker rm my-container || true
            docker run -d \
              --name my-container \
              -e MY_VAR1='my_value1' \
              -e MY_VAR2='my_value2' \
              --restart always \
              my-gcr-repository/my-docker-image:latest
        """
    )
```

This snippet expands on the first example. We are now doing the following:
* `docker stop my-container || true`: This command will stop a container named `my-container` if it exists. The `|| true` makes the command not error if the container is not running or doesn't exist.
*   `docker rm my-container || true`: Similar to the stop command, this removes the container if it exists, preventing issues with already running containers.
*   `-e MY_VAR1='my_value1' -e MY_VAR2='my_value2'`: we're passing through environment variables to be used by your container.
*   `--name my-container`: we name the container so we can stop and remove it easily.

Sometimes, you might need to perform additional setup before running your container, such as pulling the latest image or building it. We can incorporate this into the `command` section, too. Also, it is essential to be aware of disk space issues. Always consider cleaning up after your containers when you stop/remove them.

```python
from airflow import DAG
from airflow.providers.ssh.operators.ssh import SSHOperator
from datetime import datetime

with DAG(
    dag_id='docker_on_gce_pull_build',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:
    setup_and_run_task = SSHOperator(
        task_id='setup_and_run_container',
        ssh_conn_id='gcp_compute_instance_ssh',
         command="""
            docker stop my-container || true
            docker rm my-container || true
            docker pull my-gcr-repository/my-docker-image:latest
            docker run -d \
              --name my-container \
              -e MY_VAR1='my_value1' \
              --restart always \
              my-gcr-repository/my-docker-image:latest
         """
    )
```

In this final example, we include the `docker pull` command to ensure we're using the latest available image from the repository, before starting the container. This is crucial when you frequently update your docker images.

For further learning, I'd recommend several resources. First, for a deeper dive into airflow, “airflow in action” by jake mcguire is highly recommended, and the apache airflow documentation itself. For gcp and docker, the official documentation is excellent. Specifically, I would encourage you to explore the google cloud documentation related to compute engine startup scripts and container registry. Understanding how these two function together is critical in automating these deployments. Additionally, "docker deep dive" by nigel poulton offers an outstanding practical perspective on working with docker and understanding containers at a deeper level, which I’ve found invaluable over the years.

Deploying docker images on gcp compute engine with airflow, while it can seem intricate at first, quickly becomes manageable with the right understanding and approach. It’s not just about writing code but also about structuring your workflows correctly and thinking through the practical considerations of operating such a setup in a production environment.
