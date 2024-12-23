---
title: "How can I run pip as a non-root user in an Airflow Docker Compose setup?"
date: "2024-12-23"
id: "how-can-i-run-pip-as-a-non-root-user-in-an-airflow-docker-compose-setup"
---

Ah, running pip without root in an Airflow docker-compose setup, I’ve certainly been down that path more times than I care to remember. It's one of those common sticking points when you're trying to establish a secure and maintainable workflow. It’s particularly relevant when you're aiming for a least-privilege environment, which, frankly, should be the default these days. Let’s dive into the nuances of achieving this, shall we?

The core of the problem typically arises because, by default, the user inside the Docker container is often `root`. This means any `pip install` command, unless explicitly instructed otherwise, operates with root privileges. While it might initially seem convenient, it’s a bad practice for a multitude of reasons, primarily security. Imagine a compromised dependency executing arbitrary code with root permissions within your container; that's a massive security breach waiting to happen. We want to avoid this entirely.

My experience on a project some years back involving complex data pipelines highlighted this issue. We initially relied on default Docker setup, with root user being the norm. A security audit quickly pointed out that this was a significant vulnerability. We then pivoted to a model that consistently executed `pip` operations as a non-root user. This not only addressed security concerns but also promoted better isolation and reduced potential cross-container interference.

The solution hinges on creating a non-root user within the Docker image and then instructing your Airflow components to operate as this user. Here are the crucial steps, which i'll break down with concrete examples using docker-compose configuration and related python code.

First, you need to bake a non-root user into your Dockerfile. I often use a user named `airflow` for clarity. The Dockerfile setup should resemble something like this:

```dockerfile
FROM apache/airflow:2.8.0-python3.10

ARG AIRFLOW_UID=50000
ARG AIRFLOW_GID=50000

RUN groupadd -g ${AIRFLOW_GID} airflow && \
    useradd -m -u ${AIRFLOW_UID} -g airflow airflow

USER airflow

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .
```

In this snippet, we're adding a user and group 'airflow' with specific user and group ids through the `ARG` and `RUN` directives. It is crucial to utilize `ARG` so these values can be easily overridden during build processes, for example, if your host machine has an existing user with a conflicting uid. We then explicitly change to that user with `USER airflow`. It's important to note the `--no-cache-dir` argument for `pip`, which reduces image size as it prevents caching of install packages during the build process, which might be unnecessary and add bloat to the docker image. Finally, copy the remaining files into the docker image. This ensures all following processes run as the specified non-root user.

Next, you'll need to make a few adjustments to your `docker-compose.yml` file. You’ll want to ensure the container’s entrypoint script is also executed by the `airflow` user, and you might want to ensure proper file ownership by creating volumes that allow this. For a specific example, take this snippet:

```yaml
version: '3.8'
services:
  airflow-webserver:
    image: my-custom-airflow-image # Use the image you've built
    ports:
      - "8080:8080"
    volumes:
      - ./dags:/opt/airflow/dags
      - ./logs:/opt/airflow/logs
    user: "50000:50000"
    environment:
      - AIRFLOW__CORE__EXECUTOR=LocalExecutor

  airflow-scheduler:
    image: my-custom-airflow-image
    volumes:
      - ./dags:/opt/airflow/dags
      - ./logs:/opt/airflow/logs
    user: "50000:50000"
    environment:
      - AIRFLOW__CORE__EXECUTOR=LocalExecutor
```

Notice the inclusion of `user: "50000:50000"` in each service definition. The numbers correspond to the user and group ids defined earlier with `AIRFLOW_UID` and `AIRFLOW_GID`. This forces the Docker container to operate under the context of the `airflow` user we created. Moreover, the volumes are mapped to directories, `/opt/airflow/dags` and `/opt/airflow/logs`, which means permissions need to be appropriate for the non-root user. Ensure that the local directories of your docker-compose setup are set up with the appropriate user ownership, i.e., `chown -R 50000:50000 dags` and `chown -R 50000:50000 logs` from outside the container. This prevents any permission issues when Airflow tries to write or read files.

Finally, you might encounter scenarios where you need to install Python packages at runtime, perhaps inside a DAG. Even then, you should avoid `pip install` inside your dags using shell operators. Instead, consider the use of Python Virtual Environments with `venv`. This approach enables to manage package dependencies within isolated environments, preventing potential conflicts and keeping the system clean. Here’s a python example for a DAG that uses venv:

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import subprocess
import os

def install_and_use_package():
    venv_path = '/opt/airflow/venv'  # Define path for virtual environment. Should be a path inside container where the airflow user has write permissions.
    package_name = 'requests' # Package to be installed

    # Create virtual environment if it does not exist
    if not os.path.exists(venv_path):
        subprocess.run(['python3', '-m', 'venv', venv_path], check=True)
    # Install the required package
    subprocess.run([f'{venv_path}/bin/pip', 'install', package_name], check=True)
    # Ensure packages are not cached.
    subprocess.run([f'{venv_path}/bin/pip', 'cache', 'purge'], check=True)

    # Use the package
    from venv.bin import requests
    try:
        response = requests.get('https://www.example.com')
        print(f"Status Code: {response.status_code}")
    except Exception as e:
        print(f"Error occurred: {e}")


with DAG(
    dag_id='virtual_environment_example',
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False
) as dag:
    install_and_use_task = PythonOperator(
        task_id='install_and_use_package',
        python_callable=install_and_use_package
    )
```

In this example, within `install_and_use_package()`, we are creating a new virtual environment, installing the necessary packages and using those packages. We should only use relative paths to the virtual environment location inside the container. This avoids any problems related to paths from the host. This example is using subprocess to interact with python tools, but an alternative is to use `virtualenv` package. The `check=True` ensures the script throws exception if any command exits with an error code. Finally the cache is cleared to reduce potential build issues.

For further reference, I would strongly recommend reading the official Docker documentation, especially the sections concerning Dockerfile instructions and user management. For a deeper dive into Airflow security best practices, the official Apache Airflow documentation is indispensable. Another excellent resource is “Docker in Action” by Jeff Nickoloff and Karl Matthias. It offers a detailed examination of Docker’s internals and how to achieve robust and secure container deployments. Finally, for a deeper understanding of python virtual environments, the documentation of the official python `venv` package is a must read.

In summary, running pip as a non-root user involves a combination of Dockerfile modifications, `docker-compose.yml` adjustments, and a secure packaging approach with virtual environments. While it adds a layer of complexity, the security and long-term maintainability benefits are absolutely worthwhile. This approach is about shifting away from a system where anything goes to one that prioritizes security, and that's ultimately beneficial for the stability of your applications.
