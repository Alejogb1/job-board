---
title: "How can I read files from GitHub with Airflow (Cloud Composer)?"
date: "2024-12-16"
id: "how-can-i-read-files-from-github-with-airflow-cloud-composer"
---

Okay, let’s tackle this. I’ve certainly been down the road of integrating GitHub with Airflow quite a few times, particularly when trying to orchestrate data pipelines that depend on configurations or scripts stored in version control. It's a common need, and there are definitely best practices that can make this process smoother and more reliable than just throwing together a bash operator with `curl`.

Frankly, the direct "reading" of files from GitHub isn't really what you want. What we're actually after is *accessing* them in a controlled way, often during task execution within Airflow. Think about it; we're not looking to parse massive repositories or do code analysis at runtime. Typically, we want specific files – perhaps a data schema definition, a configuration file, or a small utility script.

My experience leads me to three primary methods, each with its own set of trade-offs. Let's break them down.

**Method 1: Using the GitHub API and `requests` (or similar)**

This approach is direct and often the first thing people try. You interact with the GitHub API to fetch the file content directly, decoding it and making it available to your task. It's straightforward to implement, but be mindful of API rate limits and the overhead of fetching data each time. The main advantage is direct access, meaning no extra file system operations outside of what you define in your code.

Here’s a working example in Python, suitable for a PythonOperator within Airflow:

```python
import requests
import json
from airflow.decorators import task

@task
def fetch_github_file(owner, repo, path, branch='main', token=None):
    """Fetches a file from a GitHub repository using the GitHub API."""
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={branch}"
    headers = {}
    if token:
       headers["Authorization"] = f"token {token}"
    response = requests.get(url, headers=headers)
    response.raise_for_status()  # Raise an exception for bad status codes

    content = response.json()
    if content.get('type') == 'file':
        decoded_content = requests.get(content['download_url']).content.decode('utf-8')
        return decoded_content
    else:
        raise ValueError(f"The path '{path}' does not point to a single file.")

if __name__ == '__main__':
    file_content = fetch_github_file.override(task_id="fetch_test_file")(
        owner="my-org",
        repo="my-repo",
        path="config/my_config.json",
        branch="main",
        token="your-github-token"
    )
    print(file_content)
```

In your actual airflow dag, you would use a PythonOperator calling `fetch_github_file` and likely passing the fetched file as a parameter to a downstream task via XCom, ensuring you keep your sensitive token managed outside of the dag definition itself (consider using Airflow variables or secrets backends).

**Method 2: Git Cloning within an Airflow task**

This method is a bit heavier, but very suitable if you need access to multiple files within the repository or need frequent updates. You'd use `git clone` within a BashOperator (or similar) to pull the repository to a local directory within the executor's environment, and then your subsequent tasks can access the necessary files locally. This reduces API calls but comes with its own management overhead, specifically managing the cloning operation and ensuring the correct version is present. It's advantageous for complex projects, however.

Here's an example of a BashOperator within your Airflow DAG definition:

```python
from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

with DAG(
    dag_id='git_clone_example',
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False,
) as dag:

    git_clone = BashOperator(
        task_id='git_clone_repo',
        bash_command='''
            git clone https://github.com/my-org/my-repo.git /tmp/my_repo
            cd /tmp/my_repo
            git checkout main #or specific commit hash
        '''
    )

    process_config = BashOperator(
        task_id='process_config_file',
        bash_command='''
            cat /tmp/my_repo/config/my_config.json | jq '.'
            rm -rf /tmp/my_repo
        ''',
        dag=dag
        )

    git_clone >> process_config
```

Here, a BashOperator clones the repository to `/tmp/my_repo` and then a following BashOperator accesses the configuration file. Remember to adjust the repository URL and path to the config file according to your specific setup and ensure that the `/tmp` directory or your chosen directory is usable by the airflow worker. We use jq here for formatting but you could equally process this file using python operators after reading it in.

**Method 3: Leveraging Cloud Storage (e.g. Google Cloud Storage or S3) for Config Distribution**

My preferred method, especially in cloud environments, involves treating GitHub as source control, not a live configuration provider for Airflow. Here, changes to files in GitHub would trigger an update process which involves moving those files to a cloud storage solution such as Google Cloud Storage, then the Airflow DAG directly fetches files from the cloud storage location. This decouples Airflow directly from git and provides a better scaling approach. For this method you need an intermediate process that, on change of a file in the repository will upload the file to cloud storage. This is usually done with github actions/workflows that trigger on a change to specific files. This allows for the configurations to be accessed extremely quickly by airflow while separating access and rate limiting concerns from Git. In the following example I will show a method of retrieving such configurations using the Google Cloud Storage hook in airflow.

```python
from airflow import DAG
from airflow.providers.google.cloud.transfers.gcs_to_local import GCSToLocalOperator
from airflow.decorators import task
from datetime import datetime
import json

with DAG(
    dag_id='gcs_config_example',
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False,
) as dag:
   download_config_from_gcs = GCSToLocalOperator(
        task_id="download_config_gcs",
        bucket="your-gcs-bucket",
        object_names="config/my_config.json",
        save_to_path="/tmp/my_config.json",
    )

   @task
   def process_downloaded_config():
       with open("/tmp/my_config.json","r") as f:
            config = json.load(f)
            print(f"Loaded config:{config}")
   process_config = process_downloaded_config()
   download_config_from_gcs >> process_config
```

Here, we use the `GCSToLocalOperator` to copy the configuration file to `/tmp`, followed by a python task to load it and do something with it. This provides fast and simple access while reducing dependancies on github at runtime. Ensure you adjust the bucket and object name to match your own GCS setup.

**Key Considerations and Recommendations**

Regardless of which approach you take, a few things remain critical. Firstly, use a specific commit hash or branch when accessing files from git, not just a branch name which can change between executions. Secondly, avoid hardcoding any credentials – leverage Airflow's connection management system. Thirdly, implement robust error handling to gracefully recover from any issues with GitHub API calls, git clone failures, or missing files from GCS. Finally, consider if the file you are accessing is indeed a config or data file, and not a code file that would require a different approach to access (i.e. moving to a deployment process rather than a config read).

For a deeper dive, I'd highly recommend reviewing the GitHub API documentation directly, especially regarding authentication, content retrieval, and rate limiting. For git basics, “Pro Git” by Scott Chacon and Ben Straub is a superb resource. If you choose the GCS approach, make sure you have reviewed the google cloud storage documentation and any documentation regarding triggering a workflow on change from a git repository. Understanding the limitations of each will help you avoid major headaches down the line.

In conclusion, while "reading" files from GitHub isn't a single step, these three methods provide practical ways to incorporate file access into your Airflow pipelines, and should be considered based on your specific use case and context. Choose the right approach for your needs, and you’ll find GitHub integration becomes a smooth, repeatable part of your workflow.
