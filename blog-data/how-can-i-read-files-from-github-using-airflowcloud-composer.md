---
title: "How can I read files from Github using Airflow/Cloud Composer?"
date: "2024-12-23"
id: "how-can-i-read-files-from-github-using-airflowcloud-composer"
---

Alright, let's tackle reading files from GitHub using Airflow, particularly within a Cloud Composer context. This is a common scenario, and frankly, I've debugged this enough times over the years to have a fairly clear picture of the best approaches. The initial inclination might be to reach for a simple bash operator, but that can quickly become unwieldy and less maintainable. So, we'll explore more robust methods.

When I encountered this a few years back at a previous company, we were deploying data pipelines that pulled in configuration files and occasionally small datasets from a private GitHub repository. We initially relied on simple `git clone` commands in bash operators, but we soon realized that security, authentication, and versioning became messy quickly. We needed a more controlled and integrated solution.

The core problem here isn't necessarily *reading* the file; it's securely and efficiently *accessing* the file from GitHub within the Airflow environment. Cloud Composer runs on Google Cloud Platform (GCP), and there's a beautiful set of features within GCP that we can leverage. The most effective path generally involves a combination of these components:

1.  **GitHub Personal Access Tokens (PATs):** Instead of embedding credentials directly in code, which is a huge security no-no, we use PATs to authenticate with GitHub. These tokens have granular permissions, letting us control exactly what our pipeline can access.
2.  **Google Secret Manager:** We store the GitHub PAT securely within Google Secret Manager. Composer integrates seamlessly with Secret Manager, allowing Airflow DAGs to retrieve secrets without exposing them.
3.  **Python Libraries:** We use the `requests` or `PyGithub` library, both of which excel at handling HTTP requests and interacting with the GitHub API, to actually grab the file contents.

Let's break this down with some practical examples. Assume we want to read a `config.json` file located in a specific repository and branch.

**Example 1: Using `requests` with a raw file URL**

This approach is straightforward for public repositories or for files you wish to retrieve via a direct URL if you have appropriate authentication. In our company, this was useful for retrieving relatively static configuration data.

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import requests
import json
from google.cloud import secretmanager

def fetch_github_file_requests(repo_owner, repo_name, branch, file_path, secret_id):
    client = secretmanager.SecretManagerServiceClient()
    secret_version_name = client.secret_version_path(
        project=client.project_path(project='your-gcp-project'),
        secret=secret_id,
        secret_version='latest'
    )
    response = client.access_secret_version(name=secret_version_name)
    github_pat = response.payload.data.decode("UTF-8")

    url = f"https://raw.githubusercontent.com/{repo_owner}/{repo_name}/{branch}/{file_path}"
    headers = {'Authorization': f'token {github_pat}'}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes
        file_content = response.text
        return json.loads(file_content) # Assuming it's JSON
    except requests.exceptions.RequestException as e:
        print(f"Error fetching file: {e}")
        return None

with DAG(
    dag_id='fetch_github_file_requests_dag',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False
) as dag:
    fetch_file_task = PythonOperator(
        task_id='fetch_github_config',
        python_callable=fetch_github_file_requests,
        op_kwargs={
            'repo_owner': 'your-github-user',
            'repo_name': 'your-repo',
            'branch': 'main',
            'file_path': 'config.json',
            'secret_id': 'your-github-pat-secret'  # Secret name in Secret Manager
        }
    )
```

*Explanation:*
This snippet demonstrates how to fetch a file directly using its raw URL. It's important to handle potential errors gracefully using `response.raise_for_status()`. Remember to replace placeholders like `your-gcp-project`, `your-github-user`, `your-repo`, and `your-github-pat-secret` with your actual values. The `requests` library makes HTTP calls clean, and the Google Secret Manager client handles secure retrieval of the token. This is often sufficient for simpler use cases.

**Example 2: Using `PyGithub` for a more feature-rich approach**

When we needed more advanced functionality, like checking for specific commits or interacting with different branches dynamically, we switched to `PyGithub`. It provides a Pythonic interface to the entire GitHub API.

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from github import Github
from google.cloud import secretmanager
import json

def fetch_github_file_pygithub(repo_owner, repo_name, branch, file_path, secret_id):
    client = secretmanager.SecretManagerServiceClient()
    secret_version_name = client.secret_version_path(
        project=client.project_path(project='your-gcp-project'),
        secret=secret_id,
        secret_version='latest'
    )
    response = client.access_secret_version(name=secret_version_name)
    github_pat = response.payload.data.decode("UTF-8")

    g = Github(github_pat)
    repo = g.get_user(repo_owner).get_repo(repo_name)
    try:
        file_content = repo.get_contents(file_path, ref=branch)
        return json.loads(file_content.decoded_content.decode()) #Assuming its JSON
    except Exception as e:
        print(f"Error fetching file: {e}")
        return None

with DAG(
    dag_id='fetch_github_file_pygithub_dag',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False
) as dag:
    fetch_file_task = PythonOperator(
        task_id='fetch_github_config',
        python_callable=fetch_github_file_pygithub,
        op_kwargs={
            'repo_owner': 'your-github-user',
            'repo_name': 'your-repo',
            'branch': 'main',
            'file_path': 'config.json',
            'secret_id': 'your-github-pat-secret'  # Secret name in Secret Manager
        }
    )
```

*Explanation:*
This example leverages the `PyGithub` library for a more API-driven approach. It first obtains a `Github` object using the PAT, then fetches the repository, and finally, retrieves the file's content. This pattern is helpful when you need more metadata associated with the file or if you plan on doing more with the GitHub API. Similar to the first example, handling potential errors with `try-except` blocks is critical.

**Example 3: Handling larger files and potential API rate limits**

For very large files or high-frequency fetching, it's wise to consider potential rate limits imposed by the GitHub API. This example shows using a local temporary file.

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import requests
from google.cloud import secretmanager
import os

def fetch_large_github_file(repo_owner, repo_name, branch, file_path, secret_id, tmp_path):
    client = secretmanager.SecretManagerServiceClient()
    secret_version_name = client.secret_version_path(
        project=client.project_path(project='your-gcp-project'),
        secret=secret_id,
        secret_version='latest'
    )
    response = client.access_secret_version(name=secret_version_name)
    github_pat = response.payload.data.decode("UTF-8")

    url = f"https://raw.githubusercontent.com/{repo_owner}/{repo_name}/{branch}/{file_path}"
    headers = {'Authorization': f'token {github_pat}'}

    try:
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status() #Check HTTP status

        with open(tmp_path, 'wb') as f:
             for chunk in response.iter_content(chunk_size=8192):
                 f.write(chunk)
        return tmp_path # returns path to the file
    except requests.exceptions.RequestException as e:
        print(f"Error fetching file: {e}")
        return None

with DAG(
    dag_id='fetch_large_github_file_dag',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False
) as dag:
    fetch_file_task = PythonOperator(
        task_id='fetch_github_large_file',
        python_callable=fetch_large_github_file,
        op_kwargs={
            'repo_owner': 'your-github-user',
            'repo_name': 'your-repo',
            'branch': 'main',
            'file_path': 'large_data.csv',
             'secret_id': 'your-github-pat-secret', # Secret name in Secret Manager
            'tmp_path': '/tmp/large_data.csv' #local tmp file
        }
    )
```

*Explanation:* This shows how to save to disk to efficiently handle larger files. The `stream=True` parameter makes the code fetch the data in chunks which reduces memory usage.  It also helps with avoiding the rate limit issues. The local file created is accessible by other tasks and allows you to process the file without loading all data into memory.

**Further Exploration**

For a deeper understanding of the relevant technologies, I recommend exploring the following resources:

*   **GitHub API Documentation:** Directly from the source; it’s crucial for understanding rate limits, authentication, and API endpoints.
*   **Google Cloud Secret Manager documentation:** This dives into the best practices for storing and retrieving secrets in GCP.
*   **Python `requests` library documentation:**  This provides a detailed look at HTTP requests, error handling, and stream handling.
*   **`PyGithub` library documentation:** This is invaluable for anyone planning on leveraging the full power of the GitHub API.
*   **"Effective Python: 90 Specific Ways to Write Better Python" by Brett Slatkin:** This provides best practices for Python code, focusing on efficiency and clarity.

In closing, reading files from GitHub in Airflow is achievable with a bit of setup using Secret Manager, a reliable HTTP library, and a deep understanding of your data and workflow. My experience shows that starting with secure authentication and error handling from the outset can save a lot of time. It's about creating solutions that are not just functional but also maintainable and secure in the long run.
