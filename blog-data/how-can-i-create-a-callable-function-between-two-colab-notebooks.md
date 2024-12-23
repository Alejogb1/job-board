---
title: "How can I create a callable function between two Colab notebooks?"
date: "2024-12-23"
id: "how-can-i-create-a-callable-function-between-two-colab-notebooks"
---

Okay, let’s tackle this. You're looking to establish a method for invoking a function defined in one Google Colab notebook from another. This isn’t a direct, out-of-the-box feature, so it requires a bit of a workaround, but it's certainly achievable, and I've found myself needing this exact functionality in a few complex projects involving collaborative research. I recall one specific instance while developing a distributed machine learning pipeline for a large image dataset. We had separate notebooks for data preprocessing and model training, each run by a different team member. Getting the preprocessed data over to the training notebook dynamically was crucial, and the following strategies proved reliable.

Essentially, the challenge here lies in the inherent isolated nature of Colab notebook execution environments. Each notebook operates within its own virtual machine, meaning you can't simply import code from one into another directly like you would with Python modules on your local file system. Therefore, we need to externalize the function code in a way that's accessible from both notebooks.

There are three primary methods I typically employ to establish this inter-notebook communication, which I'll elaborate on with code examples: using Google Drive as an intermediate storage, leveraging python modules hosted in a git repository, and utilising a service such as cloud functions.

**Method 1: Using Google Drive as an Intermediate**

This is probably the simplest method to get started. The principle is straightforward: you write your function in the source notebook, save it as a python file (.py) in your google drive, and then import it from the destination notebook. It's not the most elegant, but it's highly effective, particularly for small utility functions.

Here's how it's structured:

1.  **Source Notebook (where the function is defined):**

    ```python
    # In source_notebook.ipynb
    from google.colab import drive
    import os

    drive.mount('/content/drive') # Mount your Google Drive

    def add_two_numbers(a, b):
        """Adds two numbers and returns the sum."""
        return a + b

    # Specify the path where you want to save the python script in your drive
    script_path = '/content/drive/My Drive/shared_functions.py'

    with open(script_path, 'w') as f:
        f.write("def add_two_numbers(a, b):\n")
        f.write('    """Adds two numbers and returns the sum."""\n')
        f.write("    return a + b\n")

    print(f"Function saved to {script_path}")

    ```

    This first snippet mounts your drive, defines a function, and then writes it out as a `.py` script to your Google Drive.

2.  **Destination Notebook (where the function is called):**

    ```python
    # In destination_notebook.ipynb
    from google.colab import drive
    import sys
    import os

    drive.mount('/content/drive') # Mount your Google Drive

    # Specify the path where the python script is located
    script_path = '/content/drive/My Drive/shared_functions.py'

    sys.path.append(os.path.dirname(script_path))

    import shared_functions

    result = shared_functions.add_two_numbers(5, 3)
    print(f"Result of function call: {result}")

    ```

    The second snippet mounts the drive, adds the path containing the python script to python path and imports the shared functions. Note that the `script_path` variable must match in both notebooks.

    *Practical Considerations:* This method is good for rapid prototyping and smaller projects. However, it can become cumbersome for larger function libraries as it necessitates manually pushing updates to Google Drive. For anything beyond a couple of functions, consider method 2.

**Method 2: Leveraging Git and Python Modules**

For more extensive shared code and better version control, utilising git is superior. This involves hosting the shared functions as a Python module in a git repository (like GitHub, GitLab or Bitbucket). This method requires a little more initial setup, but it’s much more maintainable in the long term, especially in collaborative environments.

1.  **Create a Python module and host it on Git:**

    First, create a directory structure like this:

    ```
    my_shared_module/
        __init__.py
        my_functions.py
    ```

    The `__init__.py` file can be empty. The `my_functions.py` will contain your functions:

    ```python
    # my_functions.py
    def multiply_numbers(a, b):
        """Multiplies two numbers and returns the product."""
        return a * b
    ```

    Commit this code to a git repository (e.g., github.com/your_username/my_shared_module).

2.  **Destination Notebook:**

    ```python
    # destination_notebook.ipynb
    import os
    import sys
    from subprocess import run

    # Clone your repo
    repo_url = "https://github.com/your_username/my_shared_module.git"
    repo_path = "/content/my_shared_module"

    if not os.path.exists(repo_path):
        run(['git', 'clone', repo_url, repo_path], check=True)
        print("Module Cloned")

    # Add the module's directory to the Python path
    sys.path.append(repo_path)

    from my_shared_module.my_functions import multiply_numbers

    result = multiply_numbers(4, 7)
    print(f"Result of function call: {result}")

    ```

    The code clones the git repository to the notebook's VM, appends the cloned repository's directory to Python's path, and then imports the required functions.

    *Practical Considerations:* Git provides version control, allowing multiple contributors to work on the shared function library while maintaining version history. This scales much better than the Google Drive approach but requires the initial setup of a git repository. For deeper dives into version control best practices, I suggest looking into resources like Pro Git by Scott Chacon and Ben Straub.

**Method 3: Cloud Functions**

For more complex scenarios involving asynchronous tasks, cloud functions (like Google Cloud Functions or AWS Lambda) offer a more robust solution. You deploy the function as an API endpoint, and both notebooks can invoke it via HTTP requests. This is my preferred approach for productionised scenarios.

1.  **Cloud Function Deployment:**

    This step involves writing a Cloud Function that can expose your functions as an api endpoint. The deployment process would depend on the specific cloud provider being used.

    Let’s say your cloud function receives a payload with values `a` and `b`, then it executes a division:

    ```python
    # cloud function code (e.g. main.py)
    import functions_framework
    import json

    @functions_framework.http
    def divide_numbers(request):
        """Divides two numbers and returns the result."""

        try:
            request_json = request.get_json()
            a = request_json['a']
            b = request_json['b']
            return json.dumps({"result": a/b}), 200, {'ContentType':'application/json'}

        except Exception as e:
            return json.dumps({"error":str(e)}), 500, {'ContentType':'application/json'}

    ```

    Deploy this function to a suitable environment such as Google Cloud Function.

2.  **Both Notebooks can then invoke this:**

    ```python
    # In both Notebooks:
    import requests
    import json

    # Replace with the url of your cloud function
    url = "YOUR_CLOUD_FUNCTION_URL"
    headers = {'Content-type': 'application/json'}
    payload = {"a": 10, "b": 2}


    try:
      response = requests.post(url, data = json.dumps(payload), headers=headers)
      response.raise_for_status() # Raises an exception for bad status codes
      data = response.json()
      print(f"Result of function call: {data['result']}")

    except requests.exceptions.RequestException as e:
      print(f"An error occurred: {e}")
    ```

    Both notebooks can call this API via `requests.post`.

    *Practical Considerations:* This method is more complex to set up initially but is the most robust for complex workflows, allowing for asynchronous operations and scalability. For detailed understanding of the concepts behind cloud functions, I'd recommend looking into the Cloud Native Computing Foundation's resources and the documentation for specific cloud provider's services.

In conclusion, the best approach to creating callable functions between Colab notebooks depends on the scale and complexity of your project. For small, simple function sharing, Google Drive suffices. For medium-sized projects with version control requirements, git modules are a good choice. For complex, production-ready applications, cloud functions offer robust asynchronous capabilities. I've used each of these methods successfully in various contexts and found them generally reliable. Remember to carefully consider the project’s requirements and pick the method that aligns best with them. Good luck.
