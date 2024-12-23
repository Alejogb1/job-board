---
title: "How can I integrate TestRail with Azure DevOps Pipeline?"
date: "2024-12-23"
id: "how-can-i-integrate-testrail-with-azure-devops-pipeline"
---

Alright, let's tackle this. I remember a project back in '18, migrating a large monolith to a microservices architecture, where we desperately needed a unified view of our testing efforts. We had our agile process humming along in Azure DevOps (ADO), but our test results in TestRail felt siloed. Integrating the two wasn’t trivial, but it was crucial to streamline our feedback loops and gain a holistic understanding of our build quality. The basic problem, at its core, was that ADO pipelines didn't intrinsically ‘know’ anything about TestRail. Therefore, we needed to bridge the gap. We weren't alone; many teams face this challenge. Here’s how we did it, and how you can approach it.

The primary integration method centers on leveraging the TestRail API. It allows us to programmatically interact with TestRail, updating test runs, marking results, and even creating new test cases or runs. The most effective path is to use this API through custom scripts within our ADO pipelines, rather than relying solely on out-of-the-box extensions, which can lack granular control.

Firstly, we need to consider *when* to interact with TestRail during our pipeline. Generally, it makes sense to update TestRail:

1.  **After test execution:** This involves taking the test results from your testing framework (e.g., NUnit, JUnit, pytest), parsing them, and then pushing them to the relevant TestRail test runs.
2.  **Upon pipeline completion (optionally):** After a successful deployment, you might want to mark your test runs as ‘complete’ or ‘passed’ globally in TestRail if applicable.

The key to a solid integration lies in these crucial steps:

*   **API Key Generation:** In TestRail, create a dedicated API key for this integration. Avoid using personal API keys; if a team member leaves, the integration might break.
*   **Secure Storage:** Store the API key and your TestRail URL as secure variables within your Azure DevOps project. This prevents exposing sensitive data in the pipeline definition.
*   **Scripting Language:** Choose a scripting language familiar to your team. I’ve primarily worked with Python for these types of tasks due to its robust libraries for handling HTTP requests and JSON parsing, but PowerShell or Bash are viable choices as well.
*   **Test Result Parsing:** The most complex part involves parsing your test framework's result output into a format that TestRail understands. The result usually needs to be translated into a structure that aligns with TestRail's API specification.
*   **Test Run Identification:** Determine how you will identify the correct TestRail test run to update. We used a convention of associating our ADO build IDs with specific TestRail run IDs through environment variables, but you can opt for other strategies such as using build tags or environment variables.

Let’s delve into code snippets to illustrate this, using Python as our scripting language.

**Example 1: Updating a test result**

This script assumes you've stored the TestRail URL, API key, and other required IDs (run ID, test case ID, result status) as secure variables in your ADO pipeline. It demonstrates how you might update a single test case in TestRail.

```python
import requests
import json
import os

TESTRAIL_URL = os.environ.get('TESTRAIL_URL')
TESTRAIL_API_KEY = os.environ.get('TESTRAIL_API_KEY')
TESTRAIL_RUN_ID = os.environ.get('TESTRAIL_RUN_ID')
TEST_CASE_ID = os.environ.get('TEST_CASE_ID')
TEST_RESULT_STATUS = os.environ.get('TEST_RESULT_STATUS')

def update_test_result(run_id, case_id, status_id):
    url = f"{TESTRAIL_URL}/index.php?/api/v2/add_result_for_case/{run_id}/{case_id}"
    headers = {'Content-Type': 'application/json'}
    auth = ('', TESTRAIL_API_KEY)
    data = {
        'status_id': int(status_id),
        'comment': 'Automated result from Azure DevOps Pipeline'
    }
    try:
        response = requests.post(url, headers=headers, auth=auth, data=json.dumps(data))
        response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
        print(f"Test case {case_id} updated successfully.")
    except requests.exceptions.RequestException as e:
        print(f"Failed to update test case {case_id}: {e}")

if __name__ == "__main__":
    update_test_result(TESTRAIL_RUN_ID, TEST_CASE_ID, TEST_RESULT_STATUS)
```

In the above example, we use environment variables to pass in the values from the ADO Pipeline. This script will make a `POST` request to TestRail's API, setting the test result for the specified case.

**Example 2: Parsing JUnit XML Result and Bulk Update**

This example simulates parsing a JUnit XML file (as most testing frameworks output results) and using a loop to update TestRail results for multiple cases. We'd actually parse XML here using a library like `xml.etree.ElementTree`, but for brevity, I'll simulate this with a dictionary.

```python
import requests
import json
import os

TESTRAIL_URL = os.environ.get('TESTRAIL_URL')
TESTRAIL_API_KEY = os.environ.get('TESTRAIL_API_KEY')
TESTRAIL_RUN_ID = os.environ.get('TESTRAIL_RUN_ID')

#Simulating JUnit XML Output as a Dictionary
junit_results = {
    'testcase1': {'testcase_id': 123, 'status': 1}, #1 = passed, other ids can be found from testrail status options
    'testcase2': {'testcase_id': 124, 'status': 5}, #5 = failed
    'testcase3': {'testcase_id': 125, 'status': 1}
}


def update_test_results(run_id, results_dict):
    url = f"{TESTRAIL_URL}/index.php?/api/v2/add_results_for_cases/{run_id}"
    headers = {'Content-Type': 'application/json'}
    auth = ('', TESTRAIL_API_KEY)
    test_results = []

    for _, case_data in results_dict.items():
        test_results.append({
            'case_id': case_data['testcase_id'],
            'status_id': case_data['status'],
            'comment': 'Automated Result from Azure DevOps Pipeline'
        })
    data = {'results': test_results}
    try:
        response = requests.post(url, headers=headers, auth=auth, data=json.dumps(data))
        response.raise_for_status()
        print(f"Test Results for Test Run {run_id} updated successfully.")
    except requests.exceptions.RequestException as e:
        print(f"Failed to update test cases: {e}")


if __name__ == "__main__":
    update_test_results(TESTRAIL_RUN_ID, junit_results)
```

This script shows the use of `add_results_for_cases`, which is a more performant way to update multiple tests. It takes a list of test results as input, making use of dictionaries to store test results details, and updates them in TestRail using its bulk endpoint.

**Example 3: Marking Test Run as Complete**

Finally, here's how to mark a test run as complete after all tests have run. We'd usually run this at the end of a pipeline stage after confirming successful execution and test results updates.

```python
import requests
import os

TESTRAIL_URL = os.environ.get('TESTRAIL_URL')
TESTRAIL_API_KEY = os.environ.get('TESTRAIL_API_KEY')
TESTRAIL_RUN_ID = os.environ.get('TESTRAIL_RUN_ID')

def close_test_run(run_id):
    url = f"{TESTRAIL_URL}/index.php?/api/v2/close_run/{run_id}"
    auth = ('', TESTRAIL_API_KEY)
    try:
        response = requests.post(url, auth=auth)
        response.raise_for_status()
        print(f"Test Run {run_id} closed successfully.")
    except requests.exceptions.RequestException as e:
        print(f"Failed to close test run {run_id}: {e}")

if __name__ == "__main__":
    close_test_run(TESTRAIL_RUN_ID)
```

This simple script uses the `close_run` endpoint to mark a specific test run in TestRail as complete. This completes the integration process, offering a more streamlined flow.

For in-depth knowledge, I’d suggest focusing on TestRail’s API documentation; it's extremely comprehensive and is constantly updated. Furthermore, ‘*Continuous Delivery: Reliable Software Releases through Build, Test, and Deployment Automation*’ by Jez Humble and David Farley provides a good theoretical foundation for how this integration fits into a larger CI/CD process. Additionally, understanding API Design concepts, found in ‘*API Design Patterns*’ by JJ Geewax, will help in designing resilient integration patterns.

In summary, integrating TestRail and Azure DevOps Pipelines requires a pragmatic, programmatic approach, centered around API usage. Focus on modular scripting, secure variable storage, and thorough error handling. While there are extensions available, the ability to control the integration directly through custom scripts is invaluable in large projects, enhancing the robustness of the integration and providing more flexibility. Using the code examples provided, combined with the recommended resources, should give you a clear path to set up a strong integration process between ADO pipelines and TestRail.
