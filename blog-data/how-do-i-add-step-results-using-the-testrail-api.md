---
title: "How do I add step results using the TestRail API?"
date: "2024-12-23"
id: "how-do-i-add-step-results-using-the-testrail-api"
---

Alright, let’s tackle this. I've seen this come up more than a few times, usually in the context of automating large test suites or needing granular feedback from complex processes. Adding step results to TestRail via its API, while not conceptually difficult, requires careful construction of your requests and a solid understanding of the underlying data structure. It's definitely not a one-size-fits-all situation; the specifics of your test framework will influence the implementation.

The core idea here revolves around the `add_result_for_case` endpoint, or specifically `add_result_for_case_with_steps` if you're dealing with test cases that break down into multiple steps. I'll primarily focus on the latter, because in my experience, that's where the real complexity usually lies. I once had to rebuild a rather chaotic automated framework; the original implementers hadn't properly considered how detailed results needed to be structured in TestRail, leading to practically useless overall test runs. We rectified that, but it was a lesson in planning and data integrity.

Essentially, when interacting with the TestRail API, you're sending structured data—usually JSON—over http. This request needs to specify the test case id, the test run id, and of course, the step result details.

Let's start by breaking down the anatomy of a typical request. The crucial part is the `steps` array, which is a list of dictionaries. Each dictionary represents a single step and needs, at minimum, the `status_id` and `content`. The `status_id` corresponds to TestRail's pre-defined status values (1 for passed, 2 for blocked, 4 for failed, etc.). The `content` can be any relevant text describing what happened at that step. Optionally, you can also include `actual` (the actual result, if it differs from what was expected), and `comment` (if you need to add notes).

Here's a straightforward python example using the `requests` library:

```python
import requests
import json

def add_step_results_example_1(test_run_id, case_id, steps_data, testrail_url, testrail_user, testrail_key):
    """Adds step results to TestRail for a given test case."""
    url = f"{testrail_url}/index.php?/api/v2/add_result_for_case/{case_id}/{test_run_id}"
    headers = {'Content-Type': 'application/json'}
    auth = (testrail_user, testrail_key)

    data = {
        'steps': steps_data,
        'comment': 'Automated step results added via API',
        'status_id': 1 #overall test status, even if steps failed
    }

    try:
        response = requests.post(url, headers=headers, auth=auth, data=json.dumps(data))
        response.raise_for_status() # Raises an exception for non-2xx status codes
        print(f"Successfully added results for case {case_id}, run {test_run_id}.")
    except requests.exceptions.RequestException as e:
        print(f"Error adding results: {e}")


if __name__ == '__main__':
    # replace with your actual data
    test_run_id = 123
    case_id = 456
    testrail_url = "https://your-testrail-instance.testrail.io"
    testrail_user = "your_username"
    testrail_key = "your_api_key"

    steps_data = [
        {'content': "Step 1: User logs in", 'status_id': 1},
        {'content': "Step 2: User navigates to dashboard", 'status_id': 1},
        {'content': "Step 3: Data is loaded", 'status_id': 4, 'comment': "Load failed after timeout", 'actual': 'error response 500'}
    ]

    add_step_results_example_1(test_run_id, case_id, steps_data, testrail_url, testrail_user, testrail_key)
```

Notice the `status_id` in the `data` dictionary. This applies to the overall test case result, whereas each step has its own status. In this example, the individual step 3 has failed, but we are passing the whole test, this means the test case in TestRail will be passed, but you can navigate inside that result to see step 3 has failed.

This was simple and demonstrated basic functionality. The real world, however, is usually more complex. Often, test frameworks provide the test execution results in a format not directly compatible with TestRail. Thus, we need to translate. I encountered this specifically when working with JUnit test reports; the xml structure requires parsing and restructuring before it can be fed into the TestRail API.

Let’s look at another scenario where we obtain step results from a test framework’s output. Let's imagine you have a set of test results in a list. This list is a simplified version of what you could see from a framework like pytest.

```python
import requests
import json

def add_step_results_example_2(test_run_id, case_id, test_output, testrail_url, testrail_user, testrail_key):
    """Adds step results based on test output from a (pretend) test framework."""
    url = f"{testrail_url}/index.php?/api/v2/add_result_for_case/{case_id}/{test_run_id}"
    headers = {'Content-Type': 'application/json'}
    auth = (testrail_user, testrail_key)


    steps_data = []
    for step in test_output:
        status_id = 1 if step['status'] == 'passed' else 4 if step['status'] == 'failed' else 2 #blocked
        step_data = {'content': step['description'], 'status_id': status_id}
        if 'actual' in step:
             step_data['actual'] = step['actual']
        if 'comment' in step:
             step_data['comment'] = step['comment']
        steps_data.append(step_data)

    data = {
        'steps': steps_data,
        'comment': 'Automated step results based on framework output',
        'status_id': 1 if all(step['status'] == 'passed' for step in test_output) else 4 # overall status
    }

    try:
      response = requests.post(url, headers=headers, auth=auth, data=json.dumps(data))
      response.raise_for_status()
      print(f"Successfully added results for case {case_id}, run {test_run_id}.")
    except requests.exceptions.RequestException as e:
        print(f"Error adding results: {e}")

if __name__ == '__main__':
    # replace with your actual data
    test_run_id = 123
    case_id = 456
    testrail_url = "https://your-testrail-instance.testrail.io"
    testrail_user = "your_username"
    testrail_key = "your_api_key"

    test_output = [
        {'description': "Check page title", 'status': 'passed'},
        {'description': "Verify element visibility", 'status': 'failed', 'actual':'element not visible', 'comment': 'element missing from the DOM'},
        {'description': 'Validate form submission', 'status':'passed'}
    ]

    add_step_results_example_2(test_run_id, case_id, test_output, testrail_url, testrail_user, testrail_key)

```
Here, I have created a dictionary named ‘test\_output’ with the output of an execution of a test framework. The logic in the function `add_step_results_example_2` will convert this output format into the format needed by TestRail. The important thing to note here is that I am also setting the overall test status, based on whether all the steps passed, or at least one failed. It’s crucial that you map the status values correctly. Inconsistent mappings will lead to inaccurate TestRail reports.

Finally, let's consider a more involved scenario. Imagine your framework generates artifacts—screenshots, log files, etc.—related to individual steps. TestRail allows you to attach these artifacts, albeit via a separate API call. To attach them to steps, it's easiest if you associate them with each step via a custom field. You will need to create a custom field in TestRail of type ‘URL’, and then, when adding the results, you can set the URL of the artifact in that custom field for the corresponding step. Here is a basic version of that process without the full artifact upload (because that would complicate the snippet).

```python
import requests
import json

def add_step_results_example_3(test_run_id, case_id, steps_data, testrail_url, testrail_user, testrail_key):
    """Adds step results to TestRail, including artifact links in custom fields."""
    url = f"{testrail_url}/index.php?/api/v2/add_result_for_case/{case_id}/{test_run_id}"
    headers = {'Content-Type': 'application/json'}
    auth = (testrail_user, testrail_key)


    # Assume custom field id is 12, modify accordingly.
    # custom_field_key = "custom_step_artifact_url"  # Replace with your actual field name or identifier.

    for step in steps_data:
      if 'artifact_url' in step:
        step['custom_step_artifact_url'] = step['artifact_url']
        del step['artifact_url']

    data = {
        'steps': steps_data,
        'comment': 'Step results with artifact links',
        'status_id': 1 #overall status, assuming passed for brevity.
    }

    try:
        response = requests.post(url, headers=headers, auth=auth, data=json.dumps(data))
        response.raise_for_status()
        print(f"Successfully added results for case {case_id}, run {test_run_id}.")
    except requests.exceptions.RequestException as e:
        print(f"Error adding results: {e}")

if __name__ == '__main__':
    # replace with your actual data
    test_run_id = 123
    case_id = 456
    testrail_url = "https://your-testrail-instance.testrail.io"
    testrail_user = "your_username"
    testrail_key = "your_api_key"


    steps_data = [
        {'content': "Step 1: User opens the application", 'status_id': 1},
        {'content': "Step 2: User attempts to log in with invalid credentials", 'status_id': 4,
         'artifact_url': 'https://some-server/artifact1.png'},
        {'content': "Step 3: User sees the error message", 'status_id': 1}
    ]

    add_step_results_example_3(test_run_id, case_id, steps_data, testrail_url, testrail_user, testrail_key)
```
Here I am assuming that there is a custom field in TestRail called `custom_step_artifact_url`. I am then using this field to specify the artifact URL inside each step. Remember to create this field in TestRail before running the code.
For a deeper understanding of the TestRail API, I recommend consulting the official TestRail documentation; it’s quite comprehensive and keeps pace with updates. The “Test Automation using TestRail’s API” section is particularly helpful. Also, keep an eye on the `requests` library documentation (if you are using python). They give valuable information regarding the usage and troubleshooting of API calls. You can also delve into “Restful Web Services” by Leonard Richardson and Sam Ruby to solidify your understanding of the underlying web concepts. Good luck, and remember, meticulous planning and clear data structures are key to success in this arena.
