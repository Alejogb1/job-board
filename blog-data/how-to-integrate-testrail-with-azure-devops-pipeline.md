---
title: "How to integrate TestRail with Azure DevOps Pipeline?"
date: "2024-12-16"
id: "how-to-integrate-testrail-with-azure-devops-pipeline"
---

Let's tackle this integration head-on. I've been through a similar scenario a few times, and it's always a matter of connecting the right pieces logically. The core issue is automating the update of test results in TestRail from an Azure DevOps pipeline—a fairly common requirement. What we need is a mechanism to: (1) trigger tests within the pipeline, (2) parse the results in a format TestRail understands, and (3) transmit those results back to TestRail. It's less about a 'single-click' solution and more about orchestrating several steps.

Firstly, understand that TestRail doesn't inherently 'listen' to Azure DevOps. You need an intermediary, most often a custom script or a small application that bridges the gap. In my past experiences, I’ve found that using python, due to its strong ecosystem of libraries, to be quite flexible. We'll need to leverage TestRail's API and a structured approach to our automated test execution within Azure. For test execution itself, let’s assume we have a set of automated tests that produce output in a standard format (e.g., JUnit XML).

The essential stages can be summarized as:

1.  **Test Execution:** Within your Azure DevOps pipeline, you'll execute your automated tests, which should produce a parsable output.
2.  **Result Parsing and Formatting:** This is where the python script comes in. We'll read the test output and format the data to align with TestRail's requirements. This typically involves parsing results (passed, failed, blocked) and test case ids.
3.  **TestRail API Interaction:** Using the TestRail API, we’ll then send the formatted results, creating or updating test runs and marking individual test cases accordingly.

Now, let's dive into the code. I’ll provide three example snippets illustrating each stage. We’ll assume your tests are JUnit-formatted for the sake of demonstration, but the parsing logic could adapt to other formats with necessary changes.

**Snippet 1: Test Execution Stage (YAML in Azure Pipeline)**

This snippet outlines the basic structure for running tests and capturing the output in the pipeline. We'll be using the `dotnet test` command in this example, assuming that the test project is a .NET based one. Adapt this based on your project.

```yaml
steps:
  - task: DotNetCoreCLI@2
    displayName: 'Run Unit Tests'
    inputs:
      command: 'test'
      projects: '**/*Tests.csproj' # adjust based on your project structure
      arguments: '--logger:"junit;LogFilePath=test-results.xml"'
  - task: PublishPipelineArtifact@1
    displayName: 'Publish Test Results Artifact'
    inputs:
        targetPath: 'test-results.xml'
        artifactName: 'testResults'

```

Here, the `DotNetCoreCLI` task executes tests and generates a `test-results.xml` file in JUnit format. `PublishPipelineArtifact` makes this result file available to subsequent stages. If your tests are in another language, you'd adjust the commands accordingly. For example, you could be running `pytest` with an appropriate JUnit output option.

**Snippet 2: Python Script for Result Parsing and Formatting**

This script uses python and `xml.etree.ElementTree` for parsing the XML, alongside the TestRail API library (`testrail-api-py`). For this snippet, you’ll need to have the following libraries installed with a `pip install testrail-api-py xmltodict`. This script can be incorporated into an Azure DevOps pipeline as a command line task executing a python script.

```python
import xml.etree.ElementTree as ET
import testrail
import os
import json

def parse_junit_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    results = {}
    for testsuite in root.findall('testsuite'):
        for testcase in testsuite.findall('testcase'):
            test_name = testcase.get('name')
            if 'failure' in [child.tag for child in testcase]:
                results[test_name] = 5 # TestRail status code for failed
            elif 'skipped' in [child.tag for child in testcase]:
                results[test_name] = 4 # TestRail status code for skipped
            else:
                results[test_name] = 1 # TestRail status code for passed
    return results

def update_testrail(test_results, testrail_url, testrail_user, testrail_password, project_id, run_id):
    client = testrail.APIClient(testrail_url)
    client.user = testrail_user
    client.password = testrail_password
    case_status_updates = []

    for test_name, status_code in test_results.items():
        case_id = get_case_id_from_name(test_name)
        if case_id:
            case_status_updates.append({
                'case_id': case_id,
                'status_id': status_code
            })

    if case_status_updates:
        client.send_post(f'add_results_for_cases/{run_id}',{'results': case_status_updates})
        print(f"Results successfully uploaded for test run: {run_id}")
    else:
        print("No results to update TestRail")

def get_case_id_from_name(test_name):
    #This would need to be based on a mapping between test name and testrail case ID
    #This method below provides a simple hard-coded example and should be adapted
     case_mapping = {
         'Test_Method1': 1234,
         'Test_Method2': 1235,
         'Test_Method3': 1236
     }
     return case_mapping.get(test_name)

if __name__ == "__main__":
   test_results_file = "test-results.xml" # This would come from the previous stage by reading pipeline artifact
   test_results_parsed = parse_junit_xml(test_results_file)

   # Configure TestRail parameters
   testrail_url = os.environ['TESTRAIL_URL']
   testrail_user = os.environ['TESTRAIL_USER']
   testrail_password = os.environ['TESTRAIL_PASSWORD']
   project_id = int(os.environ['TESTRAIL_PROJECT_ID'])
   run_id = int(os.environ['TESTRAIL_RUN_ID'])


   update_testrail(test_results_parsed,testrail_url, testrail_user, testrail_password, project_id, run_id)
```

The `parse_junit_xml` function extracts results, mapping pass/fail to TestRail status codes. `update_testrail` takes the parsed results and updates TestRail with the respective status updates using the TestRail API. The `get_case_id_from_name` shows a basic example of mapping test names to their corresponding TestRail IDs. This is the most critical part – how you map test names to TestRail case IDs will heavily depend on your specific setup. I generally prefer a separate configuration file for this mapping, but a database or lookup table can also be useful. Consider environmental variables to maintain security.

**Snippet 3: Azure Pipeline Task Calling Python Script**

Finally, here's an Azure DevOps pipeline snippet illustrating the task using the python script.

```yaml
steps:
    - task: DownloadPipelineArtifact@2
      displayName: 'Download Test Results'
      inputs:
        artifactName: 'testResults'
        targetPath: $(System.DefaultWorkingDirectory)

    - task: PythonScript@0
      displayName: 'Update TestRail with Results'
      inputs:
        scriptSource: 'filePath'
        scriptPath: 'path/to/your/script.py'  # Update to your path
        arguments:
        pythonInterpreter: 'path/to/your/python'  # Update to your python installation
      env:
        TESTRAIL_URL: $(testrail_url)
        TESTRAIL_USER: $(testrail_user)
        TESTRAIL_PASSWORD: $(testrail_password)
        TESTRAIL_PROJECT_ID: $(testrail_project_id)
        TESTRAIL_RUN_ID: $(testrail_run_id)
```

This step downloads the previously uploaded artifact which contains the test results, then executes the python script. Note the usage of environmental variables which have been set in the pipeline variables for the TestRail authentication. This prevents the password from being hard coded into the script. These variables would need to be set in the pipeline. The python script is then called with the relevant parameters.

**Important Considerations:**

*   **Error Handling:** The above examples are simplified. Real-world implementations require robust error handling, logging, and retry mechanisms.
*   **Test Case Mapping:** The crux of the integration is accurately mapping test cases between your code and TestRail. Develop a robust method that works for your team. The example provides a very basic mapping, however this should be adapted with a lookup from a configuration file, database, or similar mechanism.
*   **TestRail API Client:** The `testrail-api-py` library used is one option; consider other alternatives if necessary.
*   **Test Run Creation:** You’ll likely need a mechanism to create test runs in TestRail before updating results. This can also be automated via the API (not shown in this example for brevity) and could be integrated as an initial step in your pipeline.
*   **Security:** Store your TestRail API credentials securely, using secrets management within Azure DevOps.

For further reading, I'd recommend:

1.  The official TestRail API documentation is a must-read for understanding endpoints and data formats.
2.  Refer to "Automated Software Testing" by Elfriede Dustin for a comprehensive view of testing practices, including integration with tools such as TestRail.
3.  "Effective DevOps" by Jennifer Davis and Ryn Daniels provides context around integrating automated testing within a CI/CD pipeline.
4. Look into the `testrail-api-py` library (or alternative) documentation for specific details related to its usage.

Integrating TestRail with Azure DevOps pipelines is not a plug-and-play operation; it requires careful planning and some custom scripting. However, the process is manageable when broken down into these key components. Remember that it is a process of continuous improvement, and this structure should form a solid basis for building a robust automated testing workflow.
