---
title: "How can I integrate TestRail with an Azure DevOps Pipeline?"
date: "2024-12-16"
id: "how-can-i-integrate-testrail-with-an-azure-devops-pipeline"
---

Alright, let’s tackle this. I've seen this integration challenge pop up more often than I care to admit, and it’s usually due to the various moving parts involved in both systems. Integrating TestRail with Azure DevOps pipelines isn't straightforward "out of the box," but it's definitely manageable with the correct approach. It often boils down to leveraging each tool's api and creating a bridge between them within your pipeline. I will outline a method I’ve successfully used a few times, and provide specific examples to make sure it's applicable to your needs.

The core concept is to utilize the TestRail API, primarily via http requests, within an Azure DevOps pipeline. We’ll orchestrate this using scripting, which could be powershell (preferred on Windows) or bash (common in linux environments) depending on the pipeline agent your using. This script will usually perform a few crucial actions: create a test run, update test results, and potentially mark the run as completed in TestRail once the pipeline completes. The key is to understand that your pipeline executes your test suite and is the driver and TestRail is there to record your test runs, execution and provide an interface for the results. So the pipeline has to “tell” TestRail what’s happening.

First, we need authentication. TestRail uses API keys or username/password combinations for authentication. You need to create an API key within TestRail, which is usually safer than using a username and password. In Azure DevOps, these keys should be stored as a secure variable within your pipeline variables, not directly in the code. It's crucial for maintaining security.

Let’s dive into the process using powershell. This snippet shows how to create a test run. This is usually the first step in your integration.

```powershell
$testRailUrl = "https://your-testrail-instance.testrail.io"
$testRailUser = "your_user_email"
$testRailApiKey = "your_testrail_api_key"
$projectId = 1 # Replace with your TestRail project id
$suiteId = 1 # Replace with your TestRail test suite id
$runName = "Automated Run $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
$contentType = "application/json"

$authInfo = "$($testRailUser):$($testRailApiKey)"
$authHeader = @{ Authorization = "Basic $($authInfo | ConvertTo-Base64)" }

$body = @{
    suite_id = $suiteId
    name = $runName
    include_all = $true # or set case_ids
} | ConvertTo-Json

try {
    $response = Invoke-RestMethod -Uri "$testRailUrl/index.php?/api/v2/add_run/$projectId" -Method Post -Headers $authHeader -Body $body -ContentType $contentType
    Write-Host "TestRail run created successfully: $($response.id)"
    $testRunId = $response.id #Store the run ID for later updates
}
catch {
    Write-Host "Error creating test run: $_"
    throw
}

```

This powershell script, first, prepares all necessary credentials like the API key, username, and TestRail base url. The script authenticates with the TestRail api by creating the necessary header and then creates the request body as JSON. We convert to json, to ensure the api can consume the data. It then uses `Invoke-RestMethod` to make a POST request to create a test run using TestRail’s api. After success, it extracts and prints the test run ID, which you’ll need later. Critically, if an error occurs during the call, the exception is caught and thrown, which will fail your pipeline task.

Now, let's assume your pipeline runs your tests, and each test execution produces a result. You need to map your tests to corresponding test cases in TestRail. This might involve using test case ids, or a naming convention in your tests that links back to your testrail test case. Here is how to update a result with powershell after you’ve run a test. I’ll assume you have a variable, `$testCaseId`, indicating which TestRail case id the result is for, and `$testResult` which indicates whether the test passed or failed:

```powershell
$testRailUrl = "https://your-testrail-instance.testrail.io"
$testRailUser = "your_user_email"
$testRailApiKey = "your_testrail_api_key"
$testRunId = 123 # Replace with the TestRail run id from the first step
$testCaseId = 456 # Replace with a real TestRail case id
$testResult = "passed" # This should come from your test execution

$statusId = if ($testResult -eq "passed") { 1 } else { 5 } # 1 for passed, 5 for failed
$contentType = "application/json"

$authInfo = "$($testRailUser):$($testRailApiKey)"
$authHeader = @{ Authorization = "Basic $($authInfo | ConvertTo-Base64)" }

$body = @{
    status_id = $statusId
} | ConvertTo-Json

try {
    $response = Invoke-RestMethod -Uri "$testRailUrl/index.php?/api/v2/add_result_for_case/$testRunId/$testCaseId" -Method Post -Headers $authHeader -Body $body -ContentType $contentType
    Write-Host "TestRail result added for case $($testCaseId) successfully."
}
catch {
    Write-Host "Error updating TestRail result for case $($testCaseId): $_"
    throw
}
```

This snippet is similar to the previous one. We establish auth and prepare for an API call. We now use a conditional statement to map results to TestRail's status codes (1 being passed and 5 being failed). Then, a POST request to `/add_result_for_case` updates the specific test case's status within the specified test run.

Finally, after all tests are executed and results have been uploaded to TestRail, you might want to close or mark the test run as completed. This keeps TestRail organized. Here’s another quick example of how to achieve that:

```powershell
$testRailUrl = "https://your-testrail-instance.testrail.io"
$testRailUser = "your_user_email"
$testRailApiKey = "your_testrail_api_key"
$testRunId = 123  # Replace with your actual TestRail run id
$contentType = "application/json"


$authInfo = "$($testRailUser):$($testRailApiKey)"
$authHeader = @{ Authorization = "Basic $($authInfo | ConvertTo-Base64)" }

try {
    $response = Invoke-RestMethod -Uri "$testRailUrl/index.php?/api/v2/close_run/$testRunId" -Method Post -Headers $authHeader -ContentType $contentType
    Write-Host "TestRail run $($testRunId) closed successfully."
}
catch {
    Write-Host "Error closing TestRail run $($testRunId): $_"
    throw
}
```

This powershell script initiates a POST request to TestRail's `/close_run` endpoint. The main point to note is this final script is very basic and just closes the run. In practice, you might want to incorporate retry logic, or better error handling.

For implementation details, I highly recommend referring to TestRail’s official documentation. They maintain a comprehensive api guide which is crucial for getting all the details correct. Also the ‘Test Automation and Reporting Patterns’ by James Bach would be useful for understanding how to organize your tests and their results.

Remember, the examples here are basic and need further refinement. Error handling, retry logic, logging, and robust mapping between test IDs are vital for a production-ready setup. You’ll need to encapsulate this within Azure DevOps pipeline tasks, typically using powershell or bash tasks depending on the agent. Store your api keys as pipeline variables and use the secure variable mechanism in your Azure DevOps pipelines. I also recommend using something like the ‘Azure Key Vault’ to manage the api keys securely.
