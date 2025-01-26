---
title: "How can I filter S3 get_object CloudTrail logs?"
date: "2025-01-26"
id: "how-can-i-filter-s3-getobject-cloudtrail-logs"
---

CloudTrail logs for `s3:GetObject` events often become voluminous, and efficient filtering is crucial for security analysis and operational troubleshooting. I've spent considerable time parsing these logs for incident response, and a straightforward grep approach quickly becomes impractical. The logs, being JSON objects, necessitate a more nuanced approach, typically leveraging tools designed for JSON processing or log aggregation systems.

The fundamental challenge arises from the nested structure of the CloudTrail logs. Each log entry, representing a single event, contains various fields. The event details relevant for filtering, like the S3 bucket name, key, or user identity, are often deeply embedded within these JSON structures. Direct string matching is error-prone due to variations in formatting and potential extraneous data. Effective filtering requires extracting specific values based on their key path within the JSON.

My preferred method involves utilizing a combination of Python with the `jq` command-line utility, which excels at processing JSON data. While Python can handle the entire process, `jq` provides significant performance gains for complex filtering tasks, especially when dealing with large datasets. Python facilitates the orchestration of `jq` and provides a more programmatic way to iterate through multiple log files, or even directly ingest logs from S3. I generally use Python's `subprocess` module to interact with `jq`.

Here's a typical workflow: First, I use Python to locate the relevant CloudTrail log files (either locally or by downloading them from an S3 bucket), which might be specified via command line argument or a configuration file. Then, for each log file, Python invokes `jq` using `subprocess.Popen`, providing the filtering criteria as a command-line argument. Finally, Python handles the filtered output, writing it to a file, database, or console as needed. This combination provides a balance between the expressive power of Python and the efficiency of `jq`.

Let's illustrate with a few practical examples:

**Example 1: Filtering for specific S3 bucket and key.**

Suppose I want to find all `s3:GetObject` events related to a specific bucket named `my-sensitive-data` and key `confidential.txt`. The `jq` filter would target the JSON path leading to these values. This filter works by traversing through the deeply nested JSON structure and ensuring that the expected values match using the `select` operator. The `| .` extracts the entire JSON object that matches these criteria for further output.

```python
import subprocess
import json

def filter_s3_get_object(log_file, bucket_name, key_name):
    jq_filter = f'.Records[] | select(.eventName=="GetObject" and .eventSource=="s3.amazonaws.com" and .requestParameters.bucketName == "{bucket_name}" and .requestParameters.key == "{key_name}")'
    try:
        process = subprocess.Popen(
            ['jq', jq_filter, log_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate()
        if stderr:
            print(f"Error during jq execution: {stderr}")
            return None
        filtered_logs = [json.loads(line) for line in stdout.strip().split('\n') if line]
        return filtered_logs

    except FileNotFoundError:
        print("Error: jq command not found. Ensure it's installed and in your PATH.")
        return None


if __name__ == "__main__":
    log_file = 'cloudtrail-log-example.json'  # Replace with your log file
    bucket_name = 'my-sensitive-data'
    key_name = 'confidential.txt'
    filtered_results = filter_s3_get_object(log_file, bucket_name, key_name)

    if filtered_results:
        for log in filtered_results:
          print(json.dumps(log, indent=2))

```

In this Python code:
*   `jq_filter` dynamically creates the command line argument for `jq` to use for filtering.
*   `subprocess.Popen` executes the `jq` command with the filter and the log file.
*   The output is captured and split into individual JSON objects.
*  It also handles the `FileNotFoundError` exception if `jq` is not present.
*  The main block uses example `log_file`, `bucket_name`, and `key_name` for the filter.

**Example 2: Filtering by User Identity**

Another common use case is filtering logs based on the identity of the user who made the request. CloudTrail captures the user information, which can be used for access auditing.  Here, I'm filtering all `s3:GetObject` events where the `userIdentity.userName` field matches `my-iam-user`.

```python
import subprocess
import json

def filter_s3_get_object_by_user(log_file, user_name):
    jq_filter = f'.Records[] | select(.eventName=="GetObject" and .eventSource=="s3.amazonaws.com" and .userIdentity.userName == "{user_name}")'
    try:
        process = subprocess.Popen(
            ['jq', jq_filter, log_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate()
        if stderr:
            print(f"Error during jq execution: {stderr}")
            return None
        filtered_logs = [json.loads(line) for line in stdout.strip().split('\n') if line]
        return filtered_logs

    except FileNotFoundError:
        print("Error: jq command not found. Ensure it's installed and in your PATH.")
        return None

if __name__ == "__main__":
    log_file = 'cloudtrail-log-example.json'  # Replace with your log file
    user_name = 'my-iam-user'
    filtered_results = filter_s3_get_object_by_user(log_file, user_name)

    if filtered_results:
        for log in filtered_results:
          print(json.dumps(log, indent=2))
```
This example is similar to the first but uses `userIdentity.userName` in the `jq` filter instead of specific bucket and key information. The filter is modified to query user identity instead of bucket name and key parameters

**Example 3: Filtering using a wildcard or regular expression.**

Filtering based on exact matches might be too restrictive. Sometimes I need to filter based on a wildcard or pattern.  For example, I might need to find all access to objects whose keys start with 'backup-'. While `jq` itself doesn't have comprehensive regular expression support, it can integrate with other tools, using the `-r` argument to `jq` to output raw text and pipe that to grep. This allows complex filtering, as `grep` can provide regex capabilities. I'll need to tweak the Python code to facilitate this pipeline. Here I'm using `grep` to find any log with a key that starts with `backup-`, leveraging a regular expression. The `jq` query will extract the key for `grep`.

```python
import subprocess
import json

def filter_s3_get_object_by_key_pattern(log_file, key_pattern):
    jq_extract = '.Records[] | select(.eventName=="GetObject" and .eventSource=="s3.amazonaws.com") | .requestParameters.key'

    try:
        jq_process = subprocess.Popen(['jq', '-r', jq_extract, log_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        grep_process = subprocess.Popen(['grep', f'^{key_pattern}'], stdin=jq_process.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        jq_process.stdout.close() # Allow jq process to receive a SIGPIPE

        grep_stdout, grep_stderr = grep_process.communicate()

        if grep_stderr:
            print(f"Error during grep execution: {grep_stderr}")
            return None

        filtered_keys = [line for line in grep_stdout.strip().split('\n') if line]

        jq_filter_final = f'.Records[] | select(.eventName=="GetObject" and .eventSource=="s3.amazonaws.com" and (.requestParameters.key | IN({filtered_keys})) )'

        final_process = subprocess.Popen(
            ['jq', jq_filter_final, log_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = final_process.communicate()

        if stderr:
          print(f"Error during second jq execution: {stderr}")
          return None
        filtered_logs = [json.loads(line) for line in stdout.strip().split('\n') if line]

        return filtered_logs


    except FileNotFoundError:
         print("Error: jq or grep command not found. Ensure they are installed and in your PATH.")
         return None

if __name__ == "__main__":
    log_file = 'cloudtrail-log-example.json'  # Replace with your log file
    key_pattern = 'backup-' # Find any key that starts with backup-
    filtered_results = filter_s3_get_object_by_key_pattern(log_file, key_pattern)

    if filtered_results:
        for log in filtered_results:
          print(json.dumps(log, indent=2))

```
Here:
*   The `jq_extract` query will extract only the `requestParameters.key`.
*   The output is piped to `grep` to filter by the given pattern (`^backup-` specifies keys starting with `backup-`).
*   The grep output is then used to formulate the final filter to get the whole records that contain the matched key, by using `IN` operator with the results from grep.

For anyone doing serious work with CloudTrail logs, I suggest looking into several areas. Familiarize yourself with the CloudTrail documentation, especially the event record structure. For efficiency in log processing, investigate serverless processing using Lambda functions. The official documentation offers tutorials on how to use serverless tools like AWS Athena to query CloudTrail logs, that might be beneficial if the volume of the log data makes it unfeasible to use these methods. Also familiarize yourself with the details of JSONPath syntax and jq operators and functions through their respective documentation. Using a log aggregation system such as Splunk or the AWS managed Elasticsearch service can provide further enhancements to log storage and filtering capabilities. Finally, for more advanced processing, Python’s `boto3` library can assist in directly downloading, parsing and further processing large CloudTrail log files from S3 and this might alleviate the need to store the log files locally.
