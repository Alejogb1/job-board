---
title: "What are the details in S3 logs?"
date: "2025-01-30"
id: "what-are-the-details-in-s3-logs"
---
S3 server access logs provide a detailed, chronological record of requests made to Amazon Simple Storage Service (S3) buckets. These logs are crucial for auditing, security analysis, and performance monitoring, capturing a wealth of information about who is accessing your data, how they are doing it, and from where. I’ve spent considerable time analyzing these logs while optimizing a large-scale image processing pipeline, and understanding their structure has been fundamental to identifying bottlenecks and security vulnerabilities.

An S3 access log entry is formatted as a space-separated string, where each element corresponds to a particular aspect of the request. The standard log format contains approximately twenty distinct fields; however, the exact number can vary depending on the logging configuration and the type of request. Let's break down the most pertinent fields.

First, the **bucket owner** field indicates the canonical user ID of the bucket owner. This helps in scenarios where you have multiple AWS accounts or buckets, allowing for precise identification of the target bucket. Following this is the **bucket name**, the specific S3 bucket receiving the request. This combination is foundational in isolating which bucket is being accessed and by whom. Next we have the **time** at which the request occurred. This is typically recorded in Coordinated Universal Time (UTC) and provides a crucial timeline for correlating events.

The **remote IP address** identifies the source IP of the client making the request. This is vital for tracking the origin of requests, enabling you to identify legitimate traffic, potential bot activity, or suspicious user behavior. This can be followed by a **requester ID**. This represents the AWS canonical user ID of the requester, not always present as some access is done anonymously. If present, this field along with the remote IP allows for robust source analysis. The next few fields relate directly to the operation requested.

The **request ID** is a unique identifier assigned to each request by S3. This serves as a crucial reference point, especially when debugging issues with AWS support or tracing specific errors. Closely associated is the **operation** field, indicating the type of action requested by the client, such as GET, PUT, DELETE, HEAD, POST, or a more specific S3 operation. Understanding what operation was performed against a certain resource is key for security and debugging purposes. Following this, the **key** field refers to the specific object (file) being accessed or modified within the bucket. For requests that do not target a specific object, this field might be ‘-’. The **request URI** provides the full resource path as seen by S3, while the **HTTP status code** returns the result of the request, providing valuable insight into the success or failure of the operation. For example, a 200 indicates success, a 404 means “not found”, and 5xx codes typically indicate server-side errors.

Finally, information on **bytes sent** and **bytes received** indicates the amount of data involved in the operation, and is critical in assessing bandwidth consumption and costs. In addition, there’s an optional field for **time to first byte** which is a measure of response time from when a request is received until the first byte of the response is sent back to the client, and a **total time** field which is the total time taken for that request to finish. The **referrer** and **user agent** fields provide additional information about the client, particularly browser requests. There are also further fields regarding the version and HTTP method. Note that while not always present, these additional fields can be critical for nuanced analysis.

Now let’s delve into some code examples. For these examples, I will be using Python. These code snippets are focused on parsing and processing simulated log data, which represents the typical format of S3 access logs.

**Example 1: Basic Log Parsing**

```python
import re

log_line = "79a59df900b949e55d96a1e698fbacedfd6e09d98eacf8f8d5218e7cd47ef2be my-bucket [21/Apr/2023:22:30:58 +0000] 192.168.1.1 - 79a59df900b949e55d96a1e698fbacedfd6e09d98eacf8f8d5218e7cd47ef2be GET /images/photo1.jpg HTTP/1.1 200 - 1234 567 243 23 - -"
log_pattern = re.compile(r'([^ ]*) ([^ ]*) \[([^\]]*)\] ([^ ]*) ([^ ]*) ([^ ]*) ([^ ]*) ([^ ]*) ([^ ]*) ([^ ]*) ([^ ]*) ([^ ]*) ([^ ]*) ([^ ]*) ([^ ]*)')

match = log_pattern.match(log_line)
if match:
    log_data = match.groups()
    print("Bucket Owner:", log_data[0])
    print("Bucket Name:", log_data[1])
    print("Time:", log_data[2])
    print("Remote IP:", log_data[3])
    print("Request Type:", log_data[6])
    print("Key:", log_data[7])
    print("HTTP Code:", log_data[9])
    print("Bytes Sent:", log_data[10])
    print("Bytes Received:", log_data[11])
    print("Time to First Byte:", log_data[12])
    print("Total Time:", log_data[13])
```

This code snippet utilizes regular expressions to extract key information from a sample log line. The `log_pattern` is designed to capture the space-separated fields. The `match.groups()` then neatly presents the different log fields for analysis. The output clearly illustrates the basic information extractable from a log entry.

**Example 2: Filtering logs based on HTTP status codes.**

```python
import re

log_lines = [
    "79a59df900b949e55d96a1e698fbacedfd6e09d98eacf8f8d5218e7cd47ef2be my-bucket [21/Apr/2023:22:30:58 +0000] 192.168.1.1 - 79a59df900b949e55d96a1e698fbacedfd6e09d98eacf8f8d5218e7cd47ef2be GET /images/photo1.jpg HTTP/1.1 200 - 1234 567 243 23 - -",
    "79a59df900b949e55d96a1e698fbacedfd6e09d98eacf8f8d5218e7cd47ef2be my-bucket [21/Apr/2023:22:31:01 +0000] 192.168.1.2 - 79a59df900b949e55d96a1e698fbacedfd6e09d98eacf8f8d5218e7cd47ef2be PUT /data/report.json HTTP/1.1 201 - 5432 0 120 21 - -",
    "79a59df900b949e55d96a1e698fbacedfd6e09d98eacf8f8d5218e7cd47ef2be my-bucket [21/Apr/2023:22:31:05 +0000] 192.168.1.3 - - GET /logs/app.log HTTP/1.1 404 - 0 0 200 15 - -",
    "79a59df900b949e55d96a1e698fbacedfd6e09d98eacf8f8d5218e7cd47ef2be my-bucket [21/Apr/2023:22:31:08 +0000] 192.168.1.4 - 79a59df900b949e55d96a1e698fbacedfd6e09d98eacf8f8d5218e7cd47ef2be DELETE /data/archive.zip HTTP/1.1 204 - 0 0 10 1 - -"
]

log_pattern = re.compile(r'([^ ]*) ([^ ]*) \[([^\]]*)\] ([^ ]*) ([^ ]*) ([^ ]*) ([^ ]*) ([^ ]*) ([^ ]*) ([^ ]*) ([^ ]*) ([^ ]*) ([^ ]*) ([^ ]*) ([^ ]*)')

def analyze_log_status_codes(log_lines):
  error_lines = []
  for line in log_lines:
      match = log_pattern.match(line)
      if match:
          log_data = match.groups()
          status_code = log_data[9]
          if status_code != '200' and status_code != '201' and status_code != '204':
              error_lines.append(line)
  return error_lines

errors = analyze_log_status_codes(log_lines)
for error in errors:
    print("Error:", error)
```

This example showcases how to filter S3 logs based on the HTTP status codes, specifically by collecting the error logs (status codes other than 200, 201, or 204). The `analyze_log_status_codes` function iterates over several log entries, filters them based on the HTTP response code, and returns a list of error log lines. This filtering process is instrumental in isolating issues, such as failed requests or access denials.

**Example 3: Aggregate request count by source IP.**

```python
import re
from collections import defaultdict

log_lines = [
    "79a59df900b949e55d96a1e698fbacedfd6e09d98eacf8f8d5218e7cd47ef2be my-bucket [21/Apr/2023:22:30:58 +0000] 192.168.1.1 - 79a59df900b949e55d96a1e698fbacedfd6e09d98eacf8f8d5218e7cd47ef2be GET /images/photo1.jpg HTTP/1.1 200 - 1234 567 243 23 - -",
    "79a59df900b949e55d96a1e698fbacedfd6e09d98eacf8f8d5218e7cd47ef2be my-bucket [21/Apr/2023:22:31:01 +0000] 192.168.1.2 - 79a59df900b949e55d96a1e698fbacedfd6e09d98eacf8f8d5218e7cd47ef2be PUT /data/report.json HTTP/1.1 201 - 5432 0 120 21 - -",
    "79a59df900b949e55d96a1e698fbacedfd6e09d98eacf8f8d5218e7cd47ef2be my-bucket [21/Apr/2023:22:31:05 +0000] 192.168.1.1 - - GET /logs/app.log HTTP/1.1 404 - 0 0 200 15 - -",
    "79a59df900b949e55d96a1e698fbacedfd6e09d98eacf8f8d5218e7cd47ef2be my-bucket [21/Apr/2023:22:31:08 +0000] 192.168.1.3 - 79a59df900b949e55d96a1e698fbacedfd6e09d98eacf8f8d5218e7cd47ef2be DELETE /data/archive.zip HTTP/1.1 204 - 0 0 10 1 - -",
    "79a59df900b949e55d96a1e698fbacedfd6e09d98eacf8f8d5218e7cd47ef2be my-bucket [21/Apr/2023:22:31:11 +0000] 192.168.1.1 - - GET /files/largefile.dat HTTP/1.1 200 - 1000000 1000 200 15 - -",
    "79a59df900b949e55d96a1e698fbacedfd6e09d98eacf8f8d5218e7cd47ef2be my-bucket [21/Apr/2023:22:31:15 +0000] 192.168.1.4 - 79a59df900b949e55d96a1e698fbacedfd6e09d98eacf8f8d5218e7cd47ef2be PUT /data/results.json HTTP/1.1 201 - 3000 0 50 2 - -"
]


log_pattern = re.compile(r'([^ ]*) ([^ ]*) \[([^\]]*)\] ([^ ]*) ([^ ]*) ([^ ]*) ([^ ]*) ([^ ]*) ([^ ]*) ([^ ]*) ([^ ]*) ([^ ]*) ([^ ]*) ([^ ]*) ([^ ]*)')

def aggregate_requests_by_ip(log_lines):
    request_counts = defaultdict(int)
    for line in log_lines:
        match = log_pattern.match(line)
        if match:
            log_data = match.groups()
            ip_address = log_data[3]
            request_counts[ip_address] += 1
    return request_counts

request_counts = aggregate_requests_by_ip(log_lines)
for ip, count in request_counts.items():
    print(f"IP: {ip}, Request Count: {count}")
```

This example demonstrates aggregating the number of requests based on the source IP addresses using a dictionary. The `aggregate_requests_by_ip` function iterates through the log data, extracts the IP, and increments a counter for each IP, showcasing how the logs can be used to summarize request patterns from different clients. This is crucial for identifying potential high volume users, or unusual access patterns.

For further exploration of S3 logging and related tools, I recommend reviewing AWS documentation on server access logging, in addition to material on CloudTrail (for API call monitoring), and CloudWatch (for log analysis and visualization). Specifically, the documentation on best practices for S3 security and performance provides comprehensive information. Third party solutions such as Splunk, ELK stack and Datadog offer more sophisticated parsing and analysis features.
