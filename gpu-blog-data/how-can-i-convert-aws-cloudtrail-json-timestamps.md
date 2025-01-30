---
title: "How can I convert AWS CloudTrail JSON timestamps?"
date: "2025-01-30"
id: "how-can-i-convert-aws-cloudtrail-json-timestamps"
---
AWS CloudTrail logs events with timestamps represented in an ISO 8601-like string format, specifically using UTC. These strings, while technically parsable, often require conversion for human readability or integration with downstream systems that may expect timestamps in different formats or time zones. I've frequently encountered this scenario when analyzing log data for security incidents or performance debugging. Direct manipulation of these timestamp strings is often necessary, rather than relying solely on AWS provided tooling which might not always offer the precise output required.

The core problem lies in the nature of the timestamp representation itself. CloudTrail delivers timestamps as strings that generally resemble: `"2023-10-27T10:30:00Z"`. This indicates October 27th, 2023, at 10:30:00 UTC.  However, for reporting, storage in databases, or even just easier visual scanning, I've needed to convert this into more practical representations like local time or a Unix timestamp (seconds since the epoch). Doing so involves parsing the string, manipulating the datetime object, and then formatting it into a new string or numerical representation. There is no single "correct" way, it largely depends on the specific use case.

To convert these strings effectively, I primarily rely on standard library functions in scripting languages, particularly Python.  The process typically involves three key stages: parsing, manipulation, and formatting. The parsing stage involves transforming the string into a datetime object; the manipulation stage modifies the datetime object (e.g., changing the time zone); and finally, the formatting stage converts the object into the desired string representation or other data type (like a numeric timestamp). The specific tools to achieve this vary, but the underlying principles of parsing, modifying, and formatting remain constant.

Here's a simple Python example demonstrating this conversion process:

```python
import datetime
import pytz
import json

def convert_cloudtrail_timestamp(log_record, target_timezone="America/Los_Angeles"):
    """Converts CloudTrail timestamp to a specified timezone and formats it.

    Args:
        log_record (dict): A dictionary representing a single CloudTrail log record.
        target_timezone (str): The IANA timezone string for the desired timezone.

    Returns:
        dict: The log record with the converted timestamp, or the original log record if no 'eventTime' key is present.
    """

    if 'eventTime' not in log_record:
        return log_record

    timestamp_string = log_record['eventTime']
    # Parse the ISO 8601-like string into a datetime object.
    utc_datetime = datetime.datetime.fromisoformat(timestamp_string.replace("Z","+00:00")) #handle the Z timezone as +00:00

    # Convert to the target timezone.
    target_tz = pytz.timezone(target_timezone)
    target_datetime = utc_datetime.replace(tzinfo=pytz.utc).astimezone(target_tz)

    # Format the timestamp into a human-readable string
    formatted_timestamp = target_datetime.strftime("%Y-%m-%d %H:%M:%S %Z%z")
    log_record['formattedEventTime'] = formatted_timestamp # Add the formatted timestamp to the dict

    return log_record

# Example Usage:
cloudtrail_log = {
  "eventVersion": "1.08",
  "userIdentity": {
      "type": "IAMUser",
      "principalId": "EXAMPLE_USER_ID",
      "arn": "arn:aws:iam::EXAMPLE_ACCOUNT_ID:user/example_user",
      "accountId": "EXAMPLE_ACCOUNT_ID",
      "accessKeyId": "EXAMPLE_ACCESS_KEY",
      "userName": "example_user"
  },
    "eventTime": "2023-10-27T10:30:00Z",
  "eventSource": "iam.amazonaws.com",
  "eventName": "CreateUser",
  "awsRegion": "us-east-1",
    "sourceIpAddress": "192.0.2.0",
  "userAgent": "aws-cli/2.7.27 Python/3.9.11 Darwin/21.6.0 exe/x86_64 prompt/off command/iam.create-user",
  "errorCode": null,
  "errorMessage": null,
  "requestParameters": {
      "userName": "test-user"
  },
    "responseElements": {
    "user": {
        "path": "/",
        "userName": "test-user",
        "userId": "EXAMPLE_USER_ID_RESPONSE",
        "arn": "arn:aws:iam::EXAMPLE_ACCOUNT_ID:user/test-user",
        "createDate": "Oct 27, 2023 10:30:00 AM"
        }
  },
  "requestID": "EXAMPLE_REQUEST_ID",
    "eventID": "EXAMPLE_EVENT_ID",
    "eventType": "AwsApiCall",
  "recipientAccountId": "EXAMPLE_ACCOUNT_ID"
}

converted_log = convert_cloudtrail_timestamp(cloudtrail_log)
print(json.dumps(converted_log, indent=2))
```

In this code, the `convert_cloudtrail_timestamp` function takes a CloudTrail log record and a target timezone as input.  The `datetime.datetime.fromisoformat()` function is used to parse the timestamp string into a datetime object. The `.replace("Z","+00:00")` part handles the timezone identifier "Z" by converting it to "+00:00", as `fromisoformat` directly expects the offset, not "Z". Then `pytz` library converts the timestamp from UTC to the provided target timezone, then formats it using `strftime` to produce a user friendly string. The formatted timestamp is then added to the log record as the key `'formattedEventTime'`, before the record is returned. This makes the converted timestamp available in the log data alongside the original timestamp.

Next, consider the scenario where the goal is to convert the timestamp to a Unix timestamp (seconds since the epoch), often preferred for numerical computation and storage.

```python
import datetime
import time
import json

def convert_cloudtrail_to_unix(log_record):
    """Converts CloudTrail timestamp to a Unix timestamp (seconds since the epoch).

    Args:
        log_record (dict): A dictionary representing a single CloudTrail log record.

    Returns:
      dict: The log record with the converted timestamp added as a new key.
    """
    if 'eventTime' not in log_record:
        return log_record

    timestamp_string = log_record['eventTime']
    utc_datetime = datetime.datetime.fromisoformat(timestamp_string.replace("Z","+00:00")) #Handle the Z timezone as +00:00

    unix_timestamp = int(time.mktime(utc_datetime.timetuple()))
    log_record['unixEventTime'] = unix_timestamp
    return log_record

# Example Usage (using the previous log record, cloudtrail_log):
unix_timestamp_log = convert_cloudtrail_to_unix(cloudtrail_log)
print(json.dumps(unix_timestamp_log, indent=2))
```

Here, the `convert_cloudtrail_to_unix` function directly converts the UTC datetime object into a Unix timestamp using `time.mktime()`, converting the datetime object into a time tuple before being passed into `mktime`. `mktime` returns the timestamp as a float, therefore I wrap the result in `int()` to remove the decimal portion, providing an integer representation. The Unix timestamp is added to the log record dictionary under the `'unixEventTime'` key.

Finally, there are situations where only a specific portion of the timestamp, like only the date, is required. This example demonstrates extracting only the date component and presenting it as a string:

```python
import datetime
import json

def extract_cloudtrail_date(log_record):
    """Extracts the date portion from a CloudTrail timestamp.

    Args:
        log_record (dict): A dictionary representing a single CloudTrail log record.

    Returns:
         dict: The log record with the extracted date.
    """
    if 'eventTime' not in log_record:
        return log_record

    timestamp_string = log_record['eventTime']
    utc_datetime = datetime.datetime.fromisoformat(timestamp_string.replace("Z","+00:00")) #Handle the Z timezone as +00:00

    formatted_date = utc_datetime.strftime("%Y-%m-%d") #Format only the date
    log_record['eventDate'] = formatted_date
    return log_record

# Example Usage (using the previous log record, cloudtrail_log):
date_only_log = extract_cloudtrail_date(cloudtrail_log)
print(json.dumps(date_only_log, indent=2))
```

In this case, the `extract_cloudtrail_date` function uses `strftime` with the format string `"%Y-%m-%d"` to extract only the year, month, and day from the UTC datetime object, discarding the time information. The extracted date string is added to the log record dictionary with the key `eventDate`.

For further learning, I would recommend exploring the documentation of the `datetime` module and the `pytz` library within Python. Reviewing the official documentation of the ISO 8601 standard will help in understanding the timestamp format. Additionally, documentation on specific time formatting codes used within `strftime` can provide precise control over timestamp outputs. These resources provide granular details on handling timezones, different time representations, and custom formatting options allowing for versatile manipulation of timestamps for any use-case.
