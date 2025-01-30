---
title: "How to make and combine multiple requests concurrently using the `requests` module in Python and export the results to Excel?"
date: "2025-01-30"
id: "how-to-make-and-combine-multiple-requests-concurrently"
---
Python's `requests` library, while excellent for synchronous HTTP requests, becomes inefficient when fetching data from numerous endpoints. The default behavior is to wait for each request to complete before initiating the next, leading to significant latency. Therefore, implementing concurrency is essential to drastically reduce total execution time, especially in data aggregation scenarios. Here's how to achieve that by combining `requests` with Python's standard `concurrent.futures` module, followed by an export to Excel using `openpyxl`.

My experience involves building a data pipeline that pulls pricing data from various vendor APIs. Initially, with sequential requests, a job that would ultimately take mere seconds to process computationally took upwards of a minute or more, simply due to I/O wait times. Moving to concurrent requests reduced that to a much more acceptable timeframe, on par with the actual data processing work.

**Core Concepts: Concurrency and Thread Pools**

The crucial step is leveraging threads or processes that execute requests in parallel. I've generally found thread pools to be adequate for I/O bound tasks like network requests, as they are less resource intensive than managing separate processes. When a new request is submitted to a thread pool, it assigns the task to an available worker thread. If no worker is free, the task waits until one becomes available. This manages a finite pool of resources, preventing the system from being overwhelmed.

The `concurrent.futures` module provides the `ThreadPoolExecutor`, which manages these thread pools. It lets you submit callables – in our case, a function executing the HTTP requests – and returns a `Future` object for each submission. This `Future` is a placeholder for the result of that asynchronous operation. It allows you to check if a task is done, retrieve its result, and handle potential exceptions.

For exporting to Excel, I use `openpyxl`. It allows programmatic manipulation of spreadsheet files. I find it's a relatively lightweight yet capable solution for basic data dumps, as compared to other methods like creating comma-separated value files.

**Code Examples**

Let's break this process down with concrete examples. Assume we need to retrieve data from three different API endpoints. We will use `https://httpbin.org/get` as a placeholder. This endpoint returns a JSON representation of the request.

**Example 1: Basic Concurrent Requests**

This first example demonstrates how to set up the thread pool, submit tasks to it and gather the results.

```python
import requests
import concurrent.futures
import json

def fetch_data(url):
    """Fetches data from the given URL and returns the parsed JSON."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None

def main():
    urls = [
        "https://httpbin.org/get?id=1",
        "https://httpbin.org/get?id=2",
        "https://httpbin.org/get?id=3"
    ]
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
       future_to_url = {executor.submit(fetch_data, url): url for url in urls}

       for future in concurrent.futures.as_completed(future_to_url):
           url = future_to_url[future]
           try:
               data = future.result()
               if data:
                   results.append(data)
           except Exception as e:
               print(f"Error processing {url}: {e}")

    print("Fetched Data:")
    for item in results:
       print(json.dumps(item, indent=2))

if __name__ == "__main__":
    main()
```

*   `fetch_data(url)`:  This function encapsulates the request logic, returning the parsed JSON response or `None` if an error occurs. Error handling is included using a `try-except` block, as network operations can be prone to failure.
*   `ThreadPoolExecutor(max_workers=5)`: This initializes a thread pool with a limit of 5 worker threads. This value should be adjusted based on the expected number of requests and available system resources. Experimentation can reveal the optimum count for each specific task.
*   `executor.submit(fetch_data, url)`:  This submits the `fetch_data` function, along with the relevant URL as an argument, to the thread pool. The `Future` object is stored along with its associated URL in a dictionary, `future_to_url`.
*   `concurrent.futures.as_completed()`: Iterates over `Future` objects as they complete, regardless of their submission order. This prevents blocking while waiting for slower requests.
*   `future.result()`: Retrieves the result from each `Future` object. A `try-except` block catches any exception during the request or result retrieval process, printing the error and moving to the next `Future`.

**Example 2: Structuring Data for Export**

The next example expands upon the previous one and prepares the data for eventual writing to Excel.

```python
import requests
import concurrent.futures
import openpyxl

def fetch_data_for_excel(url):
    """Fetches data and extracts relevant fields, returns a list."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        json_data = response.json()
        args = json_data.get("args", {})
        return [url, args.get('id'), json_data.get("origin")]
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return [url, "Error", str(e)]

def main():
    urls = [
        "https://httpbin.org/get?id=1",
        "https://httpbin.org/get?id=2",
        "https://httpbin.org/get?id=3"
    ]
    data_rows = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_url = {executor.submit(fetch_data_for_excel, url): url for url in urls}

        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
               row = future.result()
               data_rows.append(row)
            except Exception as e:
                print(f"Error processing {url}: {e}")


    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.append(["URL", "ID", "Origin"])
    for row in data_rows:
      sheet.append(row)

    workbook.save("output.xlsx")
    print("Data exported to output.xlsx")


if __name__ == "__main__":
    main()
```
*   `fetch_data_for_excel(url)`: This function now extracts specific fields (`id` from the query parameters, as well as the "origin" IP address returned by the endpoint) and returns a list that represents a single row for the spreadsheet. It also now handles errors by placing the error in the row instead of skipping it.
*   `openpyxl.Workbook()` and `sheet.append()`:  These `openpyxl` function calls create a workbook object and add column headers before populating with data. This provides a basic structure to the exported data.

**Example 3: Adding Rate Limiting**

When interacting with APIs, it's crucial to be mindful of rate limits. This last example introduces the `time.sleep` function to demonstrate a simplified rate limiting technique. For more complex solutions one might use more elaborate methods, possibly with a third-party rate limiting library.

```python
import requests
import concurrent.futures
import openpyxl
import time

def fetch_data_with_limit(url, request_count, delay_sec):
    """Fetches data, demonstrating rate limiting."""
    try:
       if request_count % 2 == 0:  # Simulate a rate limit, delay after every 2 requests
            time.sleep(delay_sec)
       response = requests.get(url, timeout=10)
       response.raise_for_status()
       json_data = response.json()
       args = json_data.get("args", {})
       return [url, args.get('id'), json_data.get("origin")]

    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return [url, "Error", str(e)]

def main():
    urls = [
        "https://httpbin.org/get?id=1",
        "https://httpbin.org/get?id=2",
        "https://httpbin.org/get?id=3",
        "https://httpbin.org/get?id=4",
        "https://httpbin.org/get?id=5",
        "https://httpbin.org/get?id=6"
    ]
    data_rows = []
    request_count = 0
    delay = 2

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
         future_to_url = {executor.submit(fetch_data_with_limit, url, request_count+index, delay): url for index, url in enumerate(urls)}

         for future in concurrent.futures.as_completed(future_to_url):
           url = future_to_url[future]
           try:
               row = future.result()
               data_rows.append(row)
           except Exception as e:
               print(f"Error processing {url}: {e}")

    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.append(["URL", "ID", "Origin"])
    for row in data_rows:
        sheet.append(row)

    workbook.save("output_rate_limited.xlsx")
    print("Data exported to output_rate_limited.xlsx")

if __name__ == "__main__":
    main()
```

*   `fetch_data_with_limit(url, request_count, delay_sec)`: This function demonstrates a crude form of rate limiting by sleeping before certain requests. The `delay_sec` argument controls the duration of the delay.

*   The `request_count` and the sleep condition are added to demonstrate rate limits on the execution of requests.

**Resource Recommendations**

For a deeper understanding of the concepts, I would suggest reviewing the official Python documentation on the following modules: `concurrent.futures`, `requests`, and `openpyxl`. These are the most reliable and comprehensive sources of information. Additionally, books on concurrent programming in Python are valuable for grasping the nuances of thread management. While I can't provide specific URLs, searching for Python official documentation and good concurrency books will be fruitful. Pay particular attention to the error handling best practices as that is frequently a pain point in development.

By implementing these strategies – concurrent requests, structured data handling, and mindful rate limiting – I've been able to significantly enhance the performance and reliability of my data collection pipelines. These techniques provide a solid foundation for efficient network operations with Python.
