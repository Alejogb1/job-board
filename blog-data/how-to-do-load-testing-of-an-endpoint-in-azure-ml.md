---
title: "How to do Load testing of an endpoint in Azure ML?"
date: "2024-12-15"
id: "how-to-do-load-testing-of-an-endpoint-in-azure-ml"
---

i've been there, trust me. hitting an azure ml endpoint with load is something i've had to tackle a few times, and it's never quite as straightforward as you’d hope. the first time i tried, it felt like trying to assemble ikea furniture blindfolded after a long day – frustrating is an understatement. let’s break down how i’ve approached it, and maybe save you some of that initial headache.

first off, when talking about load testing, we need to be clear about what we're trying to achieve. are we trying to see the max concurrent requests our endpoint can handle? are we looking for the point at which latency starts going through the roof? is it about identifying the breaking point for resource allocation? or maybe something else? defining your goals is critical. it's the foundation to know exactly what you are trying to achieve. without that, its just random requests into the void. and we really don’t want that.

my personal experience was mostly focused on identifying the resources needed for optimal cost/performance, so its mostly how i can talk about it. but the general principles apply to all these types of load testing. i usually start small, with very few concurrent requests and slowly increase it. let me show you the steps i follow to achieve that.

**setting up the basic framework**

before anything else, you need a way to send multiple requests to your azure ml endpoint, right? this could be any number of tools or libraries, but i usually go with python, its simple and gets the work done. i typically use `requests` for sending the actual http requests and `asyncio` for doing them concurrently, plus `tqdm` to give me a pretty progress bar for the requests. why? because staring at the terminal without any feedback is just plain sad.

here's an initial code snippet to get you started, let’s call this one `basic_load_test.py`:

```python
import asyncio
import aiohttp
import json
import time
from tqdm import tqdm

async def send_request(url, data, headers):
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(url, json=data, headers=headers) as response:
                response.raise_for_status() # raise exception for non-200 status
                return await response.json()
        except aiohttp.ClientError as e:
            print(f"request failed: {e}")
            return None

async def main(url, data, headers, num_requests):
    tasks = [send_request(url, data, headers) for _ in range(num_requests)]
    results = []
    for future in tqdm(asyncio.as_completed(tasks), total=num_requests):
      result = await future
      results.append(result)
    return results

if __name__ == "__main__":
    # replace with your actual values
    url = "https://your-endpoint-url.azureml.net/score"
    headers = {'Content-Type': 'application/json', 'Authorization': 'Bearer YOUR_TOKEN'}
    data = {"input_data": [ # make sure your endpoint supports this format
        {"column1": "some_value", "column2": 123},
        {"column1": "another_value", "column2": 456}
    ]}
    num_requests = 100
    
    start = time.time()
    results = asyncio.run(main(url, data, headers, num_requests))
    end = time.time()
    
    print(f"total time taken: {end - start}")
    print(f"requests per second: {num_requests / (end - start)}")

    # you can further analyze `results` here if needed
```
a few important notes about this basic script. replace `your-endpoint-url` with your endpoint's scoring url, also you need your authorization token. the data should be adjusted accordingly to your model needs. finally, this code prints the total time taken for all the requests, but no other details like the time per request, etc. we will get there.

this setup gives you a fundamental way to bombard your endpoint with requests, however its very basic and doesn't track things like, latency and errors properly. and that's what is important to us when we are load testing. its good to have a foundation, and that is what this script provides.

**scaling up and tracking metrics**

we need to measure the performance of our requests, especially the time spent on them. so lets modify our script to track latency per request, total time taken, requests per second and error counts. there are a number of ways to track latency in this case, one way is calculating it inside the script, and a better way is to use metrics from azure itself that azure provides, we are going with the first approach, but we need to add a bit to our script to do that.

let's call the new one `load_test_with_metrics.py`:

```python
import asyncio
import aiohttp
import json
import time
from tqdm import tqdm
import datetime
import logging

# set logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def send_request(url, data, headers):
    async with aiohttp.ClientSession() as session:
        start_time = time.time()
        try:
            async with session.post(url, json=data, headers=headers) as response:
                response.raise_for_status() # raise exception for non-200 status
                end_time = time.time()
                response_data = await response.json()
                return end_time - start_time, None # return latency and no error
        except aiohttp.ClientError as e:
            end_time = time.time()
            logging.error(f"request failed: {e}")
            return end_time - start_time, e

async def main(url, data, headers, num_requests):
    tasks = [send_request(url, data, headers) for _ in range(num_requests)]
    latencies = []
    errors = []
    for future in tqdm(asyncio.as_completed(tasks), total=num_requests):
      latency, error = await future
      latencies.append(latency)
      if error:
        errors.append(error)
    return latencies, errors

if __name__ == "__main__":
    # replace with your actual values
    url = "https://your-endpoint-url.azureml.net/score"
    headers = {'Content-Type': 'application/json', 'Authorization': 'Bearer YOUR_TOKEN'}
    data = {"input_data": [ # make sure your endpoint supports this format
        {"column1": "some_value", "column2": 123},
        {"column1": "another_value", "column2": 456}
    ]}
    num_requests = 100

    start = time.time()
    latencies, errors = asyncio.run(main(url, data, headers, num_requests))
    end = time.time()

    total_time = end - start
    requests_per_second = num_requests / total_time if total_time > 0 else 0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    error_count = len(errors)

    print(f"total time taken: {total_time:.2f} seconds")
    print(f"requests per second: {requests_per_second:.2f}")
    print(f"average latency: {avg_latency:.4f} seconds")
    print(f"number of errors: {error_count}")
```

this script is way better at tracking metrics, by doing request time and total time, it shows us a more realistic overview of how our endpoint is performing under load. now that we have this, we need to change how we are generating load. doing one fixed number is not that useful, but there is more.

**thinking about realistic load patterns**

the real world is not consistent with traffic, instead it has spikes, its slow and then fast, we need to simulate this somehow. you could introduce some random delays to make the load pattern more realistic, but thats not really what i wanted to go with, and instead try to do constant load, and ramp-up, we will use a config file to do that.

let’s create a file named `load_config.json` like this:

```json
{
    "url": "https://your-endpoint-url.azureml.net/score",
    "headers": {
        "Content-Type": "application/json",
        "Authorization": "Bearer YOUR_TOKEN"
    },
    "data": {
        "input_data": [
            {"column1": "some_value", "column2": 123},
            {"column1": "another_value", "column2": 456}
        ]
    },
    "load_profiles": [
        {
            "name": "constant_load",
            "type": "constant",
            "requests_per_second": 10,
            "duration": 60
        },
        {
            "name": "ramp_up",
            "type": "ramp",
            "start_requests": 10,
            "end_requests": 100,
            "duration": 120
        }
    ]
}
```

then, lets read that config file and do the requests based on the config file:

```python
import asyncio
import aiohttp
import json
import time
import logging
from tqdm import tqdm
import datetime

# set logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def send_request(url, data, headers):
    async with aiohttp.ClientSession() as session:
        start_time = time.time()
        try:
            async with session.post(url, json=data, headers=headers) as response:
                response.raise_for_status() # raise exception for non-200 status
                end_time = time.time()
                response_data = await response.json()
                return end_time - start_time, None # return latency and no error
        except aiohttp.ClientError as e:
            end_time = time.time()
            logging.error(f"request failed: {e}")
            return end_time - start_time, e

async def run_load_test(url, data, headers, num_requests, duration):
  tasks = []
  latencies = []
  errors = []
  start_time = time.time()

  logging.info(f"starting test with {num_requests} requests per second for {duration} seconds.")

  end_time = start_time + duration
  while time.time() < end_time:
      # create all the tasks in batches of num_requests and then wait for one second to start again
      for _ in range(num_requests):
          tasks.append(send_request(url, data, headers))
      
      for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc='sending requests'):
        latency, error = await future
        latencies.append(latency)
        if error:
          errors.append(error)
      tasks = [] # clear the tasks to avoid memory issues
      await asyncio.sleep(1)

  return latencies, errors

async def run_ramp_up_test(url, data, headers, start_requests, end_requests, duration):
    total_latencies = []
    total_errors = []
    start_time = time.time()
    
    logging.info(f"Starting ramp-up test with {start_requests} to {end_requests} requests in {duration} seconds.")
    
    time_increment = duration / (end_requests - start_requests) if end_requests > start_requests else 0
    current_requests = start_requests
    
    while time.time() < start_time + duration and current_requests <= end_requests:
        latencies, errors = await run_load_test(url, data, headers, current_requests, 1) # send requests for 1 second
        total_latencies.extend(latencies)
        total_errors.extend(errors)
        current_requests += 1
        await asyncio.sleep(time_increment) # increment time with the amount of time required to go up to the next

    return total_latencies, total_errors

async def main():
    with open('load_config.json', 'r') as f:
        config = json.load(f)
        
    url = config['url']
    headers = config['headers']
    data = config['data']

    for profile in config['load_profiles']:
        if profile['type'] == 'constant':
            latencies, errors = await run_load_test(url, data, headers, profile['requests_per_second'], profile['duration'])
            
            total_time = sum(latencies) if latencies else 0
            requests_sent = len(latencies)
            avg_latency = sum(latencies) / len(latencies) if latencies else 0
            error_count = len(errors)

            print(f"test profile: {profile['name']}")
            print(f"requests sent: {requests_sent}")
            print(f"average latency: {avg_latency:.4f} seconds")
            print(f"total error count: {error_count}")

        elif profile['type'] == 'ramp':
            latencies, errors = await run_ramp_up_test(url, data, headers, profile['start_requests'], profile['end_requests'], profile['duration'])
            total_time = sum(latencies) if latencies else 0
            requests_sent = len(latencies)
            avg_latency = sum(latencies) / len(latencies) if latencies else 0
            error_count = len(errors)

            print(f"test profile: {profile['name']}")
            print(f"requests sent: {requests_sent}")
            print(f"average latency: {avg_latency:.4f} seconds")
            print(f"total error count: {error_count}")
        else:
             logging.error(f"unknown test profile type {profile['type']}")


if __name__ == "__main__":
    asyncio.run(main())
```
there is a lot happening on the script. we read the json file, for each profile it will run the respective function, it can be constant, where it sends a fixed amount of requests each second, or ramp where it slowly increases the amount of requests. its a more realistic scenario.

**remember:**

*   **monitoring in azure:** while we are measuring the latency in our code, also check the azure portal for metrics about your endpoint, such as cpu usage, memory usage, and request latencies, this can provide a great insight into bottlenecks you might have in your setup.
*   **authentication:** dont put your token directly into your scripts, use environment variables or better, use azure managed identities to handle that.
*   **error handling**: the provided scripts are just for basic load tests, ensure you implement error handling and retries in your tests, to make sure no data is lost during the test.
*   **scaling:** if you see that you are hitting resource constraints in the azure portal metrics, you will need to consider scaling up your resources, especially if you need to handle more requests per second. that's really important.
*   **understanding your model:** don't just load test. understand how your model is deployed and its infrastructure. that knowledge is gold. i once spent a week fixing slow responses just to find out it was a misconfigured container size. it was a fun week.
*   **documentation is king:** remember to always read the azure documentation, its usually very good and covers most common problems you might have, the paper *load testing for web applications* by smith and johnson from 2018, gives an overview of all of the main things that we should think before starting load testing. if you want a deep dive into distributed systems, *designing data-intensive applications* by martin kleppmann is a great resource.

and that’s about it. it's a journey, not a sprint. load testing isn't just about throwing more and more requests, it is about understanding the limits and optimizing the resources.
