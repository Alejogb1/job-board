---
title: "How can I improve 3rd party API query times using asynchronous or multi-threading?"
date: "2024-12-23"
id: "how-can-i-improve-3rd-party-api-query-times-using-asynchronous-or-multi-threading"
---

Alright, let's tackle this. I've spent more than a few cycles optimizing api integrations over the years, and the frustration with slow third-party endpoints is something I know all too well. The core issue, as you’ve highlighted, often boils down to synchronous execution where we're waiting idly for one request to finish before starting the next, especially painful with external apis that have unpredictable response times. Let's unpack how asynchronous processing and multi-threading can significantly reduce these waiting periods and improve query times.

The fundamental concept is straightforward: instead of waiting for each api request to complete in sequence, we aim to initiate multiple requests concurrently. Asynchronous programming enables a single thread to handle multiple operations without blocking. When one operation is waiting, for example, for a remote server to respond, the thread isn't held up; instead, it’s free to continue with other work. Multi-threading, on the other hand, uses multiple threads concurrently, each capable of independent execution, allowing you to push even more requests in parallel. Which approach is preferable will often depend on the nature of your workloads, the api itself, and the language or platform you're using.

Let's start with asynchronous programming, which often uses `async/await` constructs or promise-based approaches in javascript, python, and others. The goal is to perform i/o operations—like network requests—concurrently without locking the main thread. To illustrate, consider a scenario where I had to fetch data from a weather api for multiple locations. A synchronous approach would look something like this (in pseudo-python for simplicity):

```python
import time
import requests

def fetch_weather_sync(locations):
  start_time = time.time()
  for location in locations:
    print(f"Fetching weather for {location}")
    response = requests.get(f"https://fake-weather-api.com/weather/{location}")
    if response.status_code == 200:
      data = response.json()
      print(f"Weather in {location}: {data['temperature']}")
    else:
      print(f"Error fetching data for {location}")

  end_time = time.time()
  print(f"Total time sync: {end_time - start_time:.2f} seconds")

locations = ['london', 'newyork', 'tokyo', 'sydney']
fetch_weather_sync(locations)
```

Notice how each request blocks until the previous is finished. With even moderate latencies from the third-party api, that adds up quickly. The fix is to utilize `async` and `await`, alongside an asynchronous http client. Here is a similar, but asynchronous version:

```python
import asyncio
import time
import aiohttp

async def fetch_weather_async(session, location):
  print(f"Fetching weather for {location}")
  async with session.get(f"https://fake-weather-api.com/weather/{location}") as response:
    if response.status == 200:
      data = await response.json()
      print(f"Weather in {location}: {data['temperature']}")
    else:
      print(f"Error fetching data for {location}")

async def main_async():
  start_time = time.time()
  async with aiohttp.ClientSession() as session:
    tasks = [fetch_weather_async(session, location) for location in ['london', 'newyork', 'tokyo', 'sydney']]
    await asyncio.gather(*tasks)
  end_time = time.time()
  print(f"Total time async: {end_time - start_time:.2f} seconds")


asyncio.run(main_async())
```

In this version, we use `aiohttp`, an asynchronous http client for python, and `asyncio` for task management. The `await` keyword pauses the current coroutine until the operation finishes, but crucially, it doesn't block the event loop which is free to begin or resume other tasks. That results in much faster aggregate request times. The `asyncio.gather` function schedules all the async requests simultaneously and waits for them all to complete, a crucial difference from the synchronous example. You'll notice, of course, that I have replaced the `requests` library with the asynchronous equivalent `aiohttp` in this code, a very common requirement when switching from blocking, synchronous calls to asynchronous ones.

Now, let’s move on to multi-threading. While asynchronous techniques often suffice for i/o bound operations, cpu intensive tasks or situations where you need to maximize resource utilization might be better suited for threading or, if your platform supports it, multi-processing. With multi-threading we create several concurrent threads, each capable of executing similar logic in parallel, but on separate virtual cores within your system, allowing you to avoid the global interpreter lock limitations that can limit the ability of python's asynchronous libraries to take advantage of multi-core processors. This means multiple requests are not interleaved but can be done completely concurrently. Here is a threaded example:

```python
import threading
import time
import requests

def fetch_weather_threaded(location):
  print(f"Fetching weather for {location} (thread: {threading.current_thread().name})")
  response = requests.get(f"https://fake-weather-api.com/weather/{location}")
  if response.status_code == 200:
    data = response.json()
    print(f"Weather in {location}: {data['temperature']} (thread: {threading.current_thread().name})")
  else:
    print(f"Error fetching data for {location} (thread: {threading.current_thread().name})")

def main_threaded():
  start_time = time.time()
  threads = []
  locations = ['london', 'newyork', 'tokyo', 'sydney']

  for location in locations:
    thread = threading.Thread(target=fetch_weather_threaded, args=(location,))
    threads.append(thread)
    thread.start()

  for thread in threads:
    thread.join()

  end_time = time.time()
  print(f"Total time threaded: {end_time - start_time:.2f} seconds")

main_threaded()
```

Here, we initiate multiple threads using `threading.Thread`. Each thread executes the `fetch_weather_threaded` function independently. Crucially, the `thread.join()` calls ensure the main program waits for all spawned threads to complete before proceeding. Note that this approach, unlike the async one, uses multiple threads within the same process.

Which of these techniques is best depends on the specifics of your case and platform. Asynchronous programming is generally simpler to set up and more efficient when the primary delays are i/o bound, as is often the case with api requests. Multi-threading (or multi-processing) can potentially give greater performance if you want to make the most of multi-core processors but adds complexity with the need for careful synchronization and shared resource access management. In the past, in scenarios where a single instance was hitting rate limits on apis, I’ve seen multi-threading work much better, allowing me to avoid overloading a single connection but, as always, this would have to be carefully tested.

For further study, I recommend looking into “Concurrent Programming on Windows” by Joe Duffy for a deep dive into multi-threading concepts and challenges. For asynchronous programming, “Effective Modern C++” by Scott Meyers provides excellent insight on asynchronous programming in C++, and the documentation for python's `asyncio` and `aiohttp` libraries are very thorough. Additionally, the "Operating System Concepts" textbook by Silberschatz et al, is a great resource on the underlying theory of processes and threads. These resources provide solid foundations and cover concepts relevant to both approaches. The key is understanding the specific constraints of your system, api and usage case, and deciding which concurrency tool works best for your particular scenario.
