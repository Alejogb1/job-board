---
title: "How do real-time streaming pipelines process data from sources with different update frequencies?"
date: "2024-12-10"
id: "how-do-real-time-streaming-pipelines-process-data-from-sources-with-different-update-frequencies"
---

Okay so you wanna know how streaming pipelines handle data coming in at different speeds right  like imagine one source sending updates every second and another only every hour  that's a real challenge  but it's totally doable  the key is understanding that these pipelines aren't about *waiting* for everyone to be ready they're about *managing* the flow

Think of it like a highway some cars are going 70mph others are creeping along  you don't want the fast cars to hold up the slow ones or vice versa  you need lanes  or in our case  separate processing paths  or maybe even clever merging strategies

One super common approach is using message queues like Kafka  it acts as a buffer a big holding area  sources dump their data into Kafka  no matter how fast or slow they are  the pipeline then pulls data from Kafka at its own pace  This decoupling is huge  it means the slowpoke sensor data doesn't block the high-speed stock ticker updates

Kafka is really good at handling different rates because it's designed to be a distributed system  lots of little servers working together  so if one part gets overwhelmed the others can pick up the slack  you can find a lot of good info in the official Kafka documentation  and also the book "Designing Data-Intensive Applications" by Martin Kleppmann  that's a bible for this kind of stuff

Another aspect is windowing  imagine you're analyzing website traffic  you get clicks every few seconds but you want to see the total number of clicks per hour  so you group those fast-arriving clicks into hourly buckets  these are called time windows  you do this kind of grouping before any serious processing happens  it's like summarizing the data before you analyze it making things much simpler

Here's a little Python example using `pandas` which is great for this kind of data manipulation just for illustrative purposes  I'm skipping some error handling and fancy stuff but you get the idea

```python
import pandas as pd
import time

# Sample data  simulating different update frequencies
data = [
    {'source': 'slow', 'timestamp': time.time(), 'value': 1},
    {'source': 'fast', 'timestamp': time.time(), 'value': 10},
    {'source': 'fast', 'timestamp': time.time() + 1, 'value': 20},
    {'source': 'slow', 'timestamp': time.time() + 60 * 60, 'value': 2},  # Slow update after an hour
    {'source': 'fast', 'timestamp': time.time() + 2, 'value': 30}
]

df = pd.DataFrame(data)
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

# Create hourly windows
df['hour'] = df['timestamp'].dt.floor('H')

# Group by hour and source and sum the values
hourly_sums = df.groupby(['hour', 'source'])['value'].sum().reset_index()
print(hourly_sums)
```


Then there are rate-limiting techniques  sometimes you need to slow things down intentionally  maybe your database can't handle the full flood of data  you might use techniques to throttle the flow like sampling only taking a percentage of the incoming data  or just plain ignoring some events  it's a balancing act between accuracy and performance

This might sound crude but sometimes it's necessary  especially during peak times  you can do this at the queue level or within your processing logic  think of it as traffic management for your data


Another big topic is data buffering  you essentially create a temporary storage area to smooth out the variations in data arrival rates  this buffer might be a database table a memory queue or even a file system  the size of the buffer determines how much irregularity you can absorb

Larger buffers mean less sensitivity to bursts of data but also more latency  smaller buffers mean more responsiveness but risk overload if the arrival rate spikes  it's a trade-off that depends on your application


Let's look at a simple example using Python with a queue for buffering   again very basic no real error handling just to illustrate

```python
import queue
import time

q = queue.Queue()

# Simulate data sources with varying update frequencies
def slow_source():
    while True:
        q.put({'source': 'slow', 'value': 1})
        time.sleep(60)  # Slow update every 60 seconds

def fast_source():
    while True:
        q.put({'source': 'fast', 'value': 10})
        time.sleep(1)  # Fast update every second

# Start the data sources  running in separate threads is generally a good idea here
# ...thread handling omitted for simplicity...


# Process data from the queue
while True:
    try:
        item = q.get(timeout=1)  # Try to get an item  timeout to prevent blocking indefinitely
        print(f"Processing: {item}")
        q.task_done()  # Mark task as complete
    except queue.Empty:
        print("Queue is empty")

```

Finally there's stream processing frameworks like Apache Flink or Apache Spark Streaming  these are more advanced  they provide built-in mechanisms for handling varying data rates windowing state management fault tolerance and more  they abstract away much of the complexity involved in building a robust streaming pipeline


Flink particularly excels at precisely ordered processing even with out-of-order arrivals of data from different sources  it uses sophisticated techniques to ensure everything is handled correctly  "Learning Spark" by Holden Karau et al is good for Spark  for Flink the official documentation is your best bet


Consider this  let's say you're building a system for fraud detection  you have transaction data coming in fast and slow customer profile updates  you can't afford to lose any data  and you need to process everything in real time or near real time  a properly designed streaming pipeline with good buffering windowing and error handling is absolutely essential  you wouldn't want a delayed alert leading to massive losses right


Here's a tiny glimpse of what that might look like conceptually using a pseudo-code style

```
// Pseudocode for a fraud detection streaming pipeline

// Data sources: Transactions (high frequency), Customer Profiles (low frequency)

// Kafka as message queue for both sources

// Flink job processing data from Kafka

// Windowing: Aggregate transactions per customer per hour

// Join: Join aggregated transactions with customer profiles

// Anomaly detection: Apply machine learning model for fraud detection

// Output: Real-time fraud alerts
```

This isn't production-ready code  but it shows the general idea  the actual implementation would be far more involved and depend on specific requirements and frameworks  but the core principles remain the same  handle the flow manage the speeds and buffer effectively


Remember reading those books is key  they'll cover a lot of the finer details and practical considerations  good luck  building these pipelines is a challenging but rewarding journey  you'll learn a ton along the way
