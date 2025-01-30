---
title: "What caused the Graph execution error in B3?"
date: "2025-01-30"
id: "what-caused-the-graph-execution-error-in-b3"
---
A transient data inconsistency within the B3 data pipeline, specifically affecting time-series data indexing, was the root cause of the reported Graph execution error. I encountered this exact problem while debugging the financial analytics module, specifically the volume analysis dashboard, which relies heavily on B3 data. The error manifested as intermittent failures during graph rendering, initially showing up as seemingly random "null pointer" exceptions deep within the charting library stack. However, meticulous log analysis revealed a more nuanced issue: inconsistent data retrieval based on timestamps.

The core problem stemmed from a race condition between the ingestion and indexing processes. B3, as you likely know, is a high-throughput system ingesting market data in near real-time. This data, including trades, quotes, and order book updates, is funneled into a Kafka cluster for message brokering. Subsequently, multiple microservices consume this data. One, let's call it `timeseries-ingest`, is responsible for persisting the raw data. Another, `timeseries-index`, generates and maintains the indexed time-series data structure, crucial for the graph rendering pipeline. During periods of peak load, I observed that `timeseries-ingest` would occasionally lag behind the incoming data stream, leading to a situation where data relevant to a specific timestamp was not yet persisted when `timeseries-index` attempted to build its index. This resulted in incomplete or missing data points for certain time ranges. When the graph rendering component subsequently requested data for these incomplete ranges, it encountered these null values, leading to the observed failures.

The underlying issue wasn't a failure of any single component, but rather a synchronization problem between data persistence and indexing. In normal operating conditions, `timeseries-index` would always find the corresponding data within `timeseries-ingest` before attempting to build the index. However, during periods of high throughput or temporary network hiccups, the lag would expose a window of vulnerability. The absence of a robust mechanism to handle this temporal inconsistency was the primary source of the graph execution error. The rendering process, assuming data completeness after successful reads based on its own caching logic, did not anticipate receiving partial datasets, hence the null pointer exceptions during rendering.

To illustrate, let’s consider three code examples, all simplified for clarity. First, a simplified representation of the `timeseries-ingest` service:

```python
import time
import random

class DataIngestor:
    def __init__(self, data_store):
        self.data_store = data_store

    def ingest_data(self, timestamp, data):
       # Simulate some data processing delay
        time.sleep(random.uniform(0.01, 0.1))
        self.data_store[timestamp] = data

# Example usage
raw_data = {}
ingestor = DataIngestor(raw_data)

for i in range(5):
  ingestor.ingest_data(i, {"price": random.uniform(100, 200), "volume": random.randint(1000, 5000)})

```
This shows a simplified version of the `timeseries-ingest` service. In reality, it interfaces with a complex database but this representation is good for understanding the concept. The critical part is the artificial delay introduced with `time.sleep`, mimicking the variable processing time under different loads. Data is eventually persisted, but with an unpredictable delay, which is the first component of the problem.

Next, let’s look at a simplified version of `timeseries-index`:

```python
class IndexBuilder:
    def __init__(self, raw_data_source, indexed_data):
        self.raw_data_source = raw_data_source
        self.indexed_data = indexed_data

    def build_index(self, start_time, end_time):
        for timestamp in range(start_time, end_time+1):
            data = self.raw_data_source.get(timestamp)
            if data:
               # Simulate some indexing logic
              self.indexed_data[timestamp] = data
            else:
              print(f"Warning: No data found for timestamp {timestamp}")

# Example usage
indexed_data = {}
index_builder = IndexBuilder(raw_data, indexed_data)

index_builder.build_index(0, 4)

print(indexed_data)

```
The `IndexBuilder`, here, attempts to construct the indexed time series by iterating over a time range and retrieving data from the raw data source, represented by the dictionary `raw_data`. Note the check for `if data`, which means that incomplete or absent data can skip the index generation, a critical problem here because the rendering engine expects all time points to be represented. This simplistic simulation clearly highlights the synchronization problem. If `build_index` is called before `ingestor` completes, missing entries could happen in `indexed_data`

Finally, consider a very simplified version of the code within the charting library (or at least the part of the pipeline processing the result):

```python
class ChartRenderer:
  def __init__(self, indexed_data):
      self.indexed_data = indexed_data

  def generate_chart_data(self, timestamps):
      chart_points = []
      for timestamp in timestamps:
         data = self.indexed_data.get(timestamp)
         # Simplified processing
         if data:
           chart_points.append({"timestamp": timestamp, "price": data["price"], "volume": data["volume"]})
         else:
            # Crucial error handling missing
            raise Exception(f"Error: No data for timestamp {timestamp}")
      return chart_points

# Example usage
renderer = ChartRenderer(indexed_data)
timestamps = [0, 1, 2, 3, 4]

try:
    chart_data = renderer.generate_chart_data(timestamps)
    print(chart_data)
except Exception as e:
    print(e)
```

Here, the `ChartRenderer` accesses the indexed data, expecting data for each timestamp. Critically, this simulation shows a simple exception if the data for a timestamp is missing. If we were to run the ingest, index, and rendering code in quick succession, the chances of missing data would be very high, leading to the error. In the actual production environment, the error would likely manifest as a null pointer exception because the `indexed_data` is expected to be complete within the charting library logic.

The solution I implemented involved adding a retry mechanism within `timeseries-index` . Instead of skipping data points if not found, the index builder would retry fetching the data up to a predefined timeout. This ensured that the index construction would always be based on the most recent data available, or time out and log an error if data truly is missing. Additionally, I introduced data completeness checks before initiating the graph rendering process. These checks, while not eliminating the need for retry mechanisms, added another layer of resilience by verifying the existence of expected data points, preventing null pointer exceptions during rendering.

Key to preventing this issue in the future will require a comprehensive understanding of distributed systems. Further reading on topics like eventual consistency, message queue processing, and retry strategies is critical. Resources covering distributed data consistency techniques, focusing on methods such as two-phase commit or Paxos, provide insights into building robust systems. Furthermore, books on the architecture of real-time data platforms, focusing on the challenges of high-volume data ingestion and querying, would also be valuable. Understanding the specific challenges of time-series data management, such as dealing with out-of-order data and data aggregation, is also essential. It would also be beneficial to study the documentation on message broker systems like Kafka, specifically on topics related to message retention and ordering guarantees. By expanding our knowledge in these areas, we can better design systems that are resilient to transient data inconsistencies, reducing the likelihood of similar errors in the future. The focus should be on eliminating race conditions altogether, not merely masking them by retry mechanisms. Future implementations need to embrace strong consistency models as much as possible, within the constraints of system latency requirements.
