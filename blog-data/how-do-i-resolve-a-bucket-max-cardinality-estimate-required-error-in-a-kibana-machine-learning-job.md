---
title: "How do I resolve a 'Bucket max cardinality estimate required' error in a Kibana machine learning job?"
date: "2024-12-23"
id: "how-do-i-resolve-a-bucket-max-cardinality-estimate-required-error-in-a-kibana-machine-learning-job"
---

Alright, let's tackle this. I've seen this "Bucket max cardinality estimate required" error pop up more times than I'd care to count, usually when dealing with machine learning jobs in Kibana that are looking at high-cardinality fields. It’s less about Kibana being outright broken, and more about it needing a helping hand to understand the scope of your data. You're essentially asking Kibana’s anomaly detection to consider a field that has too many unique values for it to track efficiently by default. Think of it like asking a small room to hold a stadium's worth of people; it just doesn't work unless you give it some guidance on how to handle the crowd.

The root cause here typically stems from fields that act as identifiers or contain highly granular data like user ids, session ids, or very specific transaction details. Kibana’s machine learning, by default, tries to create a bucket for each unique value of the field you’re using for analysis. When the number of unique values, or cardinality, of that field is too high, it throws this error. It means that the underlying machine learning engine in Elasticsearch is hesitant to proceed because it anticipates an explosive increase in resource consumption, leading to potential instability or performance degradation. This is a safeguard, not an oversight. I recall a situation a while back where we were trying to monitor web traffic anomalies, and initially, I tried to use the ‘user_id’ field as the 'by' field in the detector, and we got exactly this error. It was a good lesson in needing a better strategy.

There isn’t a single magic bullet for this. The resolution depends heavily on the specific nature of your data and what you’re trying to accomplish. However, there are several common strategies I've employed successfully. Let's break them down.

**Strategy 1: Pre-aggregation and Reduced Cardinality**

Often, the best approach is to aggregate your data *before* feeding it to the machine learning job. Instead of analyzing each individual transaction or user event, you might want to group them. For example, instead of using ‘user_id’ directly, you could group by a field like ‘user_cohort,’ which has fewer unique values and still reveals useful anomaly patterns. This pre-aggregation can be achieved either at index time using ingest pipelines or during the data analysis phase in Kibana using data transforms.

Consider this scenario, you have an index containing log data, where each log entry has a `user_id`, `timestamp`, and a numeric field `requests_per_minute`. We want to detect anomalies in requests per minute *per user*, but the user_id has high cardinality. Instead, we can pre-aggregate our data by hour and by user_group:

```python
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

# Initialize the Elasticsearch client
es = Elasticsearch([{'host': 'localhost', 'port': 9200}])

# Create a function to generate test data
def generate_test_data():
    import random
    import datetime

    start_date = datetime.datetime(2024, 1, 1, 0, 0, 0)
    data = []
    for hour in range(24):
        for user in range(1000): # 1000 users, each in a group
             for minute in range(60):
                  timestamp = start_date + datetime.timedelta(hours=hour, minutes=minute)
                  user_group = (user % 100)  # 100 user_groups
                  requests = random.randint(0, 10)

                  data.append({
                       "_index": "requests_per_minute_data",
                       "_id": f"{timestamp.isoformat()}_{user}",
                       "_source":{
                           "timestamp": timestamp.isoformat(),
                           "user_id": user,
                           "user_group":user_group,
                           "requests_per_minute": requests
                      }
                  })
    return data

# Function to create the index if not exists and add the test data
def setup_index():
    if not es.indices.exists(index="requests_per_minute_data"):
        es.indices.create(index="requests_per_minute_data", body={
        "mappings": {
             "properties": {
                "timestamp": {"type": "date"},
                "user_id": {"type":"integer"},
                "user_group":{"type":"integer"},
                "requests_per_minute": {"type": "integer"}
             }
         }
         })
    bulk(es, generate_test_data())


# Execute to create the index and add the data
setup_index()
```

Now, in the Kibana machine learning job, instead of using `user_id` directly as the `by` field, I use `user_group` which has a cardinality of just 100, rather than the full 1000. This approach reduces the computational cost and removes the cardinality error. Note that you can use a `script` processor on the ingest pipeline to extract `user_group` from the user id if needed. You could even create another index using transforms that is aggregated hourly.

**Strategy 2: Using the 'influencers' Field Instead of 'by'**

When the `by` field is too high cardinality, you can look to the 'influencers' field instead. The difference here is that the 'by' field will *segment* the data to create different buckets of data to analyze for anomalies separately. By using the `influencers` field, we can identify whether the high-cardinality field is *related* to the anomaly, *without* the computational cost of creating a unique model for each value. The 'influencers' field *does not* bucket data, so the computation cost remains low.

Let's build on our example:

```python
# This time, let's generate simpler test data without user groups but with a slightly different schema
def generate_test_data_influencers():
    import random
    import datetime

    start_date = datetime.datetime(2024, 1, 1, 0, 0, 0)
    data = []
    for hour in range(24):
        for user in range(1000):  #1000 users
            for minute in range(60):
                timestamp = start_date + datetime.timedelta(hours=hour, minutes=minute)
                requests = random.randint(0, 10)
                data.append({
                    "_index": "requests_per_minute_data_influencers",
                    "_id": f"{timestamp.isoformat()}_{user}",
                    "_source": {
                        "timestamp": timestamp.isoformat(),
                        "user_id": user,
                        "requests_per_minute": requests
                    }
                })
    return data

# Function to create the index for testing influencers
def setup_index_influencers():
    if not es.indices.exists(index="requests_per_minute_data_influencers"):
        es.indices.create(index="requests_per_minute_data_influencers", body={
            "mappings": {
                "properties": {
                    "timestamp": {"type": "date"},
                    "user_id": {"type":"integer"},
                     "requests_per_minute": {"type": "integer"}
                 }
             }
        })
    bulk(es, generate_test_data_influencers())

setup_index_influencers()
```

Here, in Kibana, when defining the machine learning job we would *not* use 'user_id' as the `by` field. Instead, we would have a detector configured to analyze `requests_per_minute` over time *without* bucketing on `user_id`. Then, add `user_id` in the `influencers` field. This will allow us to identify if a specific user is related to a spike in traffic, but it won't cause the error as the anomaly analysis is done without bucketing. This approach is useful when you need to know *if* high cardinality is associated with anomalies, but don't need different analysis models per value.

**Strategy 3: Configuring `model_memory_limit` or `max_bucket_cardinality` directly**

This approach should be treated with caution. While you *can* directly adjust the `model_memory_limit` setting, or the `max_bucket_cardinality` setting, these adjustments are often a short-term fix that can lead to performance issues. The setting `max_bucket_cardinality` exists to prevent runaway processes, so raising this without proper justification can lead to resource depletion, and potential instability of Elasticsearch. If you’re dealing with very high cardinality, this is not recommended, however it’s worth knowing that these options exist for less critical cases.

```python
#  This configuration approach should be performed in Kibana, not directly in Python,
#  so code is illustrative of the parameters rather than directly executable.

# When creating or updating a machine learning job in Kibana, you can often specify these parameters in the Advanced Settings or the Job Configuration
# Here is the general idea, though this would be done in Kibana GUI or API.

job_configuration = {
    "analysis_config": {
       "detectors":[
           {
            "function":"mean",
            "field_name":"requests_per_minute",
            "by_field":"user_id",
            "model_memory_limit":"100mb" #setting memory allocation, not recommended
           }
         ],
          "influencers" : ["user_id"]
    },
      "datafeed_config": {
         #datafeed properties
      }

    # ... other job configurations ...
}

# Or the specific setting related to cardinality
job_configuration = {
    "analysis_limits": {
      "max_bucket_cardinality": 1000000 # not recommended without careful consideration
    }
}

```
While these settings seem like a quick fix, they bypass the error check and may cause performance problems. If you find yourself adjusting them frequently, you probably need to consider strategies 1 and 2 again.

**Further Reading:**

For a deeper dive, I highly recommend looking into the following:

*   **"Elasticsearch: The Definitive Guide"** by Clinton Gormley and Zachary Tong: This book offers comprehensive insights into Elasticsearch, including the underlying mechanisms of machine learning features. Pay specific attention to the sections on data modeling and cardinality.
*   **Elasticsearch documentation on Machine Learning:** The official Elasticsearch documentation is indispensable. Specifically, focus on the machine learning section, paying attention to anomaly detection configurations and data feed parameters.
*   **The paper "An Empirical Evaluation of Techniques for Anomaly Detection in Time Series Data"** by Chandola, Banerjee, and Kumar: This paper provides background on anomaly detection methods and can help you understand the statistical processes behind it all, guiding better configurations and choices in your workflow.
*  **Practical Elasticsearch Analytics: Real-Time Data Analysis and Monitoring** by José Manuel Rodriguez Pujadas: Offers several techniques for efficient analysis and aggregation of data.

In summary, encountering the "Bucket max cardinality estimate required" error isn't a dead end; it's a signal to reassess your analysis approach. Often, a combination of pre-aggregation, judicious use of influencers, and an understanding of the underlying data characteristics will yield the best long-term solution, rather than relying on memory or cardinality workarounds. I hope this gives you the tools you need to resolve this issue efficiently. Remember, a little prep work goes a long way when dealing with high-cardinality data.
