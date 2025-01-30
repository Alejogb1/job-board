---
title: "How can I access Kafka record ingestion timestamps in Apache Beam?"
date: "2025-01-30"
id: "how-can-i-access-kafka-record-ingestion-timestamps"
---
Kafka record ingestion timestamps, often critical for stream processing accuracy and ordering within Apache Beam, are not directly exposed as part of the Kafka message payload when using the standard `KafkaIO` connector. Instead, they are typically found within the record’s metadata, requiring specific handling within your Beam pipeline. My past projects involving real-time analytics using Kafka and Beam have made me acutely aware of these nuances. This response details how I typically address this issue, incorporating timestamp extraction and utilization within a Beam pipeline.

The core challenge stems from the fact that `KafkaIO` primarily returns a `ConsumerRecord` object, encapsulating the message's key, value, and associated metadata. The 'timestamp' field present in `ConsumerRecord` is not necessarily the ingestion time. It is the timestamp assigned by the Kafka producer at the time the record was originally sent, potentially introducing latency and inaccuracies if relying on this for processing order within a Beam pipeline. The desired "ingestion" timestamp refers to the moment the Kafka broker received and committed the message, usually exposed through the message headers or, less commonly, within a specific Kafka topic partition configuration.

To access this ingestion timestamp, the typical approach is to leverage the message headers provided by Kafka, provided the Kafka broker is configured to include these. If the Kafka broker settings expose ingestion time as a field within the record headers (e.g. "ingestion_timestamp"), a `DoFn` must be crafted to extract this data. If ingestion time is not in headers or is a specific attribute of the record that must be derived from the data itself, a similarly implemented `DoFn` is employed. The crucial step involves mapping the raw record to a `KV<String, Long>` pair, using the key for later processing and the long for the ingestion timestamp.

Here's a breakdown of the implementation through several code examples, each building upon the previous, increasing in complexity.

**Example 1: Basic timestamp extraction from headers (assuming 'ingestion_timestamp' is present)**

```java
import org.apache.beam.sdk.transforms.DoFn;
import org.apache.beam.sdk.values.KV;
import org.apache.kafka.clients.consumer.ConsumerRecord;
import java.nio.ByteBuffer;

public class ExtractIngestionTimestamp extends DoFn<ConsumerRecord<String, String>, KV<String, Long>> {
    @ProcessElement
    public void processElement(@Element ConsumerRecord<String, String> record, OutputReceiver<KV<String, Long>> out) {
      byte[] timestampBytes = record.headers().lastHeader("ingestion_timestamp").value();
      if (timestampBytes != null) {
          long timestamp = ByteBuffer.wrap(timestampBytes).getLong();
          out.output(KV.of(record.key(), timestamp));
      }
    }
}
```

*Commentary:* This `DoFn` demonstrates a basic extraction scenario. It retrieves the header named "ingestion_timestamp," extracts the byte array representing a long value, converts it to a `long`, and creates a `KV<String, Long>`, outputting it to the next stage of the pipeline. The record key is preserved for identification purposes in subsequent steps. Error handling (e.g., for cases where the header is missing or not a valid long) would typically be required in production use, but is omitted here for brevity. This approach assumes little to no processing within the header and assumes that the timestamp is directly available.

**Example 2: Timestamp Extraction with error handling and timestamp formatting**

```java
import org.apache.beam.sdk.transforms.DoFn;
import org.apache.beam.sdk.values.KV;
import org.apache.kafka.clients.consumer.ConsumerRecord;
import java.nio.ByteBuffer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class ExtractIngestionTimestampEnhanced extends DoFn<ConsumerRecord<String, String>, KV<String, Long>> {

    private static final Logger LOG = LoggerFactory.getLogger(ExtractIngestionTimestampEnhanced.class);
    private static final String TIMESTAMP_HEADER = "ingestion_timestamp";


    @ProcessElement
    public void processElement(@Element ConsumerRecord<String, String> record, OutputReceiver<KV<String, Long>> out) {
        try {
            if(record.headers().lastHeader(TIMESTAMP_HEADER) != null){
                byte[] timestampBytes = record.headers().lastHeader(TIMESTAMP_HEADER).value();
                 if(timestampBytes == null){
                   LOG.warn("Found " + TIMESTAMP_HEADER + " header, but is empty for record: " + record.key());
                   return;
                }
                long timestamp = ByteBuffer.wrap(timestampBytes).getLong();
                out.output(KV.of(record.key(), timestamp));
            } else {
              LOG.warn("Missing " + TIMESTAMP_HEADER + " header in record: " + record.key());
           }

        } catch (Exception e) {
            LOG.error("Error processing record: " + record.key() + ", reason:" + e.getMessage());
        }
    }
}
```

*Commentary:* This example incorporates logging and basic error handling. It explicitly checks if the specified header exists before attempting to extract the timestamp, logs any missing headers, and encapsulates the entire extraction within a try-catch block for exception handling. If a header is present but is empty, that scenario is also handled with a warning. This represents a more robust approach suitable for production pipelines where missing or malformed data is a common occurrence. The header name has been externalized as a constant for easier maintenance. This example addresses some limitations of the first example by providing better visibility and maintainability.

**Example 3: Deriving Timestamp from record value**

```java
import org.apache.beam.sdk.transforms.DoFn;
import org.apache.beam.sdk.values.KV;
import org.apache.kafka.clients.consumer.ConsumerRecord;
import java.time.Instant;
import org.json.JSONObject;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class ExtractTimestampFromValue extends DoFn<ConsumerRecord<String, String>, KV<String, Long>> {

  private static final Logger LOG = LoggerFactory.getLogger(ExtractTimestampFromValue.class);
  private static final String TIMESTAMP_FIELD = "record_time"; // field present in value that contains timestamp


  @ProcessElement
    public void processElement(@Element ConsumerRecord<String, String> record, OutputReceiver<KV<String, Long>> out) {
        try{
            JSONObject jsonObject = new JSONObject(record.value());
            if(jsonObject.has(TIMESTAMP_FIELD)){
              String timestampString = jsonObject.getString(TIMESTAMP_FIELD);
               long timestamp = Instant.parse(timestampString).toEpochMilli();
               out.output(KV.of(record.key(), timestamp));
           } else {
              LOG.warn("Missing " + TIMESTAMP_FIELD + " in record: " + record.key());
            }
        } catch(Exception e){
          LOG.error("Error processing record: " + record.key() + ", reason:" + e.getMessage());
      }
  }
}
```

*Commentary:* In situations where the ingestion timestamp is not available within the Kafka headers but contained within the record value itself (in this case, assuming it is in JSON format), a different extraction strategy becomes necessary. This example demonstrates extracting a timestamp from the record value, assuming it is stored as a JSON string with a 'record_time' field formatted as an ISO-8601 date time string.  The extraction process involves parsing the JSON string, retrieving the timestamp string value, converting it into an `Instant`, and finally into an epoch milliseconds representation. As with previous examples, error handling and logging are included to enhance the robustness of the `DoFn`. A similar implementation using custom parsing would be implemented should the timestamp be present in a different format.

Following timestamp extraction using one of the methods above (or a derived variation, depending on data format and Kafka configuration), the extracted `KV<String, Long>` data can be utilized for various time-based operations, including windowing, event time processing, or ordering data within the pipeline using the `withTimestampPolicy` function available on `PCollection` instances.

**Resource Recommendations:**

To further enhance understanding and implement this within your projects, I recommend reviewing documentation on the following:
* Apache Beam documentation on windowing and time-based operations
* Kafka client library documentation, specifically around `ConsumerRecord` and record headers
* General documentation on handling date and time formats in Java

These resources should help solidify your understanding of handling record ingestion timestamps within Apache Beam pipelines. Always ensure to tailor your implementation to the specifics of your Kafka setup and data structure.
