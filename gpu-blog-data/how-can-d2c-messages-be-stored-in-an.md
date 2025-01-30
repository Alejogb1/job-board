---
title: "How can D2C messages be stored in an Azure SQL database?"
date: "2025-01-30"
id: "how-can-d2c-messages-be-stored-in-an"
---
Direct device-to-cloud (D2C) messages from IoT devices present a significant challenge for persistent storage, particularly when dealing with high ingestion rates and schema variations. I’ve directly managed systems processing millions of these messages daily, and a well-configured Azure SQL database can be a suitable option provided certain architectural decisions are made. The core challenge revolves around efficiently ingesting and querying these typically unstructured or semi-structured JSON messages. Simply inserting raw JSON into a single column isn't sustainable for any serious analytical workload; this requires pre-processing, normalization, and careful consideration for indexing.

My approach is to treat the incoming D2C messages as a stream of events, not simple records. We need a pipeline to transform these events into a structured format that an SQL database can efficiently handle. I've found the most robust pattern uses Azure Functions or Logic Apps as an intermediary between IoT Hub and the SQL database. These services can perform the necessary transformation before data is persisted.

First, the raw D2C message, received as a JSON string, will usually contain metadata from IoT Hub (e.g., device ID, message enqueue time), along with the actual payload from the device. We must separate these elements. The metadata should be stored in columns like `DeviceId`, `EnqueuedTime`, and potentially `MessageId`, which allow for quick filtering and troubleshooting. The payload, often application-specific, needs to be normalized. If the payload is a consistent JSON structure, you should map its properties to dedicated columns in the SQL table. This approach facilitates performant querying. However, the payload often evolves, with new sensors or properties added over time. This warrants consideration for handling schema drift. One approach is to use a combination of fixed and dynamic columns. Fixed columns will always map to known properties (e.g., temperature, humidity), while a separate column (e.g., `ExtendedProperties`) can store the remaining JSON payload as a `NVARCHAR(MAX)` column. This allows for both efficient querying on core data and flexibility for evolving schemas. Another viable, though more complex, approach is to use an event sourcing pattern with multiple tables designed to capture the entire event history with timestamped changes, requiring more query complexity but affording significant auditing capabilities.

Here are three code examples that demonstrate how such a pipeline can be implemented, all using C# as the language for the Azure function:

**Example 1: Basic data extraction and insertion with fixed columns:**

```csharp
using System;
using Microsoft.Azure.WebJobs;
using Microsoft.Azure.WebJobs.Extensions.EventHubs;
using Microsoft.Extensions.Logging;
using System.Data.SqlClient;
using Newtonsoft.Json.Linq;

public static class D2CDataIngestBasic
{
    [FunctionName("D2CDataIngestBasic")]
    public static void Run(
        [EventHubTrigger("iot-hub-name", Connection = "EventHubConnectionString")]
        string myEventHubMessage, ILogger log)
    {
        log.LogInformation($"C# Event Hub trigger function processed a message: {myEventHubMessage}");

        try
        {
           JObject messageJson = JObject.Parse(myEventHubMessage);
           string deviceId = messageJson.Value<string>("systemProperties.iothub-connection-device-id");
           DateTime enqueuedTime = messageJson.Value<DateTime>("systemProperties.iothub-enqueuedtime");
           double temperature = messageJson["body"].Value<double>("temperature");
           double humidity = messageJson["body"].Value<double>("humidity");

           string connectionString = Environment.GetEnvironmentVariable("SqlConnectionString");
           using (SqlConnection connection = new SqlConnection(connectionString))
           {
                connection.Open();
                string insertStatement = "INSERT INTO DeviceTelemetry (DeviceId, EnqueuedTime, Temperature, Humidity) VALUES (@DeviceId, @EnqueuedTime, @Temperature, @Humidity)";
                using (SqlCommand command = new SqlCommand(insertStatement, connection))
                {
                    command.Parameters.AddWithValue("@DeviceId", deviceId);
                    command.Parameters.AddWithValue("@EnqueuedTime", enqueuedTime);
                    command.Parameters.AddWithValue("@Temperature", temperature);
                    command.Parameters.AddWithValue("@Humidity", humidity);
                    command.ExecuteNonQuery();
                }
            }
        }
        catch (Exception ex)
        {
            log.LogError($"Error processing message: {ex.Message}");
        }

    }
}
```

This example demonstrates how to pull message metadata and fixed payload properties from the JSON message and insert them into a SQL table with matching columns (`DeviceId`, `EnqueuedTime`, `Temperature`, `Humidity`). The connection string to the SQL database is pulled from environment variables. This example assumes a fairly consistent JSON structure in the "body". The use of `JObject` from `Newtonsoft.Json` provides the parsing needed.

**Example 2: Handling dynamic data with a `ExtendedProperties` column:**

```csharp
using System;
using Microsoft.Azure.WebJobs;
using Microsoft.Azure.WebJobs.Extensions.EventHubs;
using Microsoft.Extensions.Logging;
using System.Data.SqlClient;
using Newtonsoft.Json.Linq;

public static class D2CDataIngestExtended
{
    [FunctionName("D2CDataIngestExtended")]
    public static void Run(
        [EventHubTrigger("iot-hub-name", Connection = "EventHubConnectionString")]
        string myEventHubMessage, ILogger log)
    {
        log.LogInformation($"C# Event Hub trigger function processed a message: {myEventHubMessage}");

        try
        {
           JObject messageJson = JObject.Parse(myEventHubMessage);
           string deviceId = messageJson.Value<string>("systemProperties.iothub-connection-device-id");
           DateTime enqueuedTime = messageJson.Value<DateTime>("systemProperties.iothub-enqueuedtime");
           
           JObject body = (JObject)messageJson["body"];
           double? temperature = body.Value<double?>("temperature");
           double? humidity = body.Value<double?>("humidity");
           
            //remove fixed properties
            body.Remove("temperature");
            body.Remove("humidity");


           string extendedProperties = body.ToString();

           string connectionString = Environment.GetEnvironmentVariable("SqlConnectionString");
           using (SqlConnection connection = new SqlConnection(connectionString))
           {
                connection.Open();
                string insertStatement = "INSERT INTO DeviceTelemetry (DeviceId, EnqueuedTime, Temperature, Humidity, ExtendedProperties) VALUES (@DeviceId, @EnqueuedTime, @Temperature, @Humidity, @ExtendedProperties)";
                using (SqlCommand command = new SqlCommand(insertStatement, connection))
                {
                    command.Parameters.AddWithValue("@DeviceId", deviceId);
                    command.Parameters.AddWithValue("@EnqueuedTime", enqueuedTime);
                    command.Parameters.AddWithValue("@Temperature", (object)temperature ?? DBNull.Value);
                    command.Parameters.AddWithValue("@Humidity", (object)humidity ?? DBNull.Value);
                    command.Parameters.AddWithValue("@ExtendedProperties", extendedProperties);
                    command.ExecuteNonQuery();
                }
            }
        }
        catch (Exception ex)
        {
            log.LogError($"Error processing message: {ex.Message}");
        }

    }
}
```

This enhanced example demonstrates how to extract both the known fields (temperature and humidity), while storing the rest of the body as JSON into the `ExtendedProperties` column. The nullable types (`double?`) ensure flexibility if those properties are missing. The removal of the parsed fields from the `JObject` body ensures the `ExtendedProperties` column doesn't duplicate data. This approach accommodates evolving device payloads, allowing for future properties without database schema changes.

**Example 3: Handling null values and database `NULL`s**

```csharp
using System;
using Microsoft.Azure.WebJobs;
using Microsoft.Azure.WebJobs.Extensions.EventHubs;
using Microsoft.Extensions.Logging;
using System.Data.SqlClient;
using Newtonsoft.Json.Linq;

public static class D2CDataIngestNullHandling
{
    [FunctionName("D2CDataIngestNullHandling")]
    public static void Run(
        [EventHubTrigger("iot-hub-name", Connection = "EventHubConnectionString")]
        string myEventHubMessage, ILogger log)
    {
        log.LogInformation($"C# Event Hub trigger function processed a message: {myEventHubMessage}");

        try
        {
           JObject messageJson = JObject.Parse(myEventHubMessage);
           string deviceId = messageJson.Value<string>("systemProperties.iothub-connection-device-id");
           DateTime enqueuedTime = messageJson.Value<DateTime>("systemProperties.iothub-enqueuedtime");
           
           JObject body = (JObject)messageJson["body"];
           double? temperature = body.Value<double?>("temperature");
           string status = body.Value<string>("status");


           string connectionString = Environment.GetEnvironmentVariable("SqlConnectionString");
           using (SqlConnection connection = new SqlConnection(connectionString))
           {
                connection.Open();
                string insertStatement = "INSERT INTO DeviceTelemetry (DeviceId, EnqueuedTime, Temperature, Status) VALUES (@DeviceId, @EnqueuedTime, @Temperature, @Status)";
                using (SqlCommand command = new SqlCommand(insertStatement, connection))
                {
                    command.Parameters.AddWithValue("@DeviceId", deviceId);
                    command.Parameters.AddWithValue("@EnqueuedTime", enqueuedTime);
                     command.Parameters.AddWithValue("@Temperature", (object)temperature ?? DBNull.Value);
                    command.Parameters.AddWithValue("@Status", (object)status ?? DBNull.Value);
                    command.ExecuteNonQuery();
                }
            }
        }
        catch (Exception ex)
        {
            log.LogError($"Error processing message: {ex.Message}");
        }

    }
}
```

This example explicitly demonstrates handling potentially null values. By assigning the value using the null coalescing operator and checking if null, we can explicitly insert database `NULL` values. This allows us to properly record the absence of data in our database without producing errors if a sensor reading is missing or a status property is not returned.

For those delving deeper, I would strongly recommend investigating resources on database schema design for time-series data, focusing on techniques like partitioning and columnstore indexing available in Azure SQL. Microsoft's official documentation for Azure Functions, Azure Event Hubs, and Azure SQL is also crucial. Look for articles on handling JSON data in SQL databases and strategies for schema evolution. I would also investigate data warehousing techniques with Azure Synapse if the analytics workload expands beyond the capabilities of Azure SQL.
