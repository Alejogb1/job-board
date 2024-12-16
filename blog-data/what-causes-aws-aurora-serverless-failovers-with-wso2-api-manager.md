---
title: "What causes AWS Aurora Serverless failovers with WSO2 API Manager?"
date: "2024-12-16"
id: "what-causes-aws-aurora-serverless-failovers-with-wso2-api-manager"
---

Alright, let's tackle this. I’ve seen this exact scenario play out a few times now, particularly in environments pushing the limits of what both Aurora Serverless and WSO2 API Manager can handle. It's rarely a single smoking gun, but rather a confluence of factors that lead to those frustrating failovers. I'll break down the key contributing elements, drawing on my experiences supporting large-scale deployments, and offer some concrete examples to illustrate the points.

Firstly, let's address the inherent nature of aurora serverless. Unlike provisioned instances, it scales compute and memory resources automatically based on demand. This scaling, while convenient, isn't instantaneous. When WSO2 API Manager generates a surge in database requests – perhaps from a sudden influx of API calls or a scheduled background process – aurora serverless has to spin up new resources. This scaling process, while swift, can lead to brief periods where the database is temporarily less responsive or outright unavailable. These short periods of unavailability, if not handled gracefully by the API Manager, can be misconstrued as a failure, triggering its internal failover mechanisms. This is not an aurora failure in the traditional sense; it is a consequence of the scaling process being slower than the immediate request from the application.

A second common culprit lies within the connection management strategies employed by WSO2 API Manager. Poorly configured connection pools or inadequate health check routines can exacerbate problems. For instance, if the connection pool doesn't gracefully handle temporary network interruptions or brief database unavailability during aurora serverless scaling, the API Manager may interpret dropped connections as a complete failure of the database, leading to a failover attempt. These dropped connections often result in persistent connection exhaustion, making things worse. It's crucial to have robust retry mechanisms within the API Manager’s database interaction layer.

Third, and this is often overlooked, is the sheer volume and complexity of the queries originating from WSO2 API Manager. Insufficient indexing, poorly optimized queries, and a lack of query caching can put an excessive load on the database, even under relatively normal conditions. During a scaling event on aurora serverless, these already stressed resources can buckle, triggering the failover. These situations are often more about poorly performing SQL rather than an actual database failure.

Let’s illustrate these points with a few code examples, concentrating on the relevant bits.

**Example 1: Connection Pool Configuration (Illustrative, not specific implementation)**

Imagine a connection pool configuration within the WSO2 environment that lacks adequate resilience. Let’s represent it conceptually using a fictional java-like configuration snippet:

```java
public class ConnectionPoolConfig {
    int maxPoolSize = 10; // Relatively small pool size
    int minPoolSize = 2;
    int connectionTimeout = 30; // Seconds, insufficient for scaling events
    boolean validateOnBorrow = false; // A potential issue, we should validate connections
    RetryStrategy retryStrategy = new RetryStrategy(); //Assume no retry logic in place
}
```

Here, the small `maxPoolSize` and `connectionTimeout` make the system vulnerable. A sudden spike will exhaust the pool. Further, the lack of validation (`validateOnBorrow = false`) could lead to using broken connections. Without a proper retry mechanism in the `RetryStrategy`, it's very easy to trigger an interpretation of a database outage.

**Example 2: Inadequate SQL Query Optimization (Simplified scenario)**

Suppose WSO2 API Manager runs the following hypothetical SQL query frequently:

```sql
-- Poorly Optimized Query
SELECT * FROM api_definitions WHERE api_name LIKE '%search_term%';
```

This query, using a wildcard at the beginning of the `LIKE` operator, prevents the database from using any index on the `api_name` column. As the number of api definitions grows, this query becomes increasingly slow. Combine this with a scaling event on aurora serverless, and the prolonged execution time may well cause the WSO2 API Manager to believe the database is unresponsive. It's not *down* necessarily, but it’s taking too long. A better approach would be:

```sql
-- Optimized Query
SELECT * FROM api_definitions WHERE api_name = 'specific_api_name';
--Or, if we *must* do pattern matches, use:
SELECT * FROM api_definitions WHERE api_name LIKE 'search_term%';
```

The second example shows an equality comparison, which would use an index. The third shows pattern matching from the beginning, which may be indexed, depending on database. The performance difference is massive.

**Example 3: Lack of Retry Logic within API Manager (Conceptual Java)**

Let's assume a simplified method in the API manager that queries the database:

```java
public class ApiManagerDatabase {
    public ApiDefinition fetchApiDefinition(String apiId) {
        try{
          // Code to query DB based on apiId, could use JDBC or some other persistence mechanism.
           return this.databaseAccessor.fetchRecord(apiId);

        } catch (Exception ex) {
            //No proper retry handling here:
            log.error("Error fetching API definition:", ex);
            throw new RuntimeException("Database error, aborting.");
        }
    }
}

```

In this simplified example, a failure, be it connection-related or due to slow query, results in an immediate exception and potentially triggers a failover process in the WSO2 API Manager if not handled correctly by that particular manager's code. Instead, a robust implementation should incorporate a retry with exponential backoff, as shown in example 4.

**Example 4: Proper Retry Mechanism (Conceptual Java)**
This code snippet showcases how retry attempts could be properly addressed when making a database request.

```java
import java.util.concurrent.TimeUnit;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ApiManagerDatabase {
    private static final Logger log = LoggerFactory.getLogger(ApiManagerDatabase.class);
    private final int MAX_RETRIES = 3;
    private final long BASE_DELAY = 100; // milliseconds

    public ApiDefinition fetchApiDefinition(String apiId) {
        int retryCount = 0;
        while (retryCount < MAX_RETRIES) {
            try {
                // Code to query DB based on apiId
                return this.databaseAccessor.fetchRecord(apiId);
            } catch (Exception ex) {
                retryCount++;
                if (retryCount >= MAX_RETRIES){
                    log.error("Maximum retries exceeded while fetching API definition:", ex);
                   throw new RuntimeException("Database error after multiple retries.", ex);
                } else {
                   long delay = BASE_DELAY * (long) Math.pow(2, retryCount);
                   log.warn("Error fetching API definition, retrying in " + delay + "ms: " + ex.getMessage());
                    try {
                        TimeUnit.MILLISECONDS.sleep(delay);
                     } catch (InterruptedException ie){
                        Thread.currentThread().interrupt();
                       log.error("Sleep interrupted during retry: ", ie);
                       throw new RuntimeException("Database retry interrupted", ie);
                     }

                }
            }
        }
        return null; // unreachable, added for compilation
    }
}

```
 This improved method includes retry logic and uses an exponential backoff when an issue is encountered. The `sleep` operation is crucial to avoid hammering the database, potentially worsening things. Additionally, the retries are logged, allowing for better observability.

To summarize, aurora serverless failovers with wso2 api manager are usually caused by one or more of the following: aurora scaling events outpacing application demands, poorly configured connection management within the api manager, and inefficient queries placing excessive load on the database, and a lack of retry strategies. Addressing these points systematically will greatly improve the stability and reliability of the environment.

For further reading and a deeper understanding, I would recommend exploring resources like "Database Internals" by Alex Petrov, particularly chapters focusing on database transaction processing, connection pooling, and query optimization. Also, the official AWS documentation on aurora serverless performance and best practices is indispensable. For a broader understanding of distributed system resilience, “Designing Data-Intensive Applications” by Martin Kleppmann provides excellent insights. I also highly recommend studying the WSO2 API manager documentation in detail, looking for specific settings and recommendations for connection management and database interaction. Finally, the paper on ‘Eventually Consistent Systems’ is a seminal work if we are talking about distributed systems (google for the original paper). These will provide the theoretical basis to truly understand what's happening under the hood, and how to build a more robust architecture.
