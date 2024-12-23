---
title: "Why are AWS Aurora Serverless failovers happening with WSO2 API Manager?"
date: "2024-12-23"
id: "why-are-aws-aurora-serverless-failovers-happening-with-wso2-api-manager"
---

Okay, let's tackle this. I've seen this particular headache a few times, usually when the architecture seems solid on paper but real-world load throws things into disarray. The combination of WSO2 API Manager and AWS Aurora Serverless, especially when experiencing failovers, isn’t as straightforward as one might hope. The core issue often boils down to how connection pooling, resource scaling, and database configuration interact, or rather, *don’t* interact perfectly.

First, let’s unpack what’s likely happening. Aurora Serverless, by its nature, automatically scales resources based on demand, including the underlying compute capacity and memory. This is fantastic for cost optimization, but it introduces variability in the database endpoint itself as it adds or removes resources behind the scenes. While these transitions are designed to be transparent, WSO2 API Manager, particularly if not configured optimally, can be caught off guard.

The biggest problem revolves around persistent connections and the API Manager's internal connection pool. WSO2 API Manager, like many Java-based applications, maintains a pool of database connections for performance reasons. When Aurora Serverless scales down, it might terminate some of these established connections, and if the connection pool isn't robust enough to handle these disconnects or reconnect to the new endpoint quickly, you’re in for trouble. Instead of graceful disconnection and reconnection, you experience failed database transactions, error responses, and ultimately, the perception of a failover at the application layer. To Aurora itself, it's not a failover in the traditional sense; the underlying database service remains available. From the perspective of the API Manager, however, these connection drops look exactly like a failure.

Let’s illustrate with a scenario I encountered at a past project. We were dealing with a relatively high request volume to a specific API exposed via WSO2. We initially configured WSO2 to use the standard JDBC driver without diving deep into connection pool tuning. We saw periodic "failovers," which always coincided with serverless database scaling events. We would see SQLException exceptions like “connection reset by peer” or “connection closed” in the WSO2 logs.

The key realization here was that WSO2’s default connection pool settings were not optimal for the dynamic environment of Aurora Serverless. The maxIdle setting was too high, causing inactive connections to persist indefinitely. When Aurora scaled down, these stale connections would be invalidated, leading to connection errors instead of automatic reconnection.

Here's how we approached the solution, step by step:

**Step 1: Connection Pool Tuning**

We began by adjusting the connection pool settings within the WSO2 API Manager datasources configuration. This configuration resides in `repository/conf/datasources/*.xml`. We made the following changes, specifically targeting the `<DataSource>` element used for the database connection.

```xml
<dataSource>
  <definition type="RDBMS">
    <configuration>
        <url>jdbc:mysql://your-aurora-endpoint/your-database</url>
        <username>your-username</username>
        <password>your-password</password>
        <driverClassName>com.mysql.cj.jdbc.Driver</driverClassName>
        <maxActive>100</maxActive>
        <minIdle>10</minIdle>
        <maxIdle>20</maxIdle>
        <testOnBorrow>true</testOnBorrow>
        <validationQuery>SELECT 1</validationQuery>
        <timeBetweenEvictionRunsMillis>30000</timeBetweenEvictionRunsMillis>
        <minEvictableIdleTimeMillis>60000</minEvictableIdleTimeMillis>
    </configuration>
  </definition>
</dataSource>
```

Notice the critical settings:

*   `maxActive`: The maximum number of active connections. We needed enough capacity for expected peak loads.
*   `minIdle`: The minimum number of idle connections to maintain. This ensures faster response times by not constantly creating new connections.
*   `maxIdle`: The maximum number of idle connections to retain. By lowering this from the default setting, we encouraged the pool to discard connections that were likely to be invalid after a scaling event, making room for new, valid connections.
*   `testOnBorrow`: This instructs the pool to validate connections before use, catching broken connections.
*   `validationQuery`: A simple query to validate the connection's integrity.
*   `timeBetweenEvictionRunsMillis`: The interval at which the idle connection remover runs.
*   `minEvictableIdleTimeMillis`: The minimum time a connection can sit idle before being evicted.

**Step 2: Connection Timeout Adjustments**

Another critical step is adjusting connection timeouts at the JDBC driver level. WSO2 utilizes the `mysql-connector-java` in this case, and specific parameters in the database connection URL can drastically affect how gracefully reconnection attempts are handled.

```java
jdbc:mysql://your-aurora-endpoint/your-database?connectTimeout=3000&socketTimeout=10000&autoReconnect=true&failOverReadOnly=false&serverTimezone=UTC
```

*   `connectTimeout`: The maximum time (in milliseconds) to wait for a connection attempt.
*   `socketTimeout`: The maximum time (in milliseconds) to wait for data from a socket.
*   `autoReconnect`: This tells the driver to attempt a reconnection if the connection breaks. While it's debated, we found it crucial for mitigating issues.
*   `failOverReadOnly`: We set this to `false` as we wanted WSO2 to have full read-write capabilities.
*   `serverTimezone=UTC`: Setting timezones can eliminate timezone conflicts in your system.

**Step 3: Implementing Exponential Backoff**

In situations where the driver’s built-in auto-reconnect fails or if the scaling transition takes longer, incorporating an exponential backoff retry mechanism on the application side adds resilience. This is typically implemented within your data access layer, if WSO2 API Manager application extensions are utilized. Here is a simplified pseudo-code example of how it might look:

```java
public void executeQueryWithRetry(String sql) {
   int maxRetries = 5;
   int baseDelay = 500; // milliseconds

   for (int retry = 0; retry < maxRetries; retry++) {
       try {
          //Establish connection and execute query
          executeSQL(sql);
          return; //Success!
       } catch (SQLException e) {
           if (retry == maxRetries - 1){
                throw e; // Max retries hit, throw the exception
            }
            int delay = (int) (baseDelay * Math.pow(2, retry));
           try {
                Thread.sleep(delay); // Exponential backoff
            } catch (InterruptedException ignored){
            }

          System.out.println("Retrying database operation. Attempt: " + (retry + 1) + ", Delay: " + delay + "ms. Exception: " + e.getMessage());
       }
    }
}
```

This java example demonstrates a very basic retry logic that could be implemented within a WSO2 custom mediator or other extension.

Implementing these adjustments dramatically improved the system's resilience to Aurora Serverless scaling events. The key takeaway is to treat the dynamic nature of serverless databases with explicit connection management in your application configurations.

For further reading and deeper understanding of these concepts, I would highly recommend:

1.  **"Java Concurrency in Practice" by Brian Goetz:** This is a cornerstone text for understanding concurrency and connection pool implementations, crucial for correctly configuring WSO2's environment.
2.  **"Designing Data-Intensive Applications" by Martin Kleppmann:** Provides valuable insight into building reliable systems, with detailed coverage on database interactions and strategies for handling failures.
3.  **The official documentation of your JDBC driver** : The MySQL Connector documentation, for example, includes extensive details on connection parameters and best practices.

These resources should provide the necessary theoretical background and practical knowledge to tackle similar issues and better understand the interplay of applications with dynamically scaled database systems.
