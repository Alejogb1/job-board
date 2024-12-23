---
title: "Why am I getting a Failover exception with AWS Mysql Aurora Serverless V2 and WSO2 API Manager?"
date: "2024-12-23"
id: "why-am-i-getting-a-failover-exception-with-aws-mysql-aurora-serverless-v2-and-wso2-api-manager"
---

Alright, let's unpack this failover exception situation you're encountering with AWS Aurora Serverless v2 and WSO2 API Manager. I've seen this pattern emerge a few times in my years working with distributed systems, and it often boils down to a subtle interplay between how these two technologies handle transient connection disruptions.

Initially, when aurora serverless v2 was relatively new, we ran into similar issues integrating it with our middleware services, including a specific iteration of our api gateway, which had a similar underlying connection management model to wso2 apim. These weren't straightforward database errors; it wasn't as simple as bad credentials. Instead, it manifested as these persistent failover exceptions, leading to downstream application instability, especially under load. The core problem, as we eventually discovered, was a combination of two main factors: improper connection pool configuration on the api manager side and a misunderstanding of how aurora serverless v2 handles scaling and failover events.

First, let's consider the connection pooling aspect within wso2 apim. When your apim nodes connect to the backend mysql database, they typically rely on connection pools to reduce the overhead of creating and closing database connections repeatedly. Now, these pools, while efficient, need to be configured correctly, especially concerning maximum pool size, connection timeouts, and the mechanism used to validate connections. A naive approach to pool configuration, with aggressive max pool sizes, combined with connection leakages within the application layer can lead to these exceptions. The underlying problem is that during a serverless scaling operation or a failover (even a brief one), aurora serverless v2 might momentarily close existing connections. If the connection pool isn’t aware or isn’t configured to gracefully handle this, it can throw a failover exception. The api manager sees it as a catastrophic failure because the pool becomes unusable, requiring some form of reset.

Let’s dive into the first critical piece: robust connection validation. Often overlooked, this mechanism ensures that connections in the pool are still valid before being re-used. The default validation query used by the connection pool may be insufficient for a serverless environment. In our earlier experience, a simple `select 1;` wasn't catching the closed connections during scaling operations.

Here’s a basic example of a connection pool configuration, typical of what you might find in a datasources.xml file used in the apim context:

```xml
<datasource>
    <name>apim_db</name>
    <definition>
      <type>javax.sql.DataSource</type>
        <driverClassName>com.mysql.cj.jdbc.Driver</driverClassName>
      <url>jdbc:mysql://your-aurora-endpoint:3306/apim_db</url>
      <username>your_user</username>
      <password>your_password</password>
      <properties>
           <property name="initialSize">5</property>
           <property name="maxActive">20</property>
           <property name="minIdle">5</property>
           <property name="maxWait">10000</property>
           <!-- Insufficient validation query, this is where the problem begins -->
           <property name="validationQuery">select 1;</property>
           <property name="testOnBorrow">true</property>
           <property name="testWhileIdle">true</property>
           <property name="timeBetweenEvictionRunsMillis">30000</property>
           <property name="minEvictableIdleTimeMillis">60000</property>
      </properties>
    </definition>
</datasource>
```
This demonstrates the issue, the `validationQuery` is just a simple 'select 1', which doesn't really check the connection's validity in the serverless context when a connection is terminated by Aurora.

To address this, you need a more robust query. Here's how we adapted ours, ensuring it checks both the connection state and some minimal level of functionality, which reduces false positive. This snippet demonstrates a simple improvement to use a more robust connection validation query:
```xml
<datasource>
    <name>apim_db</name>
    <definition>
      <type>javax.sql.DataSource</type>
        <driverClassName>com.mysql.cj.jdbc.Driver</driverClassName>
      <url>jdbc:mysql://your-aurora-endpoint:3306/apim_db</url>
      <username>your_user</username>
      <password>your_password</password>
      <properties>
           <property name="initialSize">5</property>
           <property name="maxActive">20</property>
           <property name="minIdle">5</property>
           <property name="maxWait">10000</property>
           <!-- Improved Validation Query that catches closed connections  -->
           <property name="validationQuery">select connection_id();</property>
           <property name="testOnBorrow">true</property>
           <property name="testWhileIdle">true</property>
           <property name="timeBetweenEvictionRunsMillis">30000</property>
           <property name="minEvictableIdleTimeMillis">60000</property>
      </properties>
    </definition>
</datasource>
```

This adjusted `validationQuery` using `select connection_id();` forces a check of the underlying database connection, and if it is closed, the validation will fail. Also, `testOnBorrow=true` and `testWhileIdle=true` ensures that validation is performed both before a connection is borrowed and while connections are idle in the pool, preventing stale connections from being returned. This significantly reduces the chances of encountering the failover exception.

The second critical element involves properly configuring the *connection retry strategy*. While connection validation is effective, some interruptions may still momentarily occur. Implementing a robust retry mechanism at the application level (within the wso2 apim in this case) can smooth over these transient errors. A simple retry strategy with exponential backoff can make the difference, and if the apim uses a framework which supports it, should be implemented at the connection level.

Let’s look at an example of how a java-based retry mechanism with the popular `com.github.rholder:guava-retrying` library might be configured around a database operation:
```java
import com.github.rholder.retry.*;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.concurrent.Callable;
import java.util.concurrent.TimeUnit;

public class RetryDatabaseOperation {
    public static void main(String[] args) {
        Retryer<ResultSet> retryer = RetryerBuilder.<ResultSet>newBuilder()
               .retryIfExceptionOfType(SQLException.class)
                .withWaitStrategy(WaitStrategies.exponentialWait(1000, TimeUnit.MILLISECONDS))
                .withStopStrategy(StopStrategies.stopAfterAttempt(5))
                .build();

        Callable<ResultSet> databaseOperation = new Callable<ResultSet>() {
            @Override
            public ResultSet call() throws Exception {
               Connection conn = null;
               PreparedStatement stmt = null;
               ResultSet rs = null;

                try {
                    conn = DriverManager.getConnection("jdbc:mysql://your-aurora-endpoint:3306/apim_db", "your_user", "your_password");
                    stmt = conn.prepareStatement("SELECT 1;");
                    rs = stmt.executeQuery();
                    return rs;
                }
                catch (SQLException e) {
                    if (conn != null) {
                       try {conn.close();} catch(Exception ex){}
                    }
                    if (stmt != null){
                        try {stmt.close();} catch (Exception ex) {}
                    }
                    throw e;
                } finally {
                    if (conn != null) {
                        try {conn.close();} catch(Exception ex){}
                     }
                    if (stmt != null){
                        try {stmt.close();} catch (Exception ex) {}
                    }
                }

            }
        };


        try {
           ResultSet result  = retryer.call(databaseOperation);
           System.out.println("Query executed successfully. result :" + result.toString());

        } catch(RetryException e) {
            System.err.println("database operation failed after all retries: " + e.getMessage());
        } catch(Exception e) {
           System.err.println("database operation failed with exception:" + e.getMessage());
        }
    }
}
```
This snippet shows that even in the case of a failure (SQLException), the database connection operation will be retried up to 5 times, with exponential backoff between retries. This demonstrates a retry mechanism, which does not directly relate to apim but showcases what needs to be implemented within the apim context or framework where database connection is used. This, when properly applied, can mitigate the impact of temporary connection interruptions.

For further learning, I'd highly recommend delving into the documentation of your connection pool library (e.g., HikariCP, DBCP), the official aws documentation on aurora serverless v2, specifically the best practices for connection management, and “Release It!: Design and Deploy Production-Ready Software” by Michael T. Nygard. Additionally, “Patterns of Enterprise Application Architecture” by Martin Fowler, specifically the sections on connection pooling and retry logic, can provide valuable insights.

In short, these failover exceptions with Aurora Serverless v2 and WSO2 API Manager are often a symptom of inadequate connection management rather than a fundamental flaw. Focusing on robust connection validation, and a well-defined retry policy can provide the needed stability. It's an iterative process, but these steps, I've found, can significantly improve the overall resilience of your system.
