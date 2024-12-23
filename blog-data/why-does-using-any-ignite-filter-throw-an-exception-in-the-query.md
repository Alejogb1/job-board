---
title: "Why does using any Ignite filter throw an exception in the query?"
date: "2024-12-23"
id: "why-does-using-any-ignite-filter-throw-an-exception-in-the-query"
---

Alright, let’s tackle this. I've seen this issue crop up more times than I care to remember, typically when someone’s just getting their hands dirty with Apache Ignite’s query capabilities. The frustrating “filter throws an exception” situation almost always boils down to how Ignite handles data distribution and processing in a distributed environment, and it’s not immediately obvious why a seemingly innocuous filter can suddenly break things.

In my past experience, particularly with a large-scale financial trading platform, we encountered this almost daily. The platform relied heavily on Ignite’s ability to process large volumes of market data in real-time. We had a complex network of caches, each holding different facets of market information, and queries that spanned multiple nodes were commonplace. The situation often arose when developers, freshly introduced to Ignite, attempted to filter query results using custom methods without fully understanding the distributed implications. So, it wasn't some bug within Ignite; rather, it was often the way we were using it.

The core problem is that Ignite distributes data across multiple nodes, dividing the data space for scalability and performance. This distribution happens by hashing the keys of the data. When you execute a query, the query execution process often involves processing data locally on each node before combining or aggregating the results. Therefore, any filter you use has to be *serializable* and *executable* within each node's JVM.

Now, what throws the exception you’re probably seeing? It's usually one of two things, or a combination of both: the use of non-serializable objects within your filter predicates or, equally common, invoking methods that are not accessible within the server-side execution context. These methods could be anything from accessing instance variables which are not part of the data being processed, to trying to interact with resources outside of the JVM the query is running on, or the use of lambdas without proper serialization consideration.

Let me illustrate this with a couple of scenarios and then we'll get into code examples. First, imagine you have a `Trader` object with a `calculateRiskScore()` method, that accesses local system properties or some other resource on your local machine, but this method is not packaged up and shipped along with your query to the various ignite nodes. When you try to filter using this logic the query fails due to serialization problems, but also due to the lack of necessary resources on the server node. Secondly, you might be using anonymous inner classes to filter the results with a logic that is dependent on the *local context* of the application. These inner classes are not automatically serializable and this results in an `IllegalArgumentException` or `NotSerializableException` on the server node.

Let's explore these issues with some practical code examples. We'll use a simplified version of the problems we saw back in our trading platform days. The following examples assume a basic setup with an Ignite instance and a cache defined.

**Example 1: Non-Serializable Filter**

Let’s say you have a `Trade` class and a filter that uses a method dependent on the surrounding environment.

```java
import org.apache.ignite.Ignite;
import org.apache.ignite.IgniteCache;
import org.apache.ignite.Ignition;
import org.apache.ignite.cache.query.SqlFieldsQuery;
import java.io.Serializable;
import java.util.List;

public class FilterExample1 {

    public static class Trade implements Serializable {
        private String symbol;
        private double price;

        public Trade(String symbol, double price) {
            this.symbol = symbol;
            this.price = price;
        }
        public String getSymbol() {
            return symbol;
        }

        public double getPrice() {
            return price;
        }

    }
     private static boolean isValidTrade(Trade trade){
        return System.currentTimeMillis() > 1678886400000L; // some arbitrary time check
     }

    public static void main(String[] args) {
         try(Ignite ignite = Ignition.start()){
            IgniteCache<Long, Trade> cache = ignite.getOrCreateCache("trades");

            cache.put(1L, new Trade("AAPL", 170.0));
            cache.put(2L, new Trade("GOOG", 2700.0));
            cache.put(3L, new Trade("TSLA", 1000.0));

            SqlFieldsQuery query = new SqlFieldsQuery("SELECT symbol, price FROM Trade where ?");
             query.setArgs((Serializable)(trade -> isValidTrade((Trade)trade)));
            try {
                List<List<?>> queryResult = cache.query(query).getAll();
                 queryResult.forEach(row -> System.out.println("Symbol: " + row.get(0) + ", Price: " + row.get(1)));

            }catch (Exception e){
                 System.out.println("Error during query:" + e.getMessage());
            }

        }
    }
}
```

This code will most likely throw an exception since the lambda used within the query is not designed for distributed processing because it uses `System.currentTimeMillis()`. The `isValidTrade()` is also not part of the data being processed or available on the server node. This highlights the pitfall of using non-serializable predicates in queries. It’s a common error I've seen countless times.

**Example 2: Using a Serializable Predicate Class**

To fix this, you need to encapsulate your filtering logic into a serializable class that can be executed by the Ignite nodes.

```java
import org.apache.ignite.Ignite;
import org.apache.ignite.IgniteCache;
import org.apache.ignite.Ignition;
import org.apache.ignite.cache.query.SqlFieldsQuery;
import org.apache.ignite.cache.query.annotations.QuerySqlField;
import org.apache.ignite.cache.query.QueryPredicate;

import javax.cache.Cache;
import java.io.Serializable;
import java.util.List;

public class FilterExample2 {

    public static class Trade implements Serializable {
        @QuerySqlField
        private String symbol;
        @QuerySqlField
        private double price;
        @QuerySqlField
        private long timestamp;

        public Trade(String symbol, double price, long timestamp) {
            this.symbol = symbol;
            this.price = price;
            this.timestamp = timestamp;
        }

        public String getSymbol() {
            return symbol;
        }
        public double getPrice() {
            return price;
        }
        public long getTimestamp() {
           return timestamp;
        }
    }
    public static class TradePredicate implements Serializable, QueryPredicate<Trade> {
        private final long thresholdTime;
         public TradePredicate(long thresholdTime) {
            this.thresholdTime = thresholdTime;
        }

        @Override
        public boolean apply(Trade trade) {
            return trade.getTimestamp() > thresholdTime;
        }
    }

    public static void main(String[] args) {
        try (Ignite ignite = Ignition.start()) {
            IgniteCache<Long, Trade> cache = ignite.getOrCreateCache("trades");

            cache.put(1L, new Trade("AAPL", 170.0, 1678886400000L + 1000));
            cache.put(2L, new Trade("GOOG", 2700.0, 1678886400000L - 1000));
            cache.put(3L, new Trade("TSLA", 1000.0, 1678886400000L + 2000));

            SqlFieldsQuery query = new SqlFieldsQuery("SELECT symbol, price FROM Trade WHERE ?");
            query.setArgs(new TradePredicate(1678886400000L));
            List<List<?>> queryResult = cache.query(query).getAll();
            queryResult.forEach(row -> System.out.println("Symbol: " + row.get(0) + ", Price: " + row.get(1)));
        }
    }
}
```

Here, the `TradePredicate` class is explicitly serializable and the `apply()` method contains the filtering logic. The timestamp is now part of the data, and the filtering is done against this data, not against any local context resources. This ensures the logic can be distributed across the Ignite cluster, and this is the common method to apply a filtering logic.

**Example 3: Using SQL-based Filtering**

Finally, often the best approach is to utilize SQL syntax with filtering options provided through standard SQL expressions or through parameterized query arguments, eliminating the need to write custom filter classes. This approach benefits from Ignite's optimized SQL processing engine.

```java
import org.apache.ignite.Ignite;
import org.apache.ignite.IgniteCache;
import org.apache.ignite.Ignition;
import org.apache.ignite.cache.query.SqlFieldsQuery;
import org.apache.ignite.cache.query.annotations.QuerySqlField;

import java.io.Serializable;
import java.util.List;

public class FilterExample3 {
    public static class Trade implements Serializable {
        @QuerySqlField
        private String symbol;
        @QuerySqlField
        private double price;
        @QuerySqlField
        private long timestamp;
        public Trade(String symbol, double price, long timestamp) {
            this.symbol = symbol;
            this.price = price;
            this.timestamp = timestamp;
        }
        public String getSymbol() {
            return symbol;
        }

        public double getPrice() {
            return price;
        }

        public long getTimestamp(){
            return timestamp;
        }
    }

    public static void main(String[] args) {
        try (Ignite ignite = Ignition.start()) {
            IgniteCache<Long, Trade> cache = ignite.getOrCreateCache("trades");
            cache.put(1L, new Trade("AAPL", 170.0, 1678886400000L + 1000));
            cache.put(2L, new Trade("GOOG", 2700.0, 1678886400000L - 1000));
            cache.put(3L, new Trade("TSLA", 1000.0, 1678886400000L + 2000));

            long thresholdTime = 1678886400000L;
            SqlFieldsQuery query = new SqlFieldsQuery("SELECT symbol, price FROM Trade WHERE timestamp > ?");
            query.setArgs(thresholdTime);
            List<List<?>> queryResult = cache.query(query).getAll();
            queryResult.forEach(row -> System.out.println("Symbol: " + row.get(0) + ", Price: " + row.get(1)));
        }
    }
}
```

In this example, I've used a standard SQL `WHERE` clause, which allows Ignite to optimize the query execution and it is simpler than custom predicates.

To delve deeper into these concepts, I strongly recommend reading the Apache Ignite documentation on distributed queries and data filtering, paying close attention to the sections about serialization and deployment. The book *Apache Ignite In Action* by Scott Varga and Daniel Takacs also provides detailed explanations and hands-on examples related to these concepts. Further, reviewing the sources of the specific exceptions you're seeing, like `NotSerializableException` is invaluable, as they usually pinpoint exactly which class is the culprit. Finally, digging into the source code of `org.apache.ignite.cache.query` will provide a clearer picture of how Ignite executes queries.

The key takeaway here is to always be mindful of where your code is executing within a distributed environment. Serializing your predicates or utilizing SQL will consistently resolve most "filter throws exception" issues. Remember that Ignite distributes your data, therefore your processing must be distributed too. It's a fundamental shift in thinking when coming from traditional single-node environments, but once understood, it becomes quite straightforward.
