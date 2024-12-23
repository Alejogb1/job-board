---
title: "Why are Rails applications hanging on every request?"
date: "2024-12-23"
id: "why-are-rails-applications-hanging-on-every-request"
---

, let’s tackle this. I've seen this particular type of deadlock more times than I’d care to recount, often under the seemingly innocent guise of 'everything was working fine yesterday.' The scenario of a Rails application hanging on every request, manifesting as an eerie silence where responsiveness should be, is frequently rooted in concurrency issues, resource exhaustion, or misconfigured components. It's almost never a single smoking gun, but rather a confluence of factors that, when aligned poorly, bring your application to its knees.

From experience, I recall a rather large e-commerce platform where we had this exact problem. It was a frantic, all-hands-on-deck situation, with customer service getting hammered, and development teams scrambling. The culprit, in that instance, turned out to be a combination of poorly configured database connection pooling and an unchecked background job queue that had spiraled into a resource-hogging vortex. It taught me a lot about anticipating and planning for scale.

So, let's break down the common causes systematically. Primarily, you're facing a situation where requests are being processed too slowly or not being processed at all. Here are a few suspects:

**1. Thread Pool Exhaustion (Web Server/Application Server):**

The heart of most Rails applications beats within a web server (like Puma or Unicorn) or an application server (such as Passenger). These servers use thread pools (or process pools) to handle concurrent requests. If all threads are busy and a new request comes in, it has to wait for a thread to become available. If this wait never ends because threads are constantly processing very long-running tasks, you have a hang. This could be due to slow database queries, external API calls with poor latency, or overly complex logic within your controllers.

To diagnose this, use monitoring tools to observe the thread pool utilization. If you consistently see all threads in use, you have a problem. Increase the size of your thread pool, but this is usually a band-aid solution unless coupled with optimizing the slow-running operations. It’s essential to use a tool that gives you insights into what each thread is doing. For example, using Puma, check the server logs and also monitor resource utilization with tools such as `top` and `htop` on linux to ensure thread usage isn’t overly excessive.

**2. Database Connection Pool Starvation:**

Rails, by default, uses a database connection pool to reuse database connections, avoiding the overhead of repeatedly establishing new connections for each request. However, if your application’s requests consistently require more database connections than available in the pool, threads will wait until a connection becomes free. This waiting translates into the perceived "hang" we’re discussing.

When dealing with database connection issues, the ActiveRecord::Base.connection_pool is particularly relevant. You can often check its size in your `database.yml` file, and monitor the connection pool usage, which is often logged directly or accessible via a status interface of your underlying database. Increasing the size of the connection pool can help. However, you should avoid just increasing these values ad infinitum since that may strain resources on your database server itself. Instead, investigate and reduce the number of required connections by reviewing your active record queries.

Here's a simple example to show how you can inspect the connection pool:

```ruby
# Rails console example

puts ActiveRecord::Base.connection_pool.size  # shows the configured size
puts ActiveRecord::Base.connection_pool.connections.size # Shows the current active connections

puts ActiveRecord::Base.connection_pool.checkout_timeout # Shows the timeout for getting a connection, in seconds

# A simple example that shows the connection pool in action

def perform_with_connection
  ActiveRecord::Base.connection_pool.with_connection do |conn|
    # Use the connection to perform database operations
     puts "Successfully used connection #{conn.object_id}"
    end
end

10.times { perform_with_connection } # run 10 times
```

This snippet shows a simple example of how to inspect and utilize the database connection pool. You can use this inside a rails console to see the configured parameters, current connections and how they are used.

**3. Deadlocks in Application Code:**

Deadlocks happen when two or more threads wait on each other indefinitely. This can occur due to improper locking, mutex usage, or even a complex chain of dependencies where resources are held. While less frequent than connection or thread pool issues, such deadlocks will manifest as a total hang.

Here’s a simplified (and intentionally faulty) example demonstrating a common deadlock scenario:

```ruby
# Example (intentionally has a deadlock)

require 'thread'

mutex1 = Mutex.new
mutex2 = Mutex.new

thread1 = Thread.new do
  mutex1.synchronize do
    puts "Thread 1 acquired mutex1"
    sleep 0.1 # simulate some work
    mutex2.synchronize do # waits on mutex2 and leads to deadlock since thread 2 has it
      puts "Thread 1 acquired mutex2"
    end
  end
  puts "Thread 1 finished"
end

thread2 = Thread.new do
  mutex2.synchronize do
    puts "Thread 2 acquired mutex2"
    sleep 0.1 # simulate some work
    mutex1.synchronize do # waits on mutex1 and leads to deadlock
      puts "Thread 2 acquired mutex1"
    end
  end
  puts "Thread 2 finished"
end


thread1.join
thread2.join

puts "Application finished"
```

In this scenario, `thread1` acquires `mutex1` and then tries to acquire `mutex2`, while `thread2` acquires `mutex2` and then attempts to acquire `mutex1`. This circular dependency creates the deadlock: neither thread can proceed, causing the application to hang.

**4. Background Job Queue Problems:**

If you use background jobs (such as via Sidekiq, Resque, or DelayedJob), these queues can become a bottleneck if the worker processes are overloaded, or if jobs fail without proper error handling, clogging the queue with retries. If you’re waiting on a side effect of a background job that has not completed or completed successfully, this can also appear as the application hanging.

Monitoring job queue lengths and worker utilization is key. Here's an illustrative example of how to perform queue monitoring with Sidekiq:

```ruby

# Using Sidekiq client for information

require 'sidekiq/api'

stats = Sidekiq::Stats.new

puts "Processed: #{stats.processed}"
puts "Failed: #{stats.failed}"
puts "Enqueued: #{stats.enqueued}"
puts "Workers size: #{stats.workers_size}"

# Fetching information about each queue:

Sidekiq::Queue.all.each do |queue|
 puts "Queue Name: #{queue.name}"
 puts "Queue Size: #{queue.size}"
end
```

This provides important information about queue status. When troubleshooting a hang, checking the status of your background job queue is often a good place to start.

**Recommendations for Further Reading and Learning:**

To deepen your understanding, consider diving into the following resources:

*   **"Designing Data-Intensive Applications" by Martin Kleppmann:** This book provides a detailed overview of many aspects of distributed systems, including concepts vital to understanding concurrency and bottlenecks.
*   **"Database Internals: A Deep Dive into How Distributed Data Systems Work" by Alex Petrov:** Essential for understanding how databases work and how to properly manage and tune them to meet performance demands.
*   **Ruby documentation for threading and concurrency primitives:** The core Ruby documentation on `Thread`, `Mutex`, and other threading features provides an important foundational understanding.
*   **Rails Guides:** The official Rails documentation on database connections and performance is crucial for understanding framework-specific best practices.
*   **Monitoring tool documentation (e.g., Prometheus, Datadog):** Learning to use monitoring tools effectively is crucial for diagnosing issues like these. The official documentation of your monitoring stack is key.

In summary, a Rails application hanging on all requests is rarely a single issue. It's often an indication of resource contention, concurrency problems, or misconfigured settings. By methodically examining thread pool utilization, database connection pool status, potential deadlocks, and background job queues, and by utilizing proper monitoring, you can identify and address the underlying problems and restore the responsiveness of your application. My experience has taught me that thorough monitoring is the most reliable way to catch these problems early, long before they turn into a major outage.
