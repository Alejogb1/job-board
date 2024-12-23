---
title: "How can I integrate New Relic with Sidekiq metrics?"
date: "2024-12-23"
id: "how-can-i-integrate-new-relic-with-sidekiq-metrics"
---

Let's dive straight in; I've been down this particular road a few times, and it's often a tad more involved than simply flicking a switch. Integrating New Relic with Sidekiq for comprehensive metric monitoring isn't automatically plug-and-play; you have to be deliberate about it. It's about collecting meaningful data that allows you to pinpoint performance bottlenecks or issues within your background job processing. Here’s my take, drawing from my experiences with similar architectures.

The core problem stems from Sidekiq and New Relic being two separate systems. Sidekiq handles asynchronous job execution within your application, while New Relic provides application performance monitoring (apm). Out of the box, New Relic's agent primarily tracks web requests and database queries associated with those requests. It doesn't automatically grasp what happens within your Sidekiq processes. So, bridging this gap requires explicit instrumentation. The idea is to notify New Relic about each job execution within Sidekiq so that it can track transaction times, success rates, failures, and so forth.

There are a few ways to approach this, and the most effective method for you will likely depend on the specifics of your setup. I'll outline what I've found works best, along with some code snippets, and we’ll go from there.

**Option 1: Using Custom Instrumentation via `Sidekiq::Middleware`**

This method involves creating a custom middleware that intercepts Sidekiq job execution, creates a New Relic transaction, and records the processing time. This provides granular data, akin to how web requests are tracked. The approach is highly flexible but needs careful implementation to avoid interference with other Sidekiq middleware.

Here’s an example middleware class in Ruby that I've adapted from some work I did a while back for a high-traffic e-commerce application:

```ruby
require 'new_relic/agent'

class NewRelicSidekiqMiddleware
  def call(worker, job, queue)
    transaction_name = "Sidekiq/#{worker.class.name}/#{job['class']}"

    ::NewRelic::Agent.with_transaction(transaction_name, category: :task) do
      begin
        yield
      rescue Exception => e
        ::NewRelic::Agent.notice_error(e, metric: { 'Sidekiq' => 'Error' })
        raise
      end
    end
  end
end
```

To utilize this, you add it to your Sidekiq configuration:

```ruby
Sidekiq.configure_server do |config|
  config.server_middleware do |chain|
    chain.add NewRelicSidekiqMiddleware
  end
end

Sidekiq.configure_client do |config|
  # Optional client-side instrumentation, if desired, but server-side is key.
end
```

This middleware uses the `NewRelic::Agent.with_transaction` method, which is central to New Relic instrumentation. It sets up a transaction that reflects the job's class, marking it as a background `task`. The `rescue` block ensures that any errors during job processing are properly reported to New Relic, offering full error tracing for your background jobs.

**Option 2: Utilizing New Relic’s Ruby Agent API for Manual Instrumentation**

Instead of using middleware, another way is to manually instrument the code within your worker classes directly, using the New Relic Agent API. This can be useful if you need more control over which parts of your worker's execution are monitored, or if you wish to instrument multiple points of concern within the job’s execution.

Here's a simplified example within a Sidekiq worker:

```ruby
require 'new_relic/agent'

class MyWorker
  include Sidekiq::Worker

  def perform(some_parameter)
    ::NewRelic::Agent.instrument_code('MyWorker/perform/processing') do
        # Actual job processing logic here
        puts "Processing #{some_parameter}"
        sleep(rand(1..3)) # Simulate some work
    end
    # Additional reporting or error logging
    ::NewRelic::Agent.increment_metric('Custom/MyWorker/JobsProcessed')
  rescue Exception => e
      ::NewRelic::Agent.notice_error(e, metric: { 'MyWorker' => 'Error' })
      raise
  end
end
```

In this case, `::NewRelic::Agent.instrument_code` is used to demarcate a segment of code that should be tracked, allowing you to monitor timings specific to that part. Additionally, a custom metric (`Custom/MyWorker/JobsProcessed`) is incremented to track the number of successfully processed jobs. Again, any exception is caught and logged into New Relic error tracking.

**Option 3: The 'Best of Both Worlds'**

Often, I combine elements of the previous two options. I use the middleware to establish the base transaction structure and then sprinkle in manual instrumentation for specific sections within long-running jobs to gain better insight into bottlenecks. It gives both overarching transaction tracking and granular data specific to worker methods.

Here's how you might do it; combining the middleware and manual instrumentation:

```ruby
# Assuming the middleware from option 1 is already implemented
require 'new_relic/agent'

class ComplexWorker
  include Sidekiq::Worker
  def perform(job_id)
    ::NewRelic::Agent.instrument_code("ComplexWorker/initial_setup") do
      # setup code here
      sleep(0.5)
    end

    ::NewRelic::Agent.instrument_code("ComplexWorker/processing_data") do
      # The real heavy lifting
      sleep(2)
    end

    ::NewRelic::Agent.increment_metric("Custom/ComplexWorker/CompletedJobs")
  rescue Exception => e
    ::NewRelic::Agent.notice_error(e, metric: { 'ComplexWorker' => 'Error' })
    raise
  end
end
```

The middleware ensures that overall transaction is tracked (as `Sidekiq/ComplexWorker/ComplexWorker`), and then the internal method `perform` uses further instrumentation.

**Key Considerations:**

*   **Transaction Names:** Make sure your transaction names are informative. Avoid static names like 'perform' since that makes distinguishing between jobs difficult. Incorporating class names into the transaction name is extremely helpful.
*   **Custom Metrics:** Don't underestimate the power of custom metrics. Track specific things that matter to your application, such as the number of emails sent, the time spent making an external API call, or any other operational specifics.
*   **Error Handling:** Always ensure errors within your background jobs are correctly captured by New Relic by utilizing `NewRelic::Agent.notice_error`.
*   **Configuration:** Ensure the New Relic agent is properly configured for your application environment, and always use the latest version for best results.
*   **Testing:** Thoroughly test your instrumentation in a non-production environment to verify it's capturing the desired metrics without introducing performance degradation.

For further reading, I'd strongly recommend starting with the official New Relic documentation for the Ruby agent. It contains detailed explanations of the API methods I've mentioned. Also, check out "Refactoring Ruby," by Martin Fowler, for best practices on writing clean and maintainable code. Additionally, explore articles or blog posts from the Thoughtbot team, who are known for robust Ruby development practices. Understanding the architecture of your system is crucial for optimizing both it’s operation and the way you monitor it, so “Patterns of Enterprise Application Architecture,” also by Martin Fowler, may also be beneficial in a broader sense.

In short, integrating New Relic with Sidekiq involves deliberate instrumentation. Choose the technique that suits your project's specific needs and complexity. The goal is to have insight into the inner workings of your background processing as granular as it is necessary to have. If done correctly, it will enhance your monitoring capabilities, enabling you to keep your services humming along smoothly.
