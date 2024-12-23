---
title: "How can I integrate New Relic And Sidekiq Metrics?"
date: "2024-12-23"
id: "how-can-i-integrate-new-relic-and-sidekiq-metrics"
---

Alright, let's talk about integrating New Relic with Sidekiq. It's a common requirement, and I’ve certainly spent my fair share of time fine-tuning it across various projects. Over the years, I've seen firsthand how crucial it is to have clear visibility into background job performance, especially when dealing with systems scaling. It's not just about knowing if Sidekiq is running; it's about understanding the nuances, like job processing times, queue backlogs, and error rates, which directly impact application responsiveness.

Essentially, we want New Relic to act as our comprehensive monitoring hub for everything happening inside our Sidekiq processes. The good news is that New Relic’s Ruby agent and Sidekiq play quite nicely together, provided you know the correct techniques. I'm going to focus on how to extract and send detailed, meaningful metrics from Sidekiq into New Relic. This isn’t just about the default metrics you get out of the box; we'll dive a bit deeper.

First and foremost, the New Relic Ruby agent does a pretty decent job of automatically instrumenting Sidekiq. That’s where you'll see basic metrics about the overall job throughput and duration. But to unlock more insightful data, we’ll need to extend this by sending custom events and metrics. Think of this as a way of annotating our existing New Relic dashboards with more specific information about what's happening inside Sidekiq. The goal is to answer questions like: "Which jobs are causing the biggest bottlenecks?" or "Are there specific queues that are consistently overloaded?"

To begin, custom events are a powerful way to track the lifecycle of a job. This means we can follow a job from the moment it’s enqueued until it completes or fails. We achieve this through Sidekiq’s middleware system, which provides hook points for various stages in a job's execution.

Here’s a code snippet illustrating how to send custom events when a job is enqueued:

```ruby
# app/middleware/new_relic_sidekiq_enqueue_middleware.rb
class NewRelicSidekiqEnqueueMiddleware
  def call(worker, msg, queue)
    begin
      ::NewRelic::Agent.record_custom_event(
        'SidekiqJobEnqueued',
        {
          'queue' => queue,
          'worker' => worker.class.name,
          'args' => msg['args'].to_s # Convert args to string for New Relic
        }
      )
      yield
    rescue => e
      ::NewRelic::Agent.notice_error(e)
      raise
    end
  end
end

Sidekiq.configure_server do |config|
  config.server_middleware do |chain|
    chain.add NewRelicSidekiqEnqueueMiddleware
  end
end

Sidekiq.configure_client do |config|
   config.client_middleware do |chain|
    chain.add NewRelicSidekiqEnqueueMiddleware
  end
end
```

Here, I’ve created a middleware, `NewRelicSidekiqEnqueueMiddleware`. This middleware gets executed every time a job is enqueued. Inside it, `NewRelic::Agent.record_custom_event` is used to send an event called `SidekiqJobEnqueued` to New Relic. Crucially, we include important context such as the queue name, worker class, and the arguments for the job. The try-catch ensures any issues are caught and reported to New Relic. Remember to require this file within your Sidekiq initialization.

Next, let's look at how to track job performance— specifically, the time it takes for a job to process. This is crucial for identifying bottlenecks. The Sidekiq ‘perform’ call is where this happens:

```ruby
# app/middleware/new_relic_sidekiq_performance_middleware.rb
class NewRelicSidekiqPerformanceMiddleware
  def call(worker, msg, queue)
    start_time = ::Process.clock_gettime(::Process::CLOCK_MONOTONIC)
    begin
      yield
    ensure
      end_time = ::Process.clock_gettime(::Process::CLOCK_MONOTONIC)
      duration = (end_time - start_time) * 1000  # Convert to milliseconds
      ::NewRelic::Agent.record_metric(
        "Custom/Sidekiq/JobDuration/#{worker.class.name}",
        duration
      )

      ::NewRelic::Agent.record_custom_event(
        'SidekiqJobCompleted',
         {
           'queue' => queue,
           'worker' => worker.class.name,
           'duration_ms' => duration
         }
      )

    end
  rescue => e
    ::NewRelic::Agent.notice_error(e)
    raise
  end
end

Sidekiq.configure_server do |config|
  config.server_middleware do |chain|
    chain.add NewRelicSidekiqPerformanceMiddleware
  end
end
```

In this middleware, `NewRelicSidekiqPerformanceMiddleware`, I'm using `Process.clock_gettime` for more accurate timing measurements than the standard ruby Time library provides.  Then, I’m calling `NewRelic::Agent.record_metric` to send the job processing time as a metric. I’m naming the metric with a dynamic component representing the worker class. I also record a custom event ‘SidekiqJobCompleted’. This gives us both aggregate performance data (through the metric) and allows more detailed filtering in New Relic's event explorer (through custom events). Again error handling ensures issues don’t go unnoticed.

Finally, it's vital to track failures. Errors in background jobs can be silent killers if left unmonitored, so ensuring New Relic picks them up is important.

Here’s how to capture these errors:

```ruby
# app/middleware/new_relic_sidekiq_error_middleware.rb
class NewRelicSidekiqErrorMiddleware
  def call(worker, msg, queue)
    begin
      yield
    rescue => e
        ::NewRelic::Agent.notice_error(e, {
            :metric => 'Custom/Sidekiq/JobFailed',
            :params => {
                'queue' => queue,
                'worker' => worker.class.name,
                'args' => msg['args'].to_s
            }
          })
        raise
    end
  end
end

Sidekiq.configure_server do |config|
  config.server_middleware do |chain|
    chain.add NewRelicSidekiqErrorMiddleware
  end
end
```

In this middleware, `NewRelicSidekiqErrorMiddleware`, we’re using `NewRelic::Agent.notice_error` to capture the error details, and including the error's class and details. Critically, I add extra metadata via the `:params` key – the queue, worker, and arguments, enriching error reports in New Relic.

Regarding resources, I’d recommend delving into the New Relic Ruby agent documentation directly, which is available through their website. You can also check out the Sidekiq wiki for details about its middleware system. Specifically, the section covering custom middleware will be invaluable. Furthermore, any good book covering Ruby performance and background job processing (such as “Effective Ruby” by Peter J. Jones or “Working with Ruby Threads” by Jesse Storimer) should reinforce these concepts.

By implementing these middlewares, we gain a much more granular view of Sidekiq’s operation inside New Relic. It's not about just being able to say that Sidekiq is running, it's about having a detailed picture of individual job performance and enabling rapid identification of problems. Custom events allow for exploration of historical trends and correlate job performance with other events in the application, while custom metrics provide real-time insight into average performance. Error tracking is critical to maintain system health. It takes a little work to set up, but it pays dividends in terms of the quality of insights and therefore, a more robust application.
