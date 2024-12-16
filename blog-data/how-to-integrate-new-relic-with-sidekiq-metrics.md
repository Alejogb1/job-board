---
title: "How to integrate New Relic with Sidekiq metrics?"
date: "2024-12-16"
id: "how-to-integrate-new-relic-with-sidekiq-metrics"
---

Let’s address integrating New Relic with Sidekiq metrics. It’s a topic I’ve spent considerable time on, particularly back in the days when we were scaling a fairly complex system that heavily relied on background processing. I recall one particular incident; we had a sudden spike in user activity, which, naturally, put considerable strain on our worker queues managed by Sidekiq. Without proper monitoring, it was akin to navigating in the dark, and we realized then the critical need for robust performance observability. This wasn't just about identifying problems; it was also about understanding normal behavior and preemptively addressing potential bottlenecks.

The core issue when integrating New Relic and Sidekiq is capturing the right kinds of data and making it digestible for analysis. Sidekiq, at its heart, is a workhorse; it executes jobs asynchronously. Without intervention, New Relic sees it as a background process but doesn't inherently understand the intricate details of the queues, processing times per job, and the error rates. The first principle here is that we need to extend what New Relic knows about our application by instrumenting the Sidekiq processing pipeline. We need to create what are essentially custom metrics and events.

My approach generally starts with the understanding that we have two primary ways to report data: custom metrics and custom events. Custom metrics are primarily for numerical values, such as queue sizes or average processing times, while custom events allow us to record specific data points about individual jobs. Choosing between them often depends on what you intend to achieve. If you need to visualize trends in queue depth over time, use custom metrics. If you need to analyze individual job failures, or success metrics, custom events are your ally.

Let’s consider an example of how to report queue size. We might implement this within a Sidekiq initializer. The code could look something like this:

```ruby
# config/initializers/sidekiq.rb
require 'newrelic_rpm'

Sidekiq.configure_server do |config|
  config.on(:startup) do
      Thread.new do
        loop do
          begin
            queue_sizes = Sidekiq::Queue.all.each_with_object({}) { |queue, hash|
              hash[queue.name] = queue.size
            }

            queue_sizes.each do |queue_name, size|
              NewRelic::Agent.record_metric("Custom/Sidekiq/Queue/#{queue_name}/Size", size)
            end

            sleep 60 # Report every minute
          rescue => e
            NewRelic::Agent.notice_error(e)
            sleep 60
          end
        end
      end
  end
end
```

Here, we use `Sidekiq.configure_server` to inject code into the server initialization. Within the thread, we loop infinitely, query the queue sizes, and report these to New Relic. This approach allows for time-series data, which can be graphed in the New Relic dashboard. The key here is to structure the metric names in a way that makes sense to you—here I’m choosing `Custom/Sidekiq/Queue/queue_name/Size`. The use of `Thread.new` ensures the monitoring is asynchronous and doesn’t interfere with Sidekiq’s main processing cycle. The `begin/rescue` block ensures any issues are reported to New Relic as well, preventing silent failures within the monitor.

Now, let’s move on to custom events. When a job completes, we often want to know the status – whether it succeeded, failed, or took an unusually long time. We can use a Sidekiq middleware to capture this data. Here’s a simplified example that illustrates this concept:

```ruby
# app/middleware/sidekiq_newrelic_middleware.rb
require 'newrelic_rpm'

class SidekiqNewRelicMiddleware
  def call(worker, job, queue)
    start = Time.now
    begin
      yield
      duration = (Time.now - start).to_f
      NewRelic::Agent.record_custom_event('SidekiqJob', {
        'queue' => queue,
        'worker' => worker.class.name,
        'status' => 'success',
        'duration' => duration,
        'job_id' => job['jid']
      })

    rescue => e
      duration = (Time.now - start).to_f
      NewRelic::Agent.record_custom_event('SidekiqJob', {
        'queue' => queue,
        'worker' => worker.class.name,
        'status' => 'failure',
        'duration' => duration,
        'job_id' => job['jid'],
        'error_class' => e.class.name,
        'error_message' => e.message
      })
      raise
    end
  end
end
```

In this middleware, we capture the job start time, execute the job via `yield`, and record a custom event called 'SidekiqJob' with relevant attributes like status, duration, and queue name. The exception handling block also records similar events, but with detailed failure information, making debugging easier. To register the middleware with Sidekiq you'd add the following to `config/initializers/sidekiq.rb`:

```ruby
Sidekiq.configure_server do |config|
  config.server_middleware do |chain|
    chain.add SidekiqNewRelicMiddleware
  end
end
```

It's crucial to note the potential performance impact of event reporting and to make sure you do not record excessive data which will impact the performance of the sidekiq workers. In situations where a job executes numerous operations you may want to throttle reporting of data points.

Now, let's look at a slightly more advanced scenario: timing individual operations within a worker. Suppose we have a Sidekiq worker that processes a document by performing several distinct tasks. We can instrument those tasks with New Relic's `trace_method` method, which allows you to track execution time for individual methods:

```ruby
# app/workers/document_processor_worker.rb
require 'newrelic_rpm'

class DocumentProcessorWorker
  include Sidekiq::Worker

  def perform(document_id)
    NewRelic::Agent.trace_method("DocumentProcessorWorker/load_document") do
      load_document(document_id)
    end

    NewRelic::Agent.trace_method("DocumentProcessorWorker/extract_text") do
      extract_text(document_id)
    end

    NewRelic::Agent.trace_method("DocumentProcessorWorker/process_text") do
       process_text(document_id)
    end
  end


  private

  def load_document(document_id)
    # Simulate loading the document
    sleep rand(0.1..0.3)
  end

  def extract_text(document_id)
    # Simulate extracting text
      sleep rand(0.2..0.5)
  end

  def process_text(document_id)
      # Simulate processing text
       sleep rand(0.3..0.7)
  end
end
```
Here, each of the methods, `load_document`, `extract_text`, and `process_text` are traced individually, which allows for visibility into which particular operations take the most time. With this level of detail you’ll be able to identify performance bottlenecks inside of the workers themselves and not just the processing as a whole.

In summary, integrating New Relic with Sidekiq effectively involves instrumenting both the global Sidekiq environment and individual worker behavior. By utilizing custom metrics, custom events, and methods such as `trace_method` you can create a comprehensive view of your Sidekiq processing pipeline. For deeper technical understanding, I would recommend exploring the official New Relic documentation, focusing on custom metrics and events. Specifically, pay close attention to New Relic's documentation on the agent API, which is critical for crafting custom instrumentation. Furthermore, "The Art of Monitoring" by James Turnbull provides a solid understanding of monitoring and observability principles, which can be particularly valuable for establishing an overall strategy. Finally, if you’re looking for a deeper dive into Sidekiq internals, the source code itself is often the best guide. Having this information at your fingertips can dramatically improve your ability to understand and enhance the performance of your system.
