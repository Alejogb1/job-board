---
title: "How can ActiveSupport around callbacks be used for tracing and logging service invocations?"
date: "2024-12-23"
id: "how-can-activesupport-around-callbacks-be-used-for-tracing-and-logging-service-invocations"
---

, let's delve into this. I remember back in the early days of my time working on a large Rails application, we had a significant problem: we were losing track of exactly which services were being called and in what order, making performance debugging a nightmare. It was like navigating a maze blindfolded. That's when we really started leveraging ActiveSupport callbacks for systematic logging and tracing. It wasn't just about seeing *if* a service was called, but also *when*, *with what parameters*, and *what the result was*. Let me walk you through how we approached this, and how you can too.

Essentially, ActiveSupport callbacks provide a hook into the lifecycle of an object. Instead of embedding logging statements directly within each service method, which is a maintenance headache, we defined callbacks that automatically triggered logging at various points: *before* the service method executes, *after* a successful execution, and *after* an execution that raised an exception. This approach decouples the logging mechanism from the service implementation itself, leading to cleaner and more maintainable code.

First, let's talk about setting up these callbacks. We generally used `ActiveSupport::Callbacks` mixin directly or, more commonly in Rails, we’d define our custom service classes inheriting from `ApplicationService` or a similarly structured base class that includes the necessary modules. Let's assume you have a `MyService` class. To add callbacks, you'd include the `ActiveSupport::Callbacks` module:

```ruby
require 'active_support/callbacks'

class MyService
  include ActiveSupport::Callbacks

  define_callbacks :call

  def call(param1, param2)
    run_callbacks :call do
      puts "Executing service with: #{param1}, #{param2}"  # Replace with actual service logic
      result = param1 + param2
      puts "Service executed successfully with result: #{result}"
      result
    end
  end
end

```

Here, we’ve defined a callback `:call` and we’re wrapping the core logic of the `call` method with `run_callbacks :call do ... end`. Now, we can define before, after, and around callbacks that will fire as the `call` method is invoked. We don't have logging yet, but we have the basic infrastructure.

Let’s build on this foundation to add logging. We'll use a before callback to log the input parameters, an after callback to log the success result, and an around callback to capture exceptions and time execution, all in a centralized manner. Let's extend the previous example:

```ruby
require 'active_support/callbacks'
require 'logger'

class MyLoggedService
  include ActiveSupport::Callbacks

  define_callbacks :call

  attr_reader :logger

  def initialize(logger: Logger.new(STDOUT))
    @logger = logger
  end

  def call(param1, param2)
    run_callbacks :call do
      puts "Executing service with: #{param1}, #{param2}"  # Replace with actual service logic
      result = param1 + param2
      puts "Service executed successfully with result: #{result}"
      result
    end
  end

  set_callback :call, :before, lambda { |this, param1, param2| this.logger.info "Service Started: #{this.class.name} with params: #{param1}, #{param2}" }
  set_callback :call, :after,  lambda { |this, result| this.logger.info "Service Finished: #{this.class.name} result: #{result}" }, if: :result
  set_callback :call, :around, lambda { |this, block, param1, param2|
    start_time = Process.clock_gettime(Process::CLOCK_MONOTONIC)
    begin
      result = block.call(param1, param2)
      result
    rescue => e
      this.logger.error "Service Failed: #{this.class.name} with error: #{e.message}"
      raise e
    ensure
      end_time = Process.clock_gettime(Process::CLOCK_MONOTONIC)
      duration = end_time - start_time
      this.logger.info "Service Duration: #{this.class.name} took: #{duration} seconds"
    end
  }
end

service = MyLoggedService.new
service.call(5, 10)
```

In this snippet, the `before` callback logs the start of the service with parameters. The `after` callback logs the successful result (it only executes when a result is present, thanks to the `:if` option). The `around` callback wraps the entire execution, capturing any exceptions and timing the duration. You'll see detailed logging output to STDOUT, demonstrating how these callbacks provide a comprehensive trace. We chose STDOUT here for simplicity, but in a real-world setting, you'd likely use a more sophisticated logging system.

One crucial aspect is using the `this` parameter in the lambdas. This lets the callback access the instance of the service, allowing us to reach the logger. The block given to the around callback executes the core method logic, and we capture both the success case and the error case. If an error happens, the `rescue` block logs it and re-raises the exception. The `ensure` block guarantees the timer is logged no matter what happens.

Now, let's discuss a slightly more complex scenario. Suppose we want to associate a unique request id or correlation id with each service invocation, for tracing across multiple services, which is a common pattern in distributed systems. We can pass this id as a parameter, or use the current context from a library like `request_store` or other context propagation tools, and incorporate that into the log messages. Here’s an example using a simplified request_store-like approach:

```ruby
require 'active_support/callbacks'
require 'logger'

module RequestStore
  @store = {}
  def self.store
    @store
  end

  def self.with_context(correlation_id, &block)
    begin
      @store[:correlation_id] = correlation_id
      block.call
    ensure
      @store[:correlation_id] = nil
    end
  end

  def self.correlation_id
     @store[:correlation_id]
  end
end


class ContextualLoggedService
  include ActiveSupport::Callbacks

  define_callbacks :call

  attr_reader :logger

  def initialize(logger: Logger.new(STDOUT))
    @logger = logger
  end

    def call(param1, param2)
      run_callbacks :call do
        puts "Executing service with: #{param1}, #{param2}"  # Replace with actual service logic
        result = param1 + param2
        puts "Service executed successfully with result: #{result}"
        result
      end
    end

    set_callback :call, :before, lambda { |this, param1, param2|
        correlation_id = RequestStore.correlation_id
        this.logger.info "Service Started: #{this.class.name} with correlation_id: #{correlation_id}, params: #{param1}, #{param2}"
    }
    set_callback :call, :after, lambda { |this, result|
        correlation_id = RequestStore.correlation_id
        this.logger.info "Service Finished: #{this.class.name} with correlation_id: #{correlation_id}, result: #{result}"
      }, if: :result
    set_callback :call, :around, lambda { |this, block, param1, param2|
      start_time = Process.clock_gettime(Process::CLOCK_MONOTONIC)
        correlation_id = RequestStore.correlation_id
        begin
          result = block.call(param1, param2)
          result
        rescue => e
          this.logger.error "Service Failed: #{this.class.name} with correlation_id: #{correlation_id}, error: #{e.message}"
          raise e
        ensure
          end_time = Process.clock_gettime(Process::CLOCK_MONOTONIC)
          duration = end_time - start_time
          this.logger.info "Service Duration: #{this.class.name} with correlation_id: #{correlation_id}, took: #{duration} seconds"
        end
     }
end


RequestStore.with_context("request-123") do
  service = ContextualLoggedService.new
  service.call(5, 10)
end
```

Here, we've added a very basic `RequestStore` module for demonstration. In a full Rails application, this functionality is provided by gems like `request_store`. The callbacks now access the current correlation id to enhance the log messages. Now, every log entry associated with a request has its specific id, making cross-service request tracing simpler.

For deeper exploration, I highly recommend the *Active Support Core Extensions* section in the official Rails documentation. It’s a treasure trove for understanding the nuances of callbacks and how they interact with the rest of the framework. Also, the book *Rails AntiPatterns: Best Practice Refactoring* by Chad Pytel and Tammer Saleh delves into how to structure services correctly using patterns like this and explains how to avoid overusing callbacks. Additionally, the *Enterprise Integration Patterns* by Gregor Hohpe and Bobby Woolf, though not Rails specific, provides indispensable insights into tracing patterns in distributed systems and how to implement similar concepts at a larger architectural level. These resources will greatly enhance your understanding of these powerful tools and their practical applications.
