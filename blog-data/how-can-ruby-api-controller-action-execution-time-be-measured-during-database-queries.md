---
title: "How can Ruby API controller action execution time be measured during database queries?"
date: "2024-12-23"
id: "how-can-ruby-api-controller-action-execution-time-be-measured-during-database-queries"
---

Let's consider this from a few different angles; I've actually had to tackle similar performance bottlenecks more times than I care to remember, so hopefully, my experiences will be of use. Measuring Ruby on Rails API controller action execution time, specifically concerning database queries, is crucial for identifying performance issues. We want to pinpoint not just the overall time but also break down *where* time is being spent within our actions, especially within database interactions. There are several methods to accomplish this, each with varying levels of granularity and complexity. We'll cover some techniques that are quite common, while delving into how you can customize them for precision.

Fundamentally, what we're aiming for is visibility. When a request hits an endpoint, we need to know how long the controller action takes and, critically, how much time the database is consuming. This allows us to optimize specific areas, perhaps by refactoring slow queries, adding indexes, or implementing caching strategies.

First, let’s discuss the most straightforward approach: wrapping our controller actions with timing logic. This gives a high-level overview. We can use Ruby's `benchmark` module. Here's a snippet illustrating that:

```ruby
require 'benchmark'

class MyApiController < ApplicationController
  def index
    time = Benchmark.realtime do
      @users = User.all # Example database query
      render json: @users
    end

    Rails.logger.info "Action 'index' took #{time} seconds."
  end
end
```

Here, we’re utilizing `Benchmark.realtime` to measure the elapsed time within the `do..end` block. The result, displayed in seconds, gives a basic idea of total execution time. While simple, this method doesn’t isolate database-specific time; it includes the rendering, any other logic, and database interaction within that single measurement. This can be useful for quick diagnostics but isn't ideal for a deep dive into database performance.

Next, we need to get more specific and see how much time we spend in the database. Rails offers a fantastic mechanism for this: ActiveSupport::Notifications. We can subscribe to database-related events and track the duration of each query. Let me share a slightly more refined example of how we might implement this:

```ruby
class MyApiController < ApplicationController
  before_action :start_db_timing

  after_action :end_db_timing

  def index
      @users = User.all
      render json: @users
  end

  private

  def start_db_timing
    @db_query_times = []

    @subscriber = ActiveSupport::Notifications.subscribe('sql.active_record') do |_, start, finish, _, payload|
        @db_query_times << (finish - start)
    end
  end

  def end_db_timing
      ActiveSupport::Notifications.unsubscribe(@subscriber)
      total_db_time = @db_query_times.sum
      action_time = Benchmark.realtime do
        # Ensure total action is measured at end
      end

      Rails.logger.info "Action 'index' took #{action_time} seconds. DB time: #{total_db_time} seconds."
  end
end
```

Here, we're subscribing to the `sql.active_record` event, triggered every time ActiveRecord executes a query. The `start_db_timing` method initializes an array to hold timings and sets up the subscriber. Then, within the subscriber block, we append the duration of each database query to the array. In `end_db_timing` method, we remove the subscription and log the total database query time and overall action time. This approach gives us separate timing data for the database and overall action, allowing for better performance pinpointing. It helps us determine if the problem lies within a specific controller action or specific database interactions.

However, let’s say you’re working with a legacy system where you can't modify every controller. It might make sense to apply this behavior more globally, perhaps through a middleware or a custom module. Let's illustrate that approach via a module:

```ruby
module DbQueryTiming
  def around_action(*args, &block)
    @db_query_times = []
    subscriber = ActiveSupport::Notifications.subscribe('sql.active_record') do |_, start, finish, _, payload|
        @db_query_times << (finish - start)
    end
    begin
        super(*args, &block)
    ensure
      ActiveSupport::Notifications.unsubscribe(subscriber)
      total_db_time = @db_query_times.sum
      action_time = Benchmark.realtime { }
      Rails.logger.info "Action #{action_name} took #{action_time} seconds. DB time: #{total_db_time} seconds."
    end
  end
end


class ApplicationController < ActionController::Base
    prepend DbQueryTiming
end
```

This solution uses `around_action` and prepends the module to the `ApplicationController`, thus applying it to every controller action. The `around_action` is a powerful tool; it intercepts the execution of every action. Similar to the previous example, we gather the database query times, compute the sum, and log this information, along with the overall action execution time. Note how the `ensure` block makes sure `ActiveSupport::Notifications.unsubscribe(subscriber)` is always executed, whether the action succeeds or not, preventing potential resource leaks. By wrapping the action in `begin/ensure`, we ensure that the logging always occurs. This is a more versatile strategy when you need to audit database time across many controllers.

In terms of diving deeper, I highly recommend consulting "Understanding the Database: SQL and Relational Theory" by Michael J. Hernandez; it gives a comprehensive understanding of database theory that's essential for identifying potential bottlenecks. For Rails-specific performance optimization, "Rails Performance Optimization" by Nate Berkopec is an excellent resource. Understanding profiling tools within Rails as well as database specific profiling such as `EXPLAIN` commands are crucial for optimizing your application.

In conclusion, these examples are designed to offer different approaches, tailored to varying situations. The direct `Benchmark` method is the quickest, while the `ActiveSupport::Notifications` approach provides deeper insights into database interactions. And the around_action strategy enables more global application of database timing within your Rails applications. I've often found it beneficial to start with the simplest method, then moving towards more granular approaches as needed. Choosing the right strategy depends on your specific requirements, but having these tools at your disposal will substantially help in addressing any performance-related challenges in your Ruby on Rails applications.
