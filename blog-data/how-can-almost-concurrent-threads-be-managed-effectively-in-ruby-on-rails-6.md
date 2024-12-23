---
title: "How can almost concurrent threads be managed effectively in Ruby on Rails 6?"
date: "2024-12-23"
id: "how-can-almost-concurrent-threads-be-managed-effectively-in-ruby-on-rails-6"
---

Okay, let's tackle this. Managing almost concurrent threads in Ruby on Rails 6 is, shall we say, not always straightforward. I've seen projects grind to a halt, become riddled with race conditions, or just plain exhibit unpredictable behavior because concurrency wasn't handled thoughtfully. I recall a particularly brutal project where we were processing large batches of user data; we went from snappy to sluggish in a matter of days. The root? Poor concurrency management. We were throwing threads at the problem without proper control. It taught me some harsh lessons, and I've learned to approach concurrent tasks with a more cautious, yet effective, toolkit since.

The key here is that, while ruby has threads, they are not true os-level threads, but green threads managed by the Ruby VM (specifically in MRI). This means that you will not be able to have true parallel execution in MRI for things like pure ruby code, but you *can* effectively manage concurrent execution. This is different than say, Java, or Go, where the language runtimes themselves have built in OS threads, and can therefore parallelize operations. Because of these constraints, our primary tools aren't going to be aggressive attempts to force parallelism via ruby threads, but rather, strategies that make efficient use of concurrency and minimize contention.

First, let's break down what "almost concurrent" might mean in the context of Rails. Most likely, we're talking about operations that are not inherently synchronous, meaning that they don't require the execution of one operation to finish entirely before the next one begins. This often involves i/o-bound tasks, such as API calls, database interactions, or other forms of external communication. Ruby's Global Interpreter Lock (GIL), or rather, its absence in JRuby and TruffleRuby, impacts how true parallelism may (or may not) occur. But our main focus is maximizing efficient use of available processing power through concurrency within Rails' operational constraints.

The primary tool at our disposal in Rails for this kind of thing is using background jobs. For that, I lean towards `ActiveJob` and an asynchronous queue like `Sidekiq`, or `Resque`. These tools allow us to push work out of the main request-response cycle and manage it independently, which is fundamental to effective concurrency. The advantage with this approach is that the jobs are run outside the context of the webserver process, and thus, do not take up request handling threads.

Here's a very basic example, using sidekiq. Let's assume we have a function that sends an email, and we want to send that email concurrently, and outside the web-request thread.

```ruby
# app/jobs/email_job.rb
class EmailJob < ApplicationJob
  queue_as :default

  def perform(user_id)
    user = User.find(user_id)
    UserMailer.welcome_email(user).deliver_now
  end
end
```
And then we'd call it within a controller or service object:

```ruby
# app/controllers/users_controller.rb
def create
  @user = User.new(user_params)
  if @user.save
      EmailJob.perform_later(@user.id)
      redirect_to @user, notice: 'User was successfully created.'
  else
    render :new, status: :unprocessable_entity
  end
end
```

Here we aren't truly parallelizing, but we're making the user experience much better; we're offloading a relatively slow task to the background and making sure the page loads *now*. While our code does not parallelize across different threads, when these jobs are handled by sidekiq, they are handled in a different thread or process, that is separate from your webserver request handler.

The second crucial element is managing resources appropriately. When dealing with concurrent processes, database access becomes a major potential bottleneck, and we *must* avoid using the same database connection across threads. In sidekiq (or other async queues), each worker thread has its own database connection pool, but you still need to handle database reads and updates carefully within your jobs. This prevents database contention and locks and ensures that all processes can be handled effectively. This is critical because most web frameworks are single-threaded in their request-handling. If the request handling threads start blocking on long I/O, then your site effectively becomes unavailable.

Here's an example of how you might make more than one API call concurrently, using concurrent-ruby gem, and offload it to a background thread:

```ruby
# app/jobs/api_job.rb
class ApiJob < ApplicationJob
  queue_as :default

  def perform(endpoint1, endpoint2)
    futures = []
    futures << Concurrent::Future.execute { make_api_call(endpoint1) }
    futures << Concurrent::Future.execute { make_api_call(endpoint2) }
    
    results = futures.map(&:value)
    process_results(results)
  end

  private
  
  def make_api_call(endpoint)
      #replace this with your actual api call.
    HTTParty.get(endpoint).body
  end

  def process_results(results)
     #handle results of api calls
     results.each do |result|
         puts result
     end
  end
end

```

and call it in your code using:

```ruby
ApiJob.perform_later("https://some.api.com/endpoint1", "https://some.other.api.com/endpoint2")
```

In this example, we're leveraging `concurrent-ruby` to fire off API calls concurrently via background threads. Here, while still being single threaded within the worker process, we're avoiding waiting for api calls to complete sequentially, speeding up the process. The `Concurrent::Future` handles the execution, allowing us to get the `value` when it's available. This approach is particularly helpful for operations involving multiple external services or tasks. Note that you could also make api calls using the `async` and `await` functionalities within Ruby 3, but for older codebases, `concurrent-ruby` is the go-to gem for background thread management.

Lastly, proper error handling is absolutely vital, and especially so in concurrent environments. Ensure that you catch exceptions in your background jobs and handle them gracefully. Requeue failed jobs, or log failures appropriately; otherwise, they will just disappear and be very hard to debug. Using a monitoring tool, such as Prometheus, Sentry, or New Relic, is essential to track the health and performance of your background processing system.

For more in-depth reading on concurrency patterns, I would suggest reviewing the "Patterns of Enterprise Application Architecture" by Martin Fowler, as it provides foundational guidance on managing concurrency, transactions and related concepts in complex software systems, which is invaluable for real-world application development. For more specifics on concurrent Ruby itself, I would recommend exploring the documentation of the "concurrent-ruby" gem and also looking into the various articles detailing ruby's Global Interpreter Lock (GIL) behaviour.

In summary, effective management of "almost concurrent" threads in Rails 6 isn’t about trying to force true parallelism with Ruby's threads, it's about architecting your application to leverage background jobs, manage database access wisely, and implement comprehensive error handling. By doing so, you move long running processes out of the main request thread, and maximize concurrency using appropriate tools. This strategy, honed through past project pains, has allowed me to build robust, scalable Rails applications. It's about using the *right* concurrency mechanisms in the right way, rather than merely trying to force concurrency in places where it's ill-suited.
