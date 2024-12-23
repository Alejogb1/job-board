---
title: "How can I trigger `after_create_commit` using `run_callbacks`?"
date: "2024-12-23"
id: "how-can-i-trigger-aftercreatecommit-using-runcallbacks"
---

 It's a question that I've certainly pondered over a few times during my career, particularly back when I was deep in the trenches of building complex rails applications with intricate business logic. The challenge lies in the specific interaction between `after_create_commit` callbacks, typically intended for asynchronous tasks post-transaction, and the `run_callbacks` method, which is designed to, well, execute callbacks more directly. The straightforward invocation you might initially attempt won't work as you expect because `after_create_commit` has unique semantics tied to the transaction lifecycle. It’s not a regular callback.

The core issue here is that `after_create_commit` callbacks are *not* triggered until the database transaction is successfully committed. They aren’t designed to run immediately when called via a method like `run_callbacks`, or even `save`. They are enqueued by the transaction manager and executed once a commit event happens. This has significant implications for how we need to approach manipulating their execution flow.

Let me illustrate the problem with a concise example. Suppose you have a simple model, `BlogPost`, and you've defined an `after_create_commit` callback to dispatch a notification:

```ruby
class BlogPost < ApplicationRecord
  after_create_commit :notify_subscribers

  private

  def notify_subscribers
    puts "Notifying subscribers after successful commit"
    # ... code to dispatch notification ...
  end
end
```

Now, if you try something naive like this:

```ruby
blog_post = BlogPost.new(title: "Test Post", content: "This is a test.")
blog_post.run_callbacks(:create) # This will NOT trigger after_create_commit
puts "After running create callbacks"

blog_post.save
```

You’ll notice that “Notifying subscribers after successful commit” does not print after `run_callbacks(:create)` but it *will* print when `blog_post.save` is called because the `save` method wraps the creation in a transaction which triggers the deferred callbacks on success. The output would look something like:

```
After running create callbacks
Notifying subscribers after successful commit
```

This confirms that `run_callbacks(:create)` bypassed our `after_create_commit` as it’s not intended to trigger commit-related lifecycle events. So, how *do* we go about triggering this type of callback specifically if we need to?

The key lies in understanding that to trigger an `after_create_commit`, we essentially need to simulate a database transaction commit. Rails provides the `transaction` method on `ActiveRecord::Base` (and therefore all of your models) to help. We essentially need to wrap our `run_callbacks` in a transaction and then commit. This simulates the process that happens when we use methods like `save`, or `create`.

Here's a more useful code snippet demonstrating how to trigger `after_create_commit` using a transaction block and, in this case, a custom transaction method:

```ruby
class BlogPost < ApplicationRecord
  after_create_commit :notify_subscribers

  private

  def notify_subscribers
    puts "Notifying subscribers after successful commit"
    # Simulate a notification service call
    sleep 0.1
  end

  def simulate_create_transaction
    BlogPost.transaction do
      run_callbacks :create do
        # You could place additional logic here, for example setting attributes
        # or any other logic relevant to the creation process
        self.title = "Simulated Post" # Example: Set a title within the transaction
        self.content = "Simulated content"
      end
    end
  end
end
```
And to use it:

```ruby
blog_post = BlogPost.new
blog_post.simulate_create_transaction
puts "After simulated transaction"

```
The console output now *will* show:

```
Notifying subscribers after successful commit
After simulated transaction
```

This demonstrates how a custom transactional method lets you force the execution of callbacks within the correct transaction context. When the transaction completes, the `after_create_commit` callback is automatically executed. This is a much more deliberate approach than simply calling `run_callbacks(:create)`.

However, let’s consider another common situation which is where a class has to trigger these callbacks on an already existing instance. In that case, you will need to force the commit to be performed when no save method is used, or if you wish to bypass the `save` method entirely but want the callback.

In such situations, here's an adaptation that works on an existing instance:

```ruby
class BlogPost < ApplicationRecord
  after_commit :notify_on_any_commit, on: [:create, :update]

  def notify_on_any_commit
    puts "Notification triggered on commit"
  end

  def simulate_update_and_force_commit
     BlogPost.transaction do
        self.title = "Updated Post Title"
        self.content = "Updated post content"
        run_callbacks :update
        # We do *not* call save here, but must commit to trigger callback.
        connection.execute "COMMIT;"
     end
  end

end

```

And the usage:

```ruby
post = BlogPost.create(title: "Original", content: "Content")
post.simulate_update_and_force_commit
puts "after custom commit"

```

This example, although less common, highlights that the key is the `connection.execute "COMMIT;"` which forces the transaction to finalize even when we have bypassed save. The `on: [:create, :update]` in the `after_commit` allows for this method to be used on more lifecycle events.

These practical examples illustrate the key principle: `after_commit` (including the variants like `after_create_commit` and `after_update_commit`) callbacks are tied explicitly to the database transaction lifecycle and require that transaction to commit. Directly calling `run_callbacks` does not provide the proper context, and thus requires wrapping the callback logic within a transaction block, and optionally, a `connection.execute "COMMIT;"` if no `save` method is called or required.

If you're seeking deeper insights into this particular facet of Rails, I recommend diving into the source code of `ActiveRecord::Transactions`. You can start by examining the `transaction.rb` file in the Rails repository. Additionally, reading Chapter 4 of *Agile Web Development with Rails 7* by Sam Ruby et al. provides a comprehensive overview of callbacks and transactional behavior within ActiveRecord. Finally, I would also suggest the excellent *Crafting Rails Applications* by Jose Valim, which includes many details on ActiveRecord and callbacks. These resources will help solidify your understanding of transactions and the intricacies of callbacks in a Rails application. You may find some older relevant talks from conferences or blog articles; just ensure they are aligned with the version of Rails you are using, as behaviours sometimes change over time.
