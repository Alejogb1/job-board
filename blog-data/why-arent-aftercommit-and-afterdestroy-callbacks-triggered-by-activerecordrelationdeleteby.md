---
title: "Why aren't after_commit and after_destroy callbacks triggered by ActiveRecord::Relation#delete_by?"
date: "2024-12-23"
id: "why-arent-aftercommit-and-afterdestroy-callbacks-triggered-by-activerecordrelationdeleteby"
---

Alright, let's talk about something that's tripped up more than a few people, including myself, back in the days of Rails 3.4 when I was knee-deep in a project that involved managing a complex user hierarchy. The question, as posed, is about why `after_commit` and `after_destroy` callbacks aren't triggered when using `ActiveRecord::Relation#delete_by`. It's a nuanced issue rooted in the way ActiveRecord operates at a lower level, and the distinction between instance-level operations and bulk operations.

Essentially, the core reason boils down to the fact that `delete_by` operates directly at the database level using SQL’s `DELETE` statement. It bypasses the conventional ActiveRecord object instantiation, validation, and lifecycle mechanisms. Think of it this way: when you call `destroy` on an ActiveRecord object, you're invoking a series of events within the ActiveRecord framework – callbacks are specifically designed to respond to those lifecycle events. However, when you use `delete_by`, you're essentially giving a SQL command to the database without involving ActiveRecord in the traditional sense, thus bypassing those hooks entirely. It’s a performance optimization, sacrificing the safety and thoroughness of instance-level operations for speed.

Let’s illustrate with some code. Imagine we have a `BlogPost` model, and we want to track whenever a blog post is deleted.

```ruby
# app/models/blog_post.rb
class BlogPost < ApplicationRecord
  after_destroy :log_deletion

  def log_deletion
    puts "Blog post with id #{self.id} was destroyed via instance method."
  end
end

# Example usage with instance-level delete
post = BlogPost.create(title: 'Test Post', content: 'This is a test')
post.destroy
# Expected output: "Blog post with id <id> was destroyed via instance method."
```

This first example showcases the `after_destroy` callback behaving as expected. We create a `BlogPost` instance, and then use its `destroy` method. Because `destroy` operates at the instance level, ActiveRecord creates the object, performs the deletion within its object lifecycle, and triggers callbacks. Let's see that contrast with the use of `delete_by` where we will not see any output from the callback.

```ruby
# Example usage with delete_by
BlogPost.create(title: 'Another Test Post', content: 'This is another test')
BlogPost.delete_by(title: 'Another Test Post')
# No output from callback
```

As you can see, with `delete_by`, the `after_destroy` callback is never triggered. ActiveRecord effectively issues a `DELETE FROM blog_posts WHERE title = 'Another Test Post';` directly to the database, and that's that. The model's instance methods and callbacks are completely bypassed. It's important to understand that this behaviour is by design, intended to make these operations as performant as possible when handling a large amount of data.

Now, one might wonder, what about `delete_all`? It behaves in a similar way to `delete_by`. Let's modify the code slightly to test this scenario.

```ruby
# Example usage with delete_all
BlogPost.create(title: 'Yet Another Test Post', content: 'This is yet another test')
BlogPost.where(title: 'Yet Another Test Post').delete_all
# No output from callback
```

Here we create a post and then we use `.where` in conjunction with `.delete_all`. Again, we do not see output from the `after_destroy` callback. The underlying database operation is the same, the database gets a direct `DELETE` command, and ActiveRecord object life cycle events are bypassed.

In a practical situation, this caused me some headaches when I was relying on `after_commit` to propagate deletions to other parts of the application. We were dealing with a large dataset, and using `delete_by` was a performance requirement. The solution we eventually settled on wasn't to try and force `delete_by` to invoke callbacks. That's just the wrong approach, and there are more suitable mechanisms.

Instead of trying to make a square peg fit a round hole by bending `delete_by`, we chose to use a different mechanism that met the performance requirements. We opted for a combination of a database trigger and an asynchronous job queue. The database trigger, a PL/pgSQL function for our Postgres database, would execute after a successful `DELETE` statement on the appropriate table. This trigger would then enqueue a job that was picked up by a background process and handled the task, which we would have previously handled with a callback. This approach separated the database operation from the callback's logic.

For anyone facing this issue, it’s essential to examine the available tools and understand why ActiveRecord behaves this way. You should not expect the callbacks to be triggered with `delete_by` or `delete_all`. If you require callbacks to function, you should be using instance methods that respect the ActiveRecord lifecycle, such as `destroy`.

If you are looking for further clarification on this and related subjects, I highly suggest that you take a look at the following resources:

*   **"Agile Web Development with Rails" by Sam Ruby, Dave Thomas, David Heinemeier Hansson:** This book, although covering general Rails development, thoroughly explains the principles behind ActiveRecord and its lifecycle. It delves into the design choices behind the framework and the distinction between instance-level and bulk operations, which directly applies to this issue.
*   **"Patterns of Enterprise Application Architecture" by Martin Fowler:** This book covers design patterns applicable in all situations, however, it provides valuable insight into data access, and batch operation mechanisms, which will prove beneficial in understanding when and why direct SQL execution is favorable or when a callback might be more applicable.

In closing, the design of ActiveRecord consciously prioritizes performance in bulk operations. Methods like `delete_by` bypass instance-level checks, validations, and callbacks, which are all inherent parts of the ActiveRecord object lifecycle, in favor of direct database interaction. The practical lesson here is to carefully consider your needs. If you require callback execution, use instance methods. If you need maximum performance, `delete_by` or `delete_all` are appropriate, but you must also develop alternative means for ensuring critical follow-up processes are still performed, as needed.
