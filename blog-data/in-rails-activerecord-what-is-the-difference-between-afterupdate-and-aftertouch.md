---
title: "In Rails ActiveRecord: what is the difference between after_update and after_touch?"
date: "2024-12-15"
id: "in-rails-activerecord-what-is-the-difference-between-afterupdate-and-aftertouch"
---

alright, so you're asking about the difference between `after_update` and `after_touch` in rails activerecord. i've definitely been there, staring at these two thinking "what's the subtle gotcha here?" let me break it down, drawing from my own battles with these callbacks.

first off, let's be clear: they're both activerecord callbacks, meaning they're methods that get executed at specific points during the lifecycle of an activerecord object. they're your hooks for doing things automatically based on changes in your database models. but their triggers are different, very different.

`after_update`, as the name suggests, fires after an existing record in the database is updated. crucial thing is: it's triggered only when the model’s attributes have actually been modified, which means a change has occurred in your database. for instance, if you had a `user` model and you changed the `email` attribute and saved that user, that's going to set off an `after_update` callback. if, however, you load the user from the database, make no changes, and call `save`, the `after_update` callback will not trigger. it only cares if something changed during that save operation. this is where i've seen people get tripped up. they expect it to run even if nothing’s changed, which it doesn’t. that’s one very specific issue where i had to rewrite a piece of my system after realizing this and wasting a day troubleshooting a particularly subtle bug in my code.

i once had a system where we were logging user activity. we had an `after_update` callback on the user model that would add an entry to the activity log. everything seemed fine until we started testing edge cases and realized that if we loaded a user, did not modify it, and saved it, nothing was logged. because, logically, there was no update in database and therefore the callback was not fired. we ended up having to create an extra method to force the creation of the log in the cases when we were doing operations that required a log, no matter what. which brings me to `after_touch`.

`after_touch` is a different beast. it triggers when you ‘touch’ a record. touching a record, in activerecord terms, means updating the `updated_at` timestamp column on a record. this happens regardless of whether other attribute changes are made or not. so even if you load an object and call the touch method, with or without a save the `updated_at` column in the database gets updated, and this will fire the `after_touch` callback, even if no other attributes were modified. this callback is specifically geared toward scenarios where you want to trigger behavior based on any ‘refresh’ of an object in the database. it is as if, at a superficial level, you are updating the object, but not changing a single piece of data, which can be weird if you don't think it carefully.

i remember a project where we had a nested comment system, and the main object was a `post`. every time a comment was either created or deleted, we needed to update the `updated_at` field of the associated post, we used `after_touch` on the comment model. this ensured that caches or anything that would use the updated time of the post was always invalidated when a comment changed or was modified, and it worked brilliantly. it didn't matter if the comment itself was changed or just added or removed, the post was always "touched", refreshing it. the code would run something like:
```ruby
class comment < applicationrecord
  belongs_to :post, touch: true
  after_destroy do
    post.touch
  end
  after_create do
    post.touch
  end
end
```
here you can see how you could handle that.

the `touch: true` option on the association automatically makes sure that anytime a change occurs in the child object the parent one gets touched. you can force a touch on any associated object by writing code like the following example too, this can come handy in certain cases.
```ruby
  def some_method
    my_post = post.find(1)
    my_post.touch
    # or
    my_post.update(updated_at: time.now)
    # these two are almost identical
  end
```
the above examples are two very common ways to use it. now, let’s compare the two, and dive into practical examples and when you'd use one or the other.

**`after_update`:**

*   **trigger:** when attributes are modified and saved. no change in database means, no trigger.
*   **use cases:**
    *   logging changes: creating audit trails of what was changed in your database records. that’s what my old logger did, it was really useful to catch the subtle bugs that would’ve been very hard to trace without it.
    *   sending notifications: alerting users when something about their data changes, like a new email confirmation or when their password has been modified.
    *   data synchronization: updating associated records with changes or external apis when it’s actually updated not just ‘touched’.
    *   calculating metrics: when a property changes you may want to run a complex calculation and storing it on the model for further use.
*   **example:**
```ruby
class user < applicationrecord
  after_update :log_user_update

  private

  def log_user_update
    # only will execute if user was actually updated
    activitylog.create(user_id: id, action: 'user updated', details: changes.inspect)
  end
end
```
here, `log_user_update` only fires after a database update has occurred, and `changes.inspect` will show the actual changes made. this is useful for auditing purposes and not logging things that have not changed, this is the key.

**`after_touch`:**

*   **trigger:** when the `updated_at` timestamp column is updated, regardless of other modifications or if other attributes were modified.
*   **use cases:**
    *   cache invalidation: as i mentioned before. making sure the changes are reflected immediately and outdated cache is not shown.
    *   updating denormalized data: when you need to refresh related models or aggregations based on the last update time, this is what our comment system needed.
    *   triggering background jobs: starting background process based on when things are ‘refreshed’ this can be anything that is not directly linked to the specific data that was changed, but the object itself, it's quite a subtle difference.
    *   refreshing data in ui: you may not have changed the data, but refreshed it by touching it. think of it as a soft-refresh of the object.
*   **example:**
```ruby
class post < applicationrecord
  after_touch :invalidate_cache

  private

  def invalidate_cache
    rails.cache.delete("post-#{id}")
  end
end
```
this example shows how you would make sure the cache is invalidated when the post is “touched”, either by itself or by an associated object.

so, which one should you use? it really depends on what you want to do.

if you need to react to actual attribute modifications in your database, use `after_update`. it is for when you are dealing with data changes and that is the reason you should execute a side effect.

if you need to react to any update to the `updated_at` timestamp of a record (even if no attributes are modified), use `after_touch`. it is more linked to cache invalidation or refresh operations.

here is a funny analogy that i can think of: imagine `after_update` as a mailman who only rings your doorbell if he has a new package for you, if you just come and open the door to him, he won't ring. now, `after_touch` is like someone checking if you're alive, they just peek at the door and update it, it doesn't matter if they have a package or not. this doesn't make sense, but it's kinda funny.

one common mistake is expecting `after_update` to trigger when it's not going to, specially if you are loading and saving without changing. another is using `after_touch` when you really need to react to actual data changes, not just timestamps. it's a very specific scenario.

when debugging, i’ve found it useful to add a logger inside both callbacks. this helps you see when exactly each one is being triggered, and what data is being passed to them. it's the best way to figure out what's going on.

for more detailed information on rails callbacks, i'd recommend the activerecord callbacks guide on the official rails website, it has helped me a lot in the past. additionally, the “agile web development with rails” book is a valuable source of information on activerecord and its behavior, it will help you grok everything with a bit more perspective and understanding of the framework. these two resources have helped me in the past, and i still go back to them regularly when i’m coding.

i hope this helps. i’ve been there, i've scratched my head on this, and i know that these details matter a lot.
