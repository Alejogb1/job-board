---
title: "Any way to run a block right after Chewy `update_index`?"
date: "2024-12-15"
id: "any-way-to-run-a-block-right-after-chewy-updateindex"
---

alright, so you're after a way to trigger some code directly after chewy's `update_index` method finishes its job, i get it. i've been there, staring at the screen, wondering how to chain things efficiently after a data indexing process. it's a pretty common need, actually. you want that clean, atomic sequence of operations, data in the index, and then some other process fired immediately.

from my own experience, and i'm talking like, back when elasticsearch was still kinda the new kid on the block for me, i messed this up countless times. i was working on a social media analytics tool (don’t ask, it was a mess) and we needed to update user profile counts in our relational database the moment their activity got indexed. our first naive attempt was just running the updates straight after the chewy `update_index` call. spoiler: that was terrible. race conditions, updates being overwritten, the works. we had phantom users that were both popular and not popular simultaneously and that was not ideal for user engagement.

anyway, let's get into the options that actually work. chewy, by itself, doesn't provide a direct 'post-index' callback or hook that's exposed to the user to do what you ask. instead, we have a few common methods, and the best approach heavily depends on how tightly coupled you need this process to be.

the most direct, though sometimes not the most elegant, method is to simply add your code right after you call `update_index`. now, this seems almost too obvious, and it is, but when you're trying to solve the issue the quickest way, that can work. this is not ideal in most real-world applications but if you really need to, this is how the code could look:

```ruby
  # assuming `my_model` is a chewy indexed model
  MyModel.find(some_id).update_index
  # your post-index code here
  puts "index updated, running some other code now!"
  # this could be calling a method
  MyModel.post_update_process(some_id)
```

as i mentioned, there are pitfalls with that. what happens if the indexing itself fails? your post-index code will probably fire anyway, which can lead to inconsistencies. plus, imagine if the index is being updated in multiple places. having the post-index logic repeated everywhere just does not scale. that’s the reason why i always try to avoid this.

a slightly better approach, at least in terms of organisation, is to extend your model with a callback on `after_commit`. but, there's a *big* caveat here: `after_commit` fires *after* the database transaction is successful, and this isn’t the same thing as having the data indexed. your data may be in the db, but elasticsearch may not know about it yet. this creates a new kind of race condition. this approach may give you the *feeling* that you solved the issue but you didn't. i've seen it a lot and it's so hard to debug this later on.

```ruby
  class MyModel < ApplicationRecord
    include Chewy::Index::Concern
    after_commit :run_post_index_logic, on: :update

    def run_post_index_logic
      # WARNING! this may fire BEFORE index is actually updated
      MyModel.post_update_process(self.id)
    end
  end

  # example usage:
  MyModel.find(some_id).update(name: "new name") # this will trigger update_index and later the callback

  # here goes a possible method on that model
  def self.post_update_process(id)
      puts "running after commit: model id #{id}."
      # do other stuff here
  end
```

the real way to do this reliably is to decouple the indexing from the post-index code by using a background job queue system. this is definitely the best way and it requires you to introduce another technology to your project, so that's something that you should assess, but generally i think it's worth the effort. for me it's the most robust and scalable way to approach it.

when i was doing that analytics tool, we did end up moving to a message queue setup. once the index was updated, we pushed a job onto a message queue using something like `sidekiq` (but any queue system like resque or delayed_job will do just as well). a worker would then pull that job off the queue and execute the post-index stuff. now that we have proper messaging, things are decoupled, the logic is in one place, and the jobs run reliably.

that’s how that code looks, as an example:

```ruby
  # assuming you have a worker configured to handle the job
  class MyModel < ApplicationRecord
    include Chewy::Index::Concern
    after_commit :enqueue_post_index_job, on: :update

    def enqueue_post_index_job
      # now we are sending a job to a worker
       PostIndexWorker.perform_async(self.id)
    end
  end

  class PostIndexWorker
    include Sidekiq::Worker
    def perform(model_id)
        puts "performing post index action id: #{model_id}"
        # now you can execute after index process reliably
      MyModel.post_update_process(model_id)
    end
  end
  # example usage:
  MyModel.find(some_id).update(name: "new name") # this will trigger update_index and queue the worker

```

using a message queue is the more robust way to do that. it's a bit more work to set up initially, but it really simplifies things in the long run and it is more fault tolerant. if the post-index process fails the first time, it can be retried with a queue system, you won't have this issue using the previous approaches.

one last thing, if your post-index process is something relatively simple and quick, you might find that just doing it synchronously after `update_index` works fine, as in the first example. but as soon as things grow in complexity, move to a message queue solution. you won't regret that decision, trust me. remember, premature optimization is the root of all evil. or was it root privileges that caused issues? well, anyway, i think you get the point.

for further reading, i'd recommend checking out the "enterprise integration patterns" book by hohpe and woolf, if you want to know the architecture reasons behind why this is the most robust way. it goes into detail about decoupling processes. also, "distributed systems: concepts and design" by coulouris et al gives a solid foundation on distributed systems design (obviously). another good read is "understanding distributed systems" by roberto vitillo, the title is self explanatory and may be of great help to understand some subtleties about all this.

also, do not forget to study in depth the documentation on chewy itself, especially if you are using a custom index, and the background job queues system you decide to use. understanding how those works in detail is important to get a robust solution.

i hope this helps. let me know if you have more questions.
