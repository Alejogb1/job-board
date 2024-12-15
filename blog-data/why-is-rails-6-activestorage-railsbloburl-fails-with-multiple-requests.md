---
title: "Why is Rails 6 ActiveStorage rails_blob_url fails with multiple requests?"
date: "2024-12-15"
id: "why-is-rails-6-activestorage-railsbloburl-fails-with-multiple-requests"
---

let's get down to brass tacks, it looks like you're running into a classic race condition with rails 6 activestorage's `rails_blob_url`. i've been there, done that, got the t-shirt and the lingering feeling of existential dread. this isn’t some weird edge case, it's a common gotcha that can absolutely kneecap your application if you aren’t careful. i've seen it bring down perfectly good e-commerce sites during sales events, it’s not pretty.

the root of the problem lies in how `rails_blob_url` (and its sibling `url`) generates those temporary signed urls. under the hood, they use a combination of the blob's id, a timestamp, and a signature, all hashed together to produce a unique, temporary url that grants access to the underlying blob stored in your cloud service like aws s3, google cloud storage or azure blob storage. now, what happens when you fire off multiple requests for the *same* blob close together?

well, picture this. your app is humming along, a user clicks a button to download a file, and *boom* multiple identical requests are sent in very rapid succession because of some impatient javascript or maybe a rogue click event handler. all these requests hit your rails server concurrently. now the `rails_blob_url` is called multiple times *almost at the same time* for the same blob. since the urls are signed, if the url is generated using the database info that is not yet commited, the generated url could be invalid or return a 403 because it doesn't match the underlying blob signature. the rails server will go through the process, but because these requests all occur nearly simultaneously, they will often end up with the same timestamp, and potentially the same unique signature and some other internal states might not yet be fully updated and visible for the other requests.

this is a case where rails, by default, uses the `after_commit` hook, which means that the record is not fully in the database until *after* the commit has completed. when you fetch a record before the commit, this can lead to a situation where multiple requests end up generating urls based on the state before the database change has been fully applied. if this database state changes after the url is generated then that url is likely to fail.

the database record hasn't yet completely synced through the whole system when the second request tries to generate the url which will generate a 403 or fail to load. sometimes it works, sometimes it doesn't. it's the worst kind of problem because it's intermittent, which makes debugging it a real pain.

in my early days working with rails, i once had a similar issue which took me almost a week to diagnose, it turned out that the problem was the database caching that was active and the `after_commit` callback, some very specific combinations of race conditions and caching invalidated the database record between requests. i learned more than i wanted to about the internals of rails active record that week.

one common mistake to avoid is using `rails_blob_url` or `url` inside loops. this creates a barrage of requests which exacerbate the problem. always try to generate your urls upfront and then pass them around. also be careful with the `cache_classes` configuration in rails. setting this to false in development can mask race conditions that will appear in production, so it's better to use the default, otherwise you could be shooting yourself in the foot later.

so, how do we fix this mess? the primary solution is to ensure the underlying database record and blob are fully committed and available *before* we try to generate the url. here are a few options:

1. **explicit commit:** you can force a commit of the active record transaction before generating the url. this can work, but it is not a full proof strategy, and the transaction could be invalid in another part of the code if you are not careful, however this is a valid option for specific cases:

```ruby
  def generate_blob_url(blob)
    ActiveRecord::Base.transaction do
      blob.reload # this makes sure the blob is up to date
      url = rails_blob_url(blob, only_path: true)
      raise ActiveRecord::Rollback unless url # just to ensure no changes in case of problems
      url
    end
  end
```

2. **use background jobs:** instead of generating the url synchronously, you can offload it to a background job using a gem like sidekiq or delayed_job. this ensures the initial request isn't blocked by the generation and prevents the race condition, this technique also help you to handle errors more gracefully. This is the more robust way to handle this race condition problem.

```ruby
  # app/jobs/generate_blob_url_job.rb
  class GenerateBlobUrlJob < ApplicationJob
    queue_as :default

    def perform(blob_id)
      blob = ActiveStorage::Blob.find(blob_id)
      url = rails_blob_url(blob, only_path: true)
      # publish the url using websockets or similar
      ActionCable.server.broadcast "blob_urls", { blob_id: blob_id, url: url }
    end
  end

  # your controller/model
  def request_blob_url(blob)
    GenerateBlobUrlJob.perform_later(blob.id)
  end

```
3. **caching generated urls:** if your blob doesn't change frequently, you could cache the generated url for a specific period. this way subsequent requests won't trigger multiple url generation calls. this also makes your application more performant. here's an example using rails cache:

```ruby
  def generate_cached_blob_url(blob)
    Rails.cache.fetch("blob_url_#{blob.id}", expires_in: 1.hour) do
        rails_blob_url(blob, only_path: true)
    end
  end
```

these approaches can help you to mitigate the issues with generating the urls, but you need to be aware of when your blobs change, otherwise your cached url can be stale, or the user can download the old blob.

finally, if you find yourself needing a deep dive into the intricacies of activestorage, i highly recommend the official rails documentation. it's a great resource, also, some books like "agile web development with rails" by sam ruby and dave thomas, or "rails 7" by noel rappin, can give you more in depth knowledge about activestorage and all the other features available in rails. they will cover the internals of activestorage and also provide some background on how the database and concurrency work in rails.

remember, this isn't a bug per se, it's more like a quirk of how rails handles asynchronous processes and the `after_commit` hooks which generate the urls.

i know this was quite a wall of text but i have been on your shoes in the past and i hope it helps you to handle the problem better.

oh, and one last thing, what do you call a lazy kangaroo? pouch potato. :)
