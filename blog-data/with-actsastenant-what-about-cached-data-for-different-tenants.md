---
title: "With Acts_as_tenant, what about cached data for different tenants?"
date: "2024-12-15"
id: "with-actsastenant-what-about-cached-data-for-different-tenants"
---

ah, i see the conundrum. tenant-specific caching with `acts_as_tenant`, that's a classic. been there, done that, got the t-shirt (and a few late nights debugging, haha). it's one of those things that seems straightforward on paper but gets tricky in practice real fast when you start scaling or adding complexity to your app.

so, fundamentally the issue revolves around the fact that, with `acts_as_tenant`, you're essentially partitioning your data by tenant. that's great for isolation and security, but it throws a spanner in the works when it comes to caching. most caching solutions aren't inherently tenant-aware. they don't know to store a different version of the same cached key for tenant a vs. tenant b. if not handled carefully, this leads to users seeing data that belongs to another organization which is bad and it's almost as bad as having a bad security breach.

in my past life i worked for a company doing saas for medical data and boy oh boy the number of edge cases we encountered. that specific issue was quite painful to fix for a specific use case for patient data related to specific doctors inside a specific clinic. i remember i had to use a lot of conditional statements in the view with cached fragment which was a horrible solution i must say. we were using the old version of rails at the time, rails 4.2, so the caching was way simpler than what is today and that added a lot of complexity to the way we solved the issue.

let’s talk about some common approaches that i've seen and implemented, and some of their gotchas.

the simplest approach is to include the tenant id directly in the cache key. it's essentially the bare minimum required to make things work. here's how that might look using the rails cache:

```ruby
def fetch_tenant_specific_data(tenant_id, data_id)
  Rails.cache.fetch("tenant_#{tenant_id}_data_#{data_id}", expires_in: 12.hours) do
    #  some database query
    Data.where(tenant_id: tenant_id, id: data_id).first
  end
end
```

notice the use of string interpolation in the key, specifically `tenant_#{tenant_id}_data_#{data_id}`. it ensures that you are retrieving only the data related to the current tenant. this is often a good starting point and will cover the majority of cases. but there is always room for improvement.

there are other aspects to consider besides the simple cache.fetch that usually involves conditional statements in the views for cache fragments. that was something very common in my experience with the old rails and old apps.

now, what if your caching logic isn't quite as simple as just fetching a database record by id? what if you need to cache the result of a more complex query, or a collection of objects? that's where you might need to introduce a different approach and introduce some cache key conventions.

here's an example of that approach:

```ruby
def fetch_tenant_specific_posts(tenant_id, query_options)
  cache_key = [
    "posts",
    "tenant_#{tenant_id}",
    query_options.sort.to_s.hash  # Using a hash to uniquely identify the query options
    ].join(":")

  Rails.cache.fetch(cache_key, expires_in: 12.hours) do
    Post.where(tenant_id: tenant_id).where(query_options).to_a
  end
end
```

in this example we are generating the cache key based on different aspects including the tenant and the query itself which it's a good approach. the `query_options.sort.to_s.hash` part is important because it ensures that two queries with the same options will result in the same cache key regardless of the key order which is a common problem if you are not careful.

this pattern can be helpful in more complex scenarios in the business side of your app, where parameters or filters for example will add complexity to the queries.

now let’s address another common issue: invalidating cached data. just appending the tenant id might not be enough if you're changing data often. one common approach for cache invalidation is to rely on callbacks of your active record models. let's say you have a `post` model:

```ruby
class Post < ApplicationRecord
  acts_as_tenant(:tenant)

  after_save :invalidate_cache
  after_destroy :invalidate_cache

  def invalidate_cache
    Rails.cache.delete("posts:tenant_#{self.tenant_id}:#{self.class.all.to_sql.hash}")
  end
end
```

this is a good solution but has a drawback because you are invalidating a whole collection if you are caching it. here the cache invalidation is quite aggressive. a more specific approach could be to store the id of the record in the cache key if you need to be more precise with your cache invalidation. this particular example has a drawback as you are using the hash of the query, so any change to the table schema will also invalidate the cache if you are not careful. but for simplicity it's good enough for this example.

now, there are some edge cases you need to be aware of. for instance if you are using sidekiq and jobs, ensure that the `acts_as_tenant` is set properly before using the cache so your job doesn't accidentally use data for the wrong tenant or worse, saves it to the wrong tenant. i've had that issue once and it was quite a problem to debug. it was like trying to find a needle in a haystack of background jobs but that's a story for another day.

also, it is worth noting that the cache key size matters. if you are building huge cache keys, maybe because of very complex query options, you might run into problems with the caching system in place. so keep that in mind.

there are more advanced techniques to consider. for very large systems, a distributed cache like memcached or redis is pretty standard, and they offer some more tools for handling invalidation and scaling. it's a whole different discussion that i won't dive deep now. and depending on your requirements and how critical is your caching for performance you might consider looking into cdn services and also graphql because it gives you more control over what you are actually caching. you will save a lot of bandwidth if you are serving only the fields required for the request instead of rendering huge json files.

if you really want to go deep into caching i would suggest reading *patterns of enterprise application architecture* by martin fowler, it contains a lot of valuable information for building applications and how to use the cache for different use cases. *database reliability engineering* by lilla gordon and betsy beyer is also a great resource, which also contains a very good chapter about caching and the different trade-offs you need to consider when designing your system.

so, there you have it: some basic tenant-aware caching approaches, the gotchas and some extra info. always remember to test your caching logic thoroughly, especially in a multi-tenant environment. it’s easy for things to go awry if you’re not careful. it's the kind of problem that might cause a hair loss. i hope this was helpful and clear, good luck.
