---
title: "Why is Rails 7 RestClient::Not found in find_each loop updating an avatar from Microsoft Graph?"
date: "2024-12-14"
id: "why-is-rails-7-restclientnot-found-in-findeach-loop-updating-an-avatar-from-microsoft-graph"
---

so, you're seeing `restclient::notfound` inside a `find_each` loop when updating avatars from microsoft graph with rails 7, huh? yeah, i've been there, wrestled with similar beasts in the past. let me try and break it down from my experience; i think i've got a pretty good handle on what's likely happening here.

first off, `find_each` is your friend for batch processing, but it has a specific flow. it fetches records in batches, not all at once. inside that loop, you're making requests to microsoft graph, which is where the `restclient` bit comes in. the `restclient::notfound` error screams that microsoft graph is telling you, "hey, the resource you asked for, it’s not here." the trouble is, *why* is it saying this only sometimes, and seemingly within the loop? well, let's untangle this ball of yarn.

my guess, based on past adventures with similar apis, is that you’ve got a race condition or stale data somewhere. here's how i picture it unfolding:

1.  **the setup:** your `find_each` loop gets a batch of, say, 10 user records.
2.  **the request:** for each user, your code constructs a request to get or update the user's avatar from microsoft graph.
3.  **the microsoft graph gremlins:** here's where the trouble hits. microsoft graph isn't instantaneous. when it gets a request to update an avatar, it’s usually a multi-step process, maybe a delay from when you get a response that indicates it has been created or updated till it is ready. when you ask for the new avatar, immediately after that request in the next loop cycle, the old avatar record is not there, or it does not exist yet and returns a 404 not found error.
4.  **the not found:** you make another request to fetch the very avatar you think you've just updated, and if you do it too soon, graph replies, "404, dude. nothing here." hence, the `restclient::notfound`.

the whole thing might be intermittent, because some updates might process faster than others, or the network might be flaky. timing can play a critical role in all of this. it’s kinda like trying to catch a raindrop after you’ve just thrown a bucket of water in the air. the raindrop is there, you know it is, but you are not fast enough to catch it in that same exact time and place. or well, maybe not like that because i said no analogies.

here’s how i’d approach debugging this. first, let’s look at the code pattern, i assume you might be doing something like this:

```ruby
User.find_each(batch_size: 10) do |user|
  # assuming you have a method or service to interact with microsoft graph
  avatar_url = microsoft_graph_service.update_user_avatar(user)
  user.update(avatar_url: avatar_url) # this update may happen before graph processes
end
```

this basic structure looks fine, but it does not handle the delay from graph. after `update_user_avatar` the code goes ahead and tries to update the user record. this will lead to the problem of stale data.

now, here’s an idea, try delaying the fetching of the avatar after the update, a small delay could help. in ruby, we can use `sleep` but this isn't ideal for production but for debugging and testing you could test something like this:

```ruby
User.find_each(batch_size: 10) do |user|
  # assuming you have a method or service to interact with microsoft graph
  avatar_url = microsoft_graph_service.update_user_avatar(user)
  sleep(2) # wait for 2 seconds (adjust as needed)
  new_avatar_url = microsoft_graph_service.get_user_avatar(user)
  user.update(avatar_url: new_avatar_url)
end
```

this snippet adds a `sleep(2)` after the update request. this could give microsoft graph enough time to process the avatar update, and the subsequent fetch will likely return the actual avatar. however, this is a hack, not a solution. sleep is never good in production.

let’s look at a better way to handle this. we need a way to handle the asynchronous nature of the api. here's an approach using a retry mechanism with exponential backoff:

```ruby
def fetch_avatar_with_retry(user, max_retries: 5, delay: 0.5)
  retries = 0
  begin
    avatar_url = microsoft_graph_service.get_user_avatar(user)
    return avatar_url
  rescue RestClient::NotFound => e
    retries += 1
    if retries > max_retries
      raise e # re-raise the exception if max retries exceeded
    else
      sleep(delay * (2**retries)) # exponential backoff
      retry
    end
  end
end

User.find_each(batch_size: 10) do |user|
  # assuming you have a method or service to interact with microsoft graph
  avatar_url = microsoft_graph_service.update_user_avatar(user)

  new_avatar_url = fetch_avatar_with_retry(user)
  user.update(avatar_url: new_avatar_url)
end
```

this updated snippet wraps the `get_user_avatar` call in a `fetch_avatar_with_retry` method. it tries to fetch the avatar up to 5 times, increasing the delay between retries exponentially. this is much more robust than a simple `sleep`.

a better, perhaps more complicated, solution would involve checking the `status_code` from the initial update request, and to avoid errors and not do retries unless necessary. we need to check if the update was successful or not. we can do that with error handling around the update to see if it was ok. then we do not even retry at all. let’s look at this:

```ruby
User.find_each(batch_size: 10) do |user|
  # assuming you have a method or service to interact with microsoft graph
  begin
      update_response = microsoft_graph_service.update_user_avatar(user)
    if update_response.code != 200 && update_response.code != 201
      # log some error
      puts "failed to update user avatar, code: #{update_response.code}"
      next
    end
    new_avatar_url = microsoft_graph_service.get_user_avatar(user)
    user.update(avatar_url: new_avatar_url)

  rescue RestClient::Exception => e
    # handle the error. log it.
    puts "something bad happened: #{e.message}"
  end
end

```

this way if the update fails we don't try to get the avatar at all and we save ourselves some time and avoid a possible error.

now, a word of advice. relying solely on retries might cover up underlying issues. it’s always better to understand why the 404 happened in the first place. could it be that you're requesting an avatar too soon? did you use the correct endpoints from microsoft graph? double check the documentation of the api.

also, be mindful of rate limiting. excessive requests, especially retries, could trigger rate limiting by microsoft graph, which is another can of worms you don't want to open. monitor your api usage and implement appropriate rate limiting mechanisms on your end.

instead of throwing links at you, i would recommend looking into some books about distributed systems for handling errors. "designing data-intensive applications" by martin kleppmann is a good one, it goes deep into eventual consistency, which applies here. for more hands-on advice on resilient coding patterns, "release it!" by michael t. nygard is also solid. also the microsoft graph documentation. it should be your bible for interacting with the graph api. and remember, read the fine print.

anyways, that’s been my experience. i hope some of this resonates and gives you a solid direction to look in, let me know if it helps. the world of apis is full of interesting corner cases, just try to not fall into them.
