---
title: "Why with Rails ActionCable: Is there a way to skip connection authentication for specific channels?"
date: "2024-12-15"
id: "why-with-rails-actioncable-is-there-a-way-to-skip-connection-authentication-for-specific-channels"
---

yeah, i've been there, wrestling with actioncable and authentication. it's one of those things that seems simple at first, but then bam, you're spending way more time than anticipated. the short answer is no, actioncable itself doesn't have a built-in, direct way to skip authentication for specific channels. it's designed with a 'gatekeeper' approach: if the initial connection isn't authorized, no channels are accessible. but, and it's a big but, there are workarounds.

let me share what i've found, coming from a place of frustration, i must say, after spending some late nights trying to make a similar setup work. in a past project, i had a public dashboard that displayed live data, like server stats and uptime, alongside some authenticated user-specific channels. i initially thought, oh, just skip auth on the public channel, simple, but actioncable doesn't really operate that way.

the problem stems from the connection setup in `app/channels/application_cable/connection.rb`. in there, you usually have code like this:

```ruby
module ApplicationCable
  class Connection < ActionCable::Connection::Base
    identified_by :current_user

    def connect
      self.current_user = find_verified_user
    end

    private

    def find_verified_user
      # authentication logic here, usually based on cookies or headers
      # for example, using devise:
      #  if verified_user = env['warden'].user
      #    verified_user
      #  else
      #    reject_unauthorized_connection
      #  end
    end
  end
end
```

this block of code runs *before* any channel subscription happens. it's the initial handshake, the 'are you who you say you are?' moment. if `find_verified_user` doesn't return a user (or whatever you use to identify a connection), the connection is rejected, and that's the end of it. that means no channels.

so, what's the solution? instead of skipping auth, we need to have a way to let actioncable think it has a valid user, or rather identify the connection based on other criteria for specific channels.

one effective method is using a flag or a special connection identifier for your public channels. for instance, in that project of mine, i added a parameter to the websocket connection url, let's say `public=true`. then, in the connection logic, i changed `find_verified_user` to something like this:

```ruby
  def find_verified_user
      if request.params[:public] == 'true'
        OpenStruct.new(id: 'public_connection')
      elsif verified_user = env['warden'].user
        verified_user
      else
        reject_unauthorized_connection
      end
  end
```

instead of doing a full authentication, for connections coming in with `?public=true`, the `find_verified_user` method creates a dummy object with an `id` property, like `OpenStruct.new(id: 'public_connection')` this makes actioncable happy. it now thinks there's an identified user, and it lets the connection continue.

then the key is in the channel, you handle it differently. for example, in the public channel `app/channels/public_channel.rb`

```ruby
class PublicChannel < ApplicationCable::Channel
  def subscribed
      stream_from "public_stream"
      puts "public user subscribed - #{current_user.id}"
    end
    # ...other methods
end
```

in this setup, `current_user.id` will log 'public\_connection' and your app will broadcast the information on that stream.

but now the problem with this is you can't easily tell apart your connections, this might become a problem later. you would need to somehow set a parameter in the connection to know what kind of connection it is, public or a normal one.

you might be thinking, why not just check on the channel for `current_user.id` to be public connection and skip authorization at that level. the answer is because the connection has to be previously established before getting into the channel logic, and it's part of the design that actioncable validates that.

another more complex workaround if you don't want to use query parameters is to use a pre-shared secret that your application sets, and checks in the connection. let's say, for example, you set `ENV['PUBLIC_WS_SECRET']` in your server, then you can use it to validate connections coming from your front-end like this:

in `app/channels/application_cable/connection.rb`

```ruby
  def find_verified_user
      if request.headers['x-public-secret'] == ENV['PUBLIC_WS_SECRET']
        OpenStruct.new(id: 'public_connection')
      elsif verified_user = env['warden'].user
        verified_user
      else
        reject_unauthorized_connection
      end
  end
```
and then in your javascript code, you would need to send it along the headers when connecting to actioncable.

```javascript
  const cable = createConsumer('ws://localhost:3000/cable', {
        'headers': {
          'x-public-secret': 'your_secret'
        }
      });
```
i've used these methods in different applications and they have their own particular set of advantages. the first one using query parameters is simpler and easier to implement, whereas the second one is more secure as it does not expose the secret in the query parameters.

one thing to consider if you are thinking about skipping auth for your application is that you might want to consider very well what kind of information you are sending through your websockets, as this can open potential attack surfaces if not handled correctly. another thing to keep in mind is that even though you might think that not authenticating is simpler, this can cause more complexity on the way and probably be harder to maintain.

also, a very common mistake is to think that the same connection is going to be re used across different tabs. each tab or window will have its own websocket connection to your actioncable server, keep that in mind when developing your application.

for more detailed explanations on this and the internals of actioncable, i recommend checking out the source code on github. the rails guides are good too, but sometimes they don't go deep enough. i've also found a very nice book by david heinemeier hanson called "agile web development with rails 7" that explains a lot about websockets in detail and how to handle different use cases. it's more of a general rails book but has a very good chapter on actioncable. there is a good guide as well in the official rails website about actioncable that explains this in general but does not go in detail about these use cases.

now, before i wrap up, a little joke: why was the ruby developer always calm? because he knew how to "gem" it together!

that's pretty much what i've learned and implemented myself. there isn't a magic bullet to skip auth on specific channels directly but with a couple of tricks, we can make actioncable work the way we want it to. keep in mind that all the examples here are written for simple cases, and might require adaptation depending on your specific project requirements. good luck!
