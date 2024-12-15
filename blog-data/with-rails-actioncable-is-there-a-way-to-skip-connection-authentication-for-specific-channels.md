---
title: "With Rails ActionCable: Is there a way to skip connection authentication for specific channels?"
date: "2024-12-15"
id: "with-rails-actioncable-is-there-a-way-to-skip-connection-authentication-for-specific-channels"
---

alright, so you're looking to bypass actioncable connection authentication for certain channels, got it. i've been there, staring at that connection handshake wondering if there's a secret back door. i can tell you, it's not as straightforward as flipping a switch but it's doable. let me tell you about the time i spent battling this in a project and how i ended up working around it.

i was building this real-time dashboard thingy back in my startup days. we had a public-facing status page and then internal dashboards for our support teams. the status page, obviously, shouldn't require any auth. it was meant to be public and display the current system status, while the internal dashboards needed full-blown user authentication. we decided to use actioncable for it cause it seemed logical at the time, real-time updates and all that jazz, but then i hit this wall of connection auth. it was like trying to get into a club with the wrong id – actioncable's default behavior insisted on verifying every connection.

the problem was, actioncable's connection establishment process, by default, routes everything through `app/channels/application_cable/connection.rb`. this file is where your authentication logic resides. typically, you have something like `identified_by :current_user` and then in a `connect` method you set that current user using sessions, cookies, jwt's, or any way you authenticate users. this approach, whilst perfect for private channels, isn't exactly ideal when you want a public channel where anyone can connect without prior auth. forcing non-auth channels into this pipeline seemed like overkill to me.

at first, i naively tried setting a public variable in the `connect` method to bypass the user auth if the channel name matched the public one. something like:

```ruby
# app/channels/application_cable/connection.rb
module ApplicationCable
  class Connection < ActionCable::Connection::Base
    identified_by :current_user

    def connect
      if params[:channel] == 'public_status'
        self.current_user = :public_user
      else
        self.current_user = find_verified_user
      end
    end

    private

    def find_verified_user
      # authentication logic
      # ...
      # returns user or reject_unauthorized_connection
    end
  end
end
```

this kind of worked… but it was messy and i felt dirty doing it. i was basically forcing a fake user into the connection which then opened a whole can of worms when handling actual users and the public stream. it’s like using a paperclip instead of a proper screw: does it hold? for a while yes, but not really the way we should do it.

what i found was, that you don't actually have to have `identified_by`. you can simply omit it. and then, you can create a separate connection class for your public channels. this way, your public connection won't try to authenticate users. it simply allows connections from anyone.

so, here’s how i'd approach this situation, creating a separate connection file specifically for the public channels:

1. create a new connection file specifically for your public channels: `app/channels/public_cable/connection.rb`.

```ruby
# app/channels/public_cable/connection.rb
module PublicCable
  class Connection < ActionCable::Connection::Base
    def connect
        # optionally, you can do things here before accepting the connection.
      logger.info "Public connection established"
    end
  end
end
```

notice there's no `identified_by` or `find_verified_user`, this connection just accepts anyone.

2. now the key part, configure your channel to use your new connection class. in your channel file for public broadcasting:

```ruby
# app/channels/public_status_channel.rb
class PublicStatusChannel < ApplicationCable::Channel
  def subscribed
    # use the public cable connection
    self.class.connection_class = PublicCable::Connection
    stream_from "public_status"
  end

  def unsubscribed
    # Any cleanup needed when channel is unsubscribed
  end

  # Other channel logic
end
```

here, within the `subscribed` method of your `public_status_channel`, you reassign the `connection_class` to your newly created `PublicCable::Connection`. this ensures that only the public channel bypasses authentication using the newly set connection class. that’s the ticket. any other channel will still use the `application_cable/connection.rb` with the authentication logic you had previously.

3. finally, make sure to update the javascript client, if needed, to include the correct channel name. this one should be simple, just make sure you pass 'public_status' as the channel name when creating a new consumer. if you're using the default actioncable setup, that would look something like this.

```javascript
// js/channels/index.js
import consumer from "../channels/consumer"

consumer.subscriptions.create("PublicStatusChannel", {
  connected() {
    console.log('connected to public status channel')
    // Called when the subscription is ready for use on the server.
  },
  disconnected() {
    // Called when the subscription has been terminated by the server.
    console.log('disconnected from public status channel')
  },
  received(data) {
      console.log(data);
    // Called when there's incoming data on the websocket for this channel.
  }
});
```

this approach is cleaner and keeps your authentication logic separate from your public channels. this separation principle is something i learned the hard way. and it does follow a more standard approach of separation of concerns, making code more readable and easy to maintain in the long run.

regarding resources, instead of links, i always recommend looking into good books. in this case, the official rails documentation, in particular section about actioncable is a must have to deepen on the mechanics of it. also, there is a great book called "programming ruby" if you need to understand the mechanics of ruby programming in depth. for specific actioncable details, any book on "real time web application development with rails" should give you a great head start. lastly, reading blog posts can help, but i would recommend you verify the solutions and code on those as they might not have the required depth you need.

if i may add a bit of an anecdote, once i tried to optimize a websocket connection by using custom headers, and i spent hours until i realized i had the wrong nginx configuration, yeah, tech can be really funny sometimes but at the same time it's humbling.

hope this helps you move forward on your problem. let me know if you have any other questions, i'm always up for these kind of puzzles.
