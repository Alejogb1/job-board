---
title: "How to pass a resource from a controller to view in rails/devise?"
date: "2024-12-15"
id: "how-to-pass-a-resource-from-a-controller-to-view-in-railsdevise"
---

so, you're trying to get some data from your rails controller into your view, and devise is involved, right? i've been there, trust me. this seems to pop up quite a bit, and it's usually a pretty simple fix once you see how it all connects. let's break it down.

the main thing to understand is that controllers are the middleman. they handle the request, grab whatever data needed, and then pass that data off to the view, which is responsible for rendering it to the user. devise, in this case, adds a layer of authentication logic, but doesn't fundamentally alter the core flow of data from controller to view.

in your situation with devise, it sounds like you need to access user-specific data (or something related) in your view *after* a devise action has happened, like a user logging in.

first thing: you absolutely should avoid directly accessing devise resources inside the view template. this creates tight coupling and makes your app harder to maintain. the controller needs to explicitly fetch and pass the data to the view.

let me give you an example. let's say you have a user model with a 'nickname' attribute and you want to display that nickname after they've logged in.

a common mistake would be to try to directly pull the current user in the view:

```ruby
# this is bad
# view/layouts/application.html.erb (for example)

<p>hello, <%= current_user.nickname %></p>

```

while this will work (most of the time), it's not good practice and it is error prone. you're directly depending on devise being available there, and if current\_user was not set, your view will simply fail. this leads to all sorts of subtle bugs. controllers exist for a reason.

instead, what you should do is, in your controller after a successful login (or in the controller rendering the view where this needs to show) you explicitly fetch the user, grab the nickname and pass it to the view.

here is an example controller action where this is done correctly:

```ruby
# app/controllers/home_controller.rb

class HomeController < ApplicationController
  before_action :authenticate_user! # devise magic ensures user is logged in (or redirects if they arent)

  def index
    @user_nickname = current_user.nickname # pass to view
  end
end

```

then in the view:

```erb
# app/views/home/index.html.erb

<p>hello, <%= @user_nickname %></p>
```

i’ve spent hours debugging something like this because i lazily decided to put that current\_user.nickname thing directly in the view, and later on things began to break, because it relied on too much implicit information. learned my lesson the hard way.

the `@user_nickname` in the controller action is an instance variable, which rails automatically makes available to the view. it’s ruby stuff. it is a pretty simple concept but understanding it is fundamental.

also note the `before_action :authenticate_user!` in the controller. this is devise's way of making sure the user is logged in before executing that action. if they are not logged in, it will redirect them to the sign-in page. so that gives you a clue on how devise handles authentication stuff and how it can work with the controller.

now, what if you need to pass more complex data? like, say you want to display a list of the user’s posts. the concept is still the same, but now you are passing an array of objects to your view:

```ruby
# app/controllers/home_controller.rb
class HomeController < ApplicationController
  before_action :authenticate_user!

  def index
    @user_posts = current_user.posts.order(created_at: :desc).limit(5) # fetch the posts here. you are free to apply any logic
  end
end
```

and in the view:

```erb
# app/views/home/index.html.erb

<p>your latest posts:</p>
<ul>
  <% @user_posts.each do |post| %>
    <li><%= post.title %> - <%= post.created_at.to_formatted_s(:short) %></li>
  <% end %>
</ul>

```

here, `@user_posts` is passed as an array of `post` objects. inside the view template, we iterate over the array, rendering each post.

this pattern is scalable. regardless of how complex your model interactions are, the key is that the controller fetches data and passes data explicitly to the view via instance variables (prefixed with `@`). this way you are not coupling the view with the model interactions but keeping the controller the single source of truth.

a good way to think about it is: the view should be just about showing things and nothing more than that. no logic there, just presentation.

for deeper reading, i would definitely look at these references. specifically, the “rails way” book by obie fernandez. you should probably look at the section explaining mvc and how data is passed back and forth, that might prove invaluable here. also, i find that the agile web development with rails book goes into very good details about views and erb syntax. another recommended reading would be the official ruby on rails guides on controllers, it explains in detail how controllers communicate with views. there also are some very in-depth explanations of the view rendering pipeline that may be helpful for understanding the full picture. also you should be comfortable with basic ruby syntax and how object orientation in ruby works, that will help understand how rails variables pass along the chain.

one last thing: you mentioned devise specifically. remember that devise has its own set of helper methods that allow you access to the current logged-in user. that's how i was able to get away with `current_user` object in that example. these helper methods are available in controllers and views, but only if you have the appropriate `before_action :authenticate_user!`. and just a friendly reminder, never use them directly in views, always pass information via the controller. why? because you are just asking for a debugging session later on (and believe me i had a few). speaking of debugging sessions, why do programmers prefer dark mode? because light attracts bugs haha.

hope this all helps, let me know if you have any more questions.
