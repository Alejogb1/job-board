---
title: "How does Rails 7 integrate with Turbo?"
date: "2024-12-23"
id: "how-does-rails-7-integrate-with-turbo"
---

Let's tackle this one. I've spent a fair amount of time migrating legacy Rails applications to incorporate modern front-end techniques, and the introduction of Turbo into Rails 7 was a significant shift. It's not just a sprinkle of client-side magic; it fundamentally alters how we think about page interactions. In essence, Turbo isn't a replacement for a fully-fledged SPA framework but rather a streamlined way to enhance the interactivity of standard server-rendered applications.

Rails 7 leverages Turbo through the `@hotwired/turbo` JavaScript library, primarily focusing on three core components: Turbo Drive, Turbo Frames, and Turbo Streams. Let’s break each down to see how they contribute to the overall developer experience.

*Turbo Drive* is probably the most impactful. Before, a user clicking a link would trigger a full page reload. Turbo Drive intercepts these link clicks and form submissions, issuing AJAX requests instead. The server responds with the updated HTML body, and Turbo then morphs the existing page with the received content. It’s crucial to note that this entire process is geared towards minimal overhead; it's only the `<body>` that's replaced, leaving `head` elements intact and preventing assets from re-downloading on every interaction. This translates to a substantially faster and smoother user experience. This alone is a significant leap from previous Rails iterations, and it greatly reduces the reliance on excessive javascript. It's not *magic*, of course, but it feels like it.

Here's a typical example of a Rails view where Turbo Drive is automatically enabled, given that your application is set up with the default `application.js` import map:

```ruby
# app/views/posts/index.html.erb
<h1>All Blog Posts</h1>
<% @posts.each do |post| %>
  <div>
    <h2><%= link_to post.title, post_path(post) %></h2>
    <p><%= post.content.truncate(100) %></p>
  </div>
<% end %>
<%= link_to 'New Post', new_post_path %>
```

Clicking any of the post titles or the "New Post" link would trigger a Turbo Drive request, fetching and morphing just the body content of those links, rather than reloading the full page, which, to be honest, feels like magic when coming from older Rails applications.

*Turbo Frames*, on the other hand, provide a more granular level of control for updating parts of a page, or independent sections. These are essentially HTML elements that are designated for asynchronous updates. You wrap an area of the view in a `<turbo-frame>` tag with a unique id and then, when a request is made to that frame, only the content inside the matching tag is updated. This becomes incredibly useful for building complex interfaces where only specific elements need updating, improving efficiency compared to full body updates. In a past project involving user dashboards, we heavily used Turbo Frames to manage different components like profile settings and recent activity feeds, allowing seamless updating of these components independently.

Here's a Rails view showcasing how Turbo Frames are typically implemented. Imagine you have a comments section for a blog post:

```ruby
# app/views/posts/show.html.erb
<h1><%= @post.title %></h1>
<p><%= @post.content %></p>

<turbo-frame id="comments">
  <h2>Comments</h2>
  <% @post.comments.each do |comment| %>
    <p><%= comment.body %></p>
  <% end %>

  <%= link_to 'Add Comment', new_post_comment_path(@post), data: {turbo_frame: "comments"} %>
</turbo-frame>
```

And now, a matching controller action that responds specifically for that frame:

```ruby
# app/controllers/comments_controller.rb
def create
  @post = Post.find(params[:post_id])
  @comment = @post.comments.create(comment_params)
  respond_to do |format|
    format.turbo_stream {
      render turbo_stream: turbo_stream.replace("comments", partial: "posts/comments", locals: {post: @post})
    }
    format.html { redirect_to @post } # fallback for non-turbo requests
  end
end
```

This demonstrates how a form submission, when targeting the `comments` turbo-frame, replaces the entire comment list, while leaving the surrounding content untouched. I should note that the  `data: { turbo_frame: "comments" }` in the link forces a turbo frame update, instead of a full Turbo Drive navigation, if you are using this within a standard HTML link.

Finally, *Turbo Streams* offer even greater flexibility for updating the page. Streams are server-side responses formatted in HTML that instruct the client how to manipulate the DOM. These are usually generated using Action Cable and are ideal for real-time updates. Instead of replacing an existing frame or element, Turbo Streams allow for various operations, such as appending, prepending, removing, or updating specific elements. I found this most powerful when implementing live notifications in a previous application. By broadcasting Turbo Streams over web sockets, you can provide live feedback to multiple connected clients.

Here's an example using Action Cable and Turbo Streams for adding a new post in real-time:

```ruby
# app/channels/posts_channel.rb
class PostsChannel < ApplicationCable::Channel
  def subscribed
    stream_from "posts"
  end
end
```

```ruby
# app/jobs/post_broadcast_job.rb
class PostBroadcastJob < ApplicationJob
  queue_as :default

  def perform(post)
     ActionCable.server.broadcast("posts", turbo_stream.prepend("posts", partial: "posts/post", locals: {post: post}))
  end
end
```

```ruby
# app/controllers/posts_controller.rb
def create
  @post = Post.new(post_params)
  if @post.save
    PostBroadcastJob.perform_later(@post)
    redirect_to posts_path
  else
      render :new, status: :unprocessable_entity
  end
end
```

```ruby
# app/views/posts/index.html.erb
<div id="posts">
  <% @posts.each do |post| %>
    <%= render partial: "posts/post", locals: {post: post} %>
  <% end %>
</div>
```

The job then broadcasts a turbo stream to the channel "posts", telling all listening clients to prepend the new post to the `div` with the ID "posts". This functionality gives developers the ability to create very interactive experiences without resorting to fully fledged front-end frameworks.

It's also important to know how these technologies interact with each other. Turbo Drive can update the entire page body, Turbo Frames update specific sections within a page, and Turbo Streams can trigger updates based on server-side events, including, but not limited to, changes in real-time. In practice, a typical application might use Turbo Drive for navigation between pages, Turbo Frames for isolated sections like forms and lists, and Turbo Streams for live updates or real time feedback.

For further study, I highly recommend diving into the official Hotwire documentation – particularly the sections detailing Turbo. Also, "Agile Web Development with Rails 7," while it might seem obvious, dedicates significant sections to Turbo. For a deeper understanding of the underlying concepts, researching how partial updates work and how websockets function would be beneficial. Understanding the rationale behind these technologies allows you to leverage their full potential.

In conclusion, Rails 7 integrates Turbo to provide a robust, practical way to build interactive web applications while retaining much of the traditional server-side rendering approach. It's about enhancing, not replacing, the traditional Rails development workflow. These features, while powerful, also require a thoughtful approach in design, as they influence how you consider state management, and overall user interaction flow. From my experience, mastering these components allows you to develop very dynamic and performant user experiences without needing to completely rethink your server-side architecture.
