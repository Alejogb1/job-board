---
title: "How can Turbo Frames be used to create an inline signup form?"
date: "2024-12-23"
id: "how-can-turbo-frames-be-used-to-create-an-inline-signup-form"
---

Alright, let's talk about inline signup forms with Turbo Frames. It’s a pattern I've implemented quite a few times, and like most things, it’s got nuances worth exploring. The core idea is to provide a smooth, uninterrupted user experience when a user decides to sign up—no full page reload, just a seamless transition from the existing context. I remember one project particularly, a social networking prototype, where getting this just *right* dramatically improved conversion. Users were much more likely to sign up when the process felt lightweight and directly integrated into their browsing flow.

The magic happens because Turbo Frames essentially create isolated islands of interactivity within a page. When a frame's content is updated, only the frame's part of the document is re-rendered. This means that we can have a button, say, on a blog post, that triggers the loading of a signup form within that button's immediate surroundings without affecting the rest of the page. It feels instantaneous, even though there's a server round-trip happening behind the scenes.

Let’s break down how to achieve this. First, the basic setup involves three key elements:

1.  **A Trigger Element:** This is the initial element, often a button or link, that the user interacts with. When clicked, it initiates the form loading process.
2.  **A Frame:** This is the container that will hold the signup form. It’s identified by a unique `id` and is crucial for Turbo to know which part of the page to update.
3.  **A Server Endpoint:** This endpoint serves the HTML for the signup form. Crucially, this endpoint must return only the content that belongs inside the frame - no surrounding layout.

The first code example shows the basic HTML structure on the page:

```html
<div id="article-container">
    <p>Some compelling content about our awesome service...</p>
    <button data-turbo-frame="signup-frame">Sign up to learn more</button>
    <turbo-frame id="signup-frame"></turbo-frame>
</div>
```

Here, the `data-turbo-frame` attribute on the button ties it to the `turbo-frame` element with the `id="signup-frame"`. When the button is clicked, Turbo intercepts the request and loads the response into the corresponding frame.

Now, let's look at a Ruby on Rails example, where I often find myself when building applications using Turbo. Assuming we have a `UsersController`, the relevant parts might look like this:

```ruby
# app/controllers/users_controller.rb
class UsersController < ApplicationController

  def new
    render partial: 'form', locals: { user: User.new }
  end

  def create
    @user = User.new(user_params)

    if @user.save
      # success logic (e.g., redirect or update the page)
    else
      render partial: 'form', locals: { user: @user }, status: :unprocessable_entity
    end
  end

  private

  def user_params
    params.require(:user).permit(:email, :password)
  end
end
```

The `new` action renders a partial named `_form.html.erb`. This partial *only* contains the form, without any layout. The `create` action attempts to save the user. If successful, we might redirect or update the page. If there are errors, we re-render the form partial with the user object containing validation errors, setting a 422 status to ensure Turbo knows to re-render the frame.  Here's how that partial might look:

```erb
<%# app/views/users/_form.html.erb %>
<%= form_with model: user, url: users_path, data: { turbo_frame: "signup-frame" } do |form| %>
  <% if user.errors.any? %>
    <div id="error_explanation">
      <h2><%= pluralize(user.errors.count, "error") %> prohibited this user from being saved:</h2>

      <ul>
        <% user.errors.each do |error| %>
          <li><%= error.full_message %></li>
        <% end %>
      </ul>
    </div>
  <% end %>

  <div>
    <%= form.label :email %>
    <%= form.text_field :email %>
  </div>

  <div>
    <%= form.label :password %>
    <%= form.password_field :password %>
  </div>

  <div>
    <%= form.submit "Sign Up" %>
  </div>
<% end %>
```

Notice that the form itself is also configured to be submitted within the "signup-frame," by means of `data: { turbo_frame: "signup-frame" }`. This ensures that on submission the form is handled as a Turbo stream update too. Crucially, both rendering paths (new and create with errors) will be handled smoothly within the frame. Without the frame specified in the form tag, the submission would result in a full page load even though the response is just the form.

Now let's explore a slightly more complex scenario: displaying a "loading" state. This is important because initial form loading, as well as form submissions, involves server roundtrips, and we need to indicate to the user that something is happening.

Here's how to handle that in javascript:

```javascript
document.addEventListener('turbo:before-frame-render', (event) => {
    const frame = event.detail.target;
    frame.innerHTML = '<div class="loading">Loading...</div>';
});

document.addEventListener('turbo:frame-render', (event) => {
  const frame = event.detail.target;
  // You can remove the loading class or do any other clean up here if needed
});

document.addEventListener('turbo:submit-start', (event) => {
  const frame = event.target.closest('turbo-frame');

  if (frame){
    frame.innerHTML = '<div class="loading">Submitting...</div>';
  }
});

```
Here we attach listeners to turbo events. `turbo:before-frame-render` fires immediately before the frame will be rendered. At that time we reset the inner html with a loading message. The `turbo:frame-render` fires immediately after the rendering is complete, and is an ideal location to perform any clean-up actions.  Finally, we listen for the submission event and change the content to a submission loading message. This provides immediate feedback to the user while the server is processing the request.

The key here, and this is something that you'll find repeated in various implementations, is to ensure that both form display (the initial load) and form submissions are handled correctly. The form must have the `turbo_frame` attribute and the controller actions must render only partials without full layout. Error handling should also be thought through; the form partial will often be rendered again with error messages and that needs to work flawlessly as a frame update.

I would strongly recommend studying the official Turbo documentation. Specifically, the sections on Frames and Streams are invaluable. As well, the book "Rails 7: Recipes" by Andrea Leopardi has great recipes for Turbo integration. The work of DHH and the basecamp team in the Hotwire/Turbo space is a great resource to further your learning, and a great place to go if you find yourself having difficulties.

Implementing inline signup forms using Turbo Frames significantly improves the user experience by making it feel both faster and more integrated. By carefully considering each element of the user flow and keeping the server responses limited to just the frame content, you can create a truly seamless experience. It takes a few tries to really internalize the workflow, but the end result is definitely worth the effort.
