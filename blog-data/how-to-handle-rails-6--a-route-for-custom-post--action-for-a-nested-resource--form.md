---
title: "How to handle Rails 6 & a route for custom post- action for a nested resource & form?"
date: "2024-12-15"
id: "how-to-handle-rails-6--a-route-for-custom-post--action-for-a-nested-resource--form"
---

alright, so, nested resources and custom post actions in rails 6 with a form, gotcha. been there, done that, got the t-shirt. or maybe more accurately, got the caffeine-fueled all-nighters and the "why is this not working?!" moments. let me break it down based on some of my own, shall we say, *adventures* with similar setups.

first, let's assume you've already got your basic models set up with the nested association. something like, say, a `blog` has many `posts`, and you're trying to add a custom action, maybe 'publish', to a specific post within that blog. i’ll throw together some snippets that will help you get the picture.

we'll start with the basic routes.rb config:

```ruby
# config/routes.rb
Rails.application.routes.draw do
  resources :blogs do
    resources :posts do
      member do
        post 'publish'
      end
    end
  end
end
```

this sets up the standard restful routes for `blogs` and `posts`, and then adds a `post` route to `/blogs/:blog_id/posts/:id/publish`. the `member` part there is crucial – it means we’re dealing with a specific `post` identified by its `id`. i’ve seen users get tripped up by using ‘collection’ there and scratching their heads because the url wouldn’t have the `id` part of it, which is needed, obviously. in that case the route created would have been something like `/blogs/:blog_id/posts/publish`. so, yeah, double check that.

next, the controller action. here’s how i’d handle it in `posts_controller.rb`:

```ruby
# app/controllers/posts_controller.rb
class PostsController < ApplicationController
  before_action :set_blog
  before_action :set_post, only: [:show, :edit, :update, :destroy, :publish]

  def publish
    if @post.update(published: true)
      redirect_to blog_post_path(@blog, @post), notice: 'post published!'
    else
      render :edit, status: :unprocessable_entity # or whatever you want to do if it fails
    end
  end

  private

  def set_blog
    @blog = Blog.find(params[:blog_id])
  end

  def set_post
    @post = @blog.posts.find(params[:id])
  end
end
```

i’ve added the `@blog` variable there so that i can keep the nested resource in context, very helpful when dealing with complex scenarios. this controller action, `publish`, locates the relevant post via the `set_post` method (which also makes sure it belongs to the current blog, again, another common issue i've seen people stumble into) and then attempts to update the `published` attribute. if the update succeeds, it redirects back to the post’s show page with a success message. if not, it renders the edit form again. and, just so we are all on the same page, you need to add that `published` column on the posts table in your migration. i saw somebody spending a whole day debugging this, and well, just wanted to make sure you are not that person.

now, the form part. since you mentioned it, i’m going to assume you want a form on the post's show page to trigger this action. here's how i'd tackle it using a form with a button:

```erb
# app/views/posts/show.html.erb
<h1><%= @post.title %></h1>
<p><%= @post.content %></p>

<% if !@post.published %>
    <%= form_with url: publish_blog_post_path(@blog, @post), method: :post do |form| %>
      <%= form.submit "publish this post", class: "button" %>
    <% end %>
<% else %>
  <p>published!</p>
<% end %>
```
this erb code shows the details of the post. the form's `url` helper, `publish_blog_post_path`, constructs the correct url, which as we saw before, it becomes something like `/blogs/:blog_id/posts/:id/publish`, so it hits the route we defined earlier. the `method: :post` makes sure that the form uses the post method and triggers the action. you’d obviously modify this erb code to fit your specific views, but the core functionality stays the same.

a key point here is the use of the form's url helper `publish_blog_post_path`. this automatically constructs the correct nested url based on your routes definition. avoiding hardcoding urls is very important. you'll thank yourself later. trust me, i've been there. debugging hardcoded urls across a big project? not fun at all.

now, you asked about resources instead of links. there are a couple of good books i can recommend that have been quite useful to me over the years. for a comprehensive grasp of rails routing, i would suggest the ‘agile web development with rails’ book, it covers everything from basic routing to advanced techniques, and it’s a great foundation for understanding how rails handles nested resources and form submissions. for a deeper dive on the principles behind restful apis, which is essential in building scalable web apps, i'd check out 'restful web services' book. it is a classic resource that helped me understand the design principles behind restful architectures.

now, to some common pitfalls:

*   **forgetting the before\_action:** make sure your `set_blog` and `set_post` are correctly grabbing the blog and post instances based on the ids passed in through params. forgetting to do so can lead to lots of errors and make your life much harder (i know, i’ve been there).
*   **typos in the route helpers:** be careful about typos in the `publish_blog_post_path` helper in your forms. a simple mistake like `publish_blog_posts_path` (plural) can throw everything off. i've lost hours hunting down those types of bugs, so be meticulous about it.
*   **csrf tokens:** rails includes csrf protection by default, so make sure that your form includes the authenticity token. the `form_with` helper automatically handles that for you so you should not worry about that. just remember that, and never try to manually construct the form yourself to avoid problems.
*   **validation issues:** if the update on your `publish` action fails, make sure your model validations and your controller handle that case appropriately. i usually add error handling using flash messages.

finally, i remember one particular project where i was trying to do something similar and was driving me crazy. i had a nested resource, and a custom action and the whole thing was not working as i expected. turns out it was a simple typo in the route config. a missing 's' in one of the resource names. spent a whole afternoon on that. the moral of the story? always double check the basics. it’s kind of like that joke about the programmer who’s wife tells him to go to the store, she says: "get a liter of milk, and if they have eggs get six”. he comes back with six liters of milk.

anyway, that's how i'd generally approach handling nested resources and custom post actions with a form in rails 6. it’s a common pattern, but it can get tricky, especially with nested routes and the intricacies of form submissions. the key is to understand the basics, to read the manuals and practice and be meticulous in everything you do.
