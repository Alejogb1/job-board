---
title: "How to pass a view value to a controller on rails?"
date: "2024-12-14"
id: "how-to-pass-a-view-value-to-a-controller-on-rails"
---

alright, so, passing a view value to a controller in rails, i’ve definitely been there, done that, got the t-shirt, and probably debugged it for longer than i'd like to remember. it's one of those things that seems straightforward but can trip you up if you're not thinking about the data flow. when i first started with rails, i got tangled in the idea of direct view-to-controller communication, like trying to make two wires connect without a proper socket. it doesn't work that way.

fundamentally, views are for displaying data, and controllers are for handling logic and requests. the communication path, in a standard rails app follows this, view->http request->controller->model->database (if needed). so, you need to use a form or a link to generate an http request. this request will carry the value, usually as a parameter, that you want to pass from your view to your controller.

let’s break it down into common scenarios i've bumped into:

**scenario 1: simple form submission**

the most common case is that you have some data input by the user, using a form. let’s say you have a simple form with an input field for, i don’t know, a user's favourite programming language. in your view (`app/views/users/new.html.erb` for example) you might have something like:

```erb
<%= form_tag users_path, method: :post do %>
  <label for="favorite_language">favorite language:</label><br>
  <input type="text" id="favorite_language" name="favorite_language"><br>
  <%= submit_tag "submit" %>
<% end %>
```
here, `form_tag users_path, method: :post` sets up a form that, when submitted, will send a post request to the `users#create` action, or whatever is defined by the `/users` route. the key part is `<input type="text" id="favorite_language" name="favorite_language">`. the `name` attribute, which is 'favorite_language' is what matters here. rails will use this name as the key in the `params` hash in your controller.

then, in your controller (`app/controllers/users_controller.rb`), you can access this value like so:

```ruby
class UsersController < ApplicationController
  def create
    favorite_language = params[:favorite_language]
    # do something with the value, like saving it to a db or something
    puts "user's fave language is: #{favorite_language}"
    redirect_to root_path, notice: "nice one!"
  end
end
```

notice how `params[:favorite_language]` retrieves the submitted value.

i remember once i had a form that kept returning empty parameters and it took me half an afternoon to realise that i had a typo in the `name` attribute in the view. classic case of eyes skipping over the details! debugging tip: always `puts params.inspect` to verify what data your controller is receiving. i learned that the hard way.

**scenario 2: passing values with links**

sometimes, you don't need a full form submission, just a simple link to a page and send an id or value along with it. you can do this by appending the value to the url as a parameter.

in your view (say in `app/views/products/index.html.erb`) you could create a link like this:

```erb
  <% @products.each do |product| %>
    <p>
      <%= product.name %>
      <%= link_to "show details", product_path(product, format: :html, category_id: product.category_id) %>
    </p>
  <% end %>
```
here, the `product_path(product, format: :html, category_id: product.category_id)` helper generates a url like `/products/123?category_id=456` (assuming `product.id` is 123 and category id is 456) also including the `format: :html`. rails will automatically append the id to the path. the `category_id: product.category_id` part shows how you can add arbitrary values as parameters in the link and `format: :html` is another, and common one, to control the type of response.
then, in the `products#show` action of your controller (`app/controllers/products_controller.rb`), you can do the following to access this value:

```ruby
class ProductsController < ApplicationController
  def show
    @product = Product.find(params[:id])
    category_id = params[:category_id]
    puts "product's id is #{params[:id]}, category id is #{category_id}"
    #render the view to show product info.
  end
end
```
just as in the form case `params[:category_id]` gives you the passed value.

i once struggled because i was trying to access the id of a record in the view, directly, before the actual http request, before the controller's actions happened. i was trying to `params[:id]` in the view, which is incorrect, of course. views render after the controller actions have run so, parameters are only available on controllers. my colleague, who is really into databases, found it funny that i tried to query the params database before the controller.

**scenario 3: hidden fields within a form**

sometimes, you need to pass a value that the user doesn't directly input, like an id of a parent record. you can do this using hidden fields inside a form. you can add a hidden field in your view, (inside any `form_tag` block, in `app/views/comments/_form.html.erb` say):
```erb
<%= form_with(model: @comment, url: comments_path) do |form| %>
    <%= form.hidden_field :post_id, value: @post.id %>
    <%= form.text_area :body, placeholder: "add a comment" %>
    <%= form.submit %>
<% end %>

```
here, `<%= form.hidden_field :post_id, value: @post.id %>` adds a hidden input with the name 'comment[post_id]' and its value would be the id of the `@post` object. note the syntax `comment[post_id]` for nested attributes which rails will treat nicely when saving data.

in your controller (`app/controllers/comments_controller.rb`), you can access it like so:

```ruby
class CommentsController < ApplicationController
  def create
    @comment = Comment.new(comment_params)
      if @comment.save
        redirect_to @comment.post, notice: "comment added."
      else
        render :new, status: :unprocessable_entity
      end
  end

  private

  def comment_params
    params.require(:comment).permit(:body, :post_id)
  end
end
```
here `params[:comment][:post_id]` would give you the hidden value. note the use of strong parameters in rails. is considered good practice.

i've used hidden fields quite extensively in my projects. i also recall that i forgot to include them in the strong parameters method and that took me a while to figure out. another lesson learned the hard way, haha.

**resources and further reading:**

for a deeper dive, i recommend checking out the official rails documentation on routing, forms, and controllers. you can also learn by doing and reading through guides in the 'ruby on rails tutorial' by michael hartl, it's a good one, it covers this stuff really well. also, anything by avdi grim and sandi metz will be useful and they have a more advanced take on the matter, and will help on designing a good overall software. the 'rails api documentation' is always your best bet, so be familiar with it as much as you can.

remember the key takeaway: data travels from the view to the controller via http requests, and parameters. understanding this flow will save you a lot of debugging time. now, go forth and pass those view values with confidence!
