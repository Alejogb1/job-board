---
title: "How to handle STI and use base model for forms and view?"
date: "2024-12-15"
id: "how-to-handle-sti-and-use-base-model-for-forms-and-view"
---

alright, so handling single table inheritance (sti) with forms and views, right? i've been down that road more times than i care to count, and it can get a bit hairy if you're not careful. basically, you're cramming multiple model types into one database table, and then trying to keep it all straight in your app. it's not inherently bad, but it does require a specific way of thinking. i'll lay out how i've approached it in the past, specifically focusing on getting those forms and views playing nice.

the core of the issue is that you have one table, let's say 'products', but you actually want to treat those rows as different things depending on their 'type' column. it could be ‘book’, ‘movie’, ‘software’, whatever. that type column is what defines the sti. ruby on rails handles this pretty well out of the box. you'll define a base model, for example `product`, then specific sub-classes, like `book`, `movie`, each inheriting from `product`.

first of all, let's talk about the models. it's pretty standard stuff. i remember the first time i tried this, i got completely tripped up on the inheritance and wasn't properly setting the `type` column on creation. that resulted in everything being a plain `product` instead of a `book` or a `movie`. learned that the hard way, after debugging for hours.

here’s a basic setup for the models:

```ruby
# app/models/product.rb
class Product < ApplicationRecord
  validates :name, presence: true
  validates :type, presence: true # crucial for sti
end

# app/models/book.rb
class Book < Product
  validates :author, presence: true
end

# app/models/movie.rb
class Movie < Product
  validates :director, presence: true
end
```

notice that the `type` column in the `products` table is what makes this inheritance work. rails automatically knows which class to use based on that column. that's some magic there, but it’s good magic. this basic setup is a good starting point. you define the common attributes on the `product` model and the specific attributes on the subclasses.

now, for forms. the trick here is to create a single form that can handle all of the different types. you’re not going to be making separate forms for `book`, `movie`, and so on. that defeats the purpose of sti. instead, we'll make a single form that can conditionally display the fields based on the `type` of product being edited or created.

here's how i’d typically do it, assuming you are using rails form builders:

```erb
<%# app/views/products/_form.html.erb %>
<%= form_with(model: product) do |form| %>
  <%= form.label :name %>
  <%= form.text_field :name %>

  <%= form.hidden_field :type, value: product.class.name %>

  <% if product.is_a?(Book) %>
    <%= form.label :author %>
    <%= form.text_field :author %>
  <% elsif product.is_a?(Movie) %>
    <%= form.label :director %>
    <%= form.text_field :director %>
  <% end %>

  <%= form.submit %>
<% end %>
```

what's happening here? first, you use `form_with(model: product)` to build the form, regardless of whether you are working with a `book` or a `movie`. we are using `product` as the model, that's key. the next important line is the hidden field: `<%= form.hidden_field :type, value: product.class.name %>`. this is what actually tells the database which kind of `product` you're creating or editing. you have to pass the class name, not some random string, or else things will break. lastly, we have a conditional block that checks the class of the `product` instance. if it’s a `book`, show the `author` field. if it’s a `movie`, show the `director` field. the form will dynamically display fields based on what the `type` column is. it’s very flexible. when creating a new product you'll need to add a select or some other method to set the type.

for rendering, i've always found partials to be incredibly helpful. instead of trying to cram all the logic into one view, create a partial for each specific type. this keeps the views cleaner and more manageable. it also make debugging much easier when things get complicated with more than 3 or 4 types, or even more complex fields in each type.

here’s how i might structure the view partials:

```erb
<%# app/views/products/_product.html.erb %>
<div>
  <h2><%= product.name %></h2>
  <p>Type: <%= product.class.name %></p>
  <%= render "products/details/#{product.class.name.downcase}", product: product %>
</div>
```

```erb
<%# app/views/products/details/_book.html.erb %>
<p>Author: <%= product.author %></p>
```

```erb
<%# app/views/products/details/_movie.html.erb %>
<p>Director: <%= product.director %></p>
```

in the main `_product` partial, we render a specific partial based on the class of the `product`. so if `product` is a `book`, it renders `app/views/products/details/_book.html.erb` which will only display the book specific fields. and, if its a movie renders `app/views/products/details/_movie.html.erb`. This helps avoid those giant, confusing view files i've seen people use (and used myself early on), and it keeps your code dry.

now for the controller. i typically use a `params[:type]` to set the class of the object when creating the new `product`, you can create a method in a module to handle all of this, but here is a simple example in the controller:

```ruby
# app/controllers/products_controller.rb
class ProductsController < ApplicationController
  def new
    @product = Product.new
  end

  def create
    klass = params[:type].constantize
    @product = klass.new(product_params)
    if @product.save
        redirect_to @product
    else
      render :new
    end
  end

  def show
    @product = Product.find(params[:id])
  end

  private
    def product_params
        params.require(:product).permit(:name, :type, :author, :director)
    end
end
```
notice that i'm using `params[:type].constantize` which turns that string of class name into an actual constant. i’ve had that not work before when dealing with namespaced models. just keep an eye out for that. also note that all the fields are allowed in `product_params` that's because the same form is used for all product types. it might seem like a security concern at first but it is not as long as you use the correct `type` and the correct classes are used in the form. if someone sends in a `director` when creating a book, that field will simply be ignored because the `Book` model does not have a `director` attribute.

a word of caution, when working with sti you should be very careful to avoid naming conflicts, if not you will enter a debugging rabbit hole that will make you cry. it's something i have personally suffered and would not wish to anyone.

for resources to deepen your understanding, i'd recommend looking into “patterns of enterprise application architecture” by martin fowler, especially the section on inheritance strategies. while it might not talk directly about sti in rails, it will give you a better understanding of the underlying patterns, how and when to use them and when to just avoid them. i've found that a theoretical foundation like that can really make you a better developer. also, the active record documentation on single table inheritance in the ruby on rails guides is an essential read.

remember, with sti you’re basically making a trade-off between database structure and application flexibility. it's powerful but not always the perfect tool. keep it simple and you'll be able to create complex applications. oh, i almost forgot, someone told me the other day that sti is like a party where everyone brings their own dish, but they all eat from the same table... i guess that was a pretty lame joke, i will leave the comedy to the professionals. in any case, that's my approach to this. hope it helps!
