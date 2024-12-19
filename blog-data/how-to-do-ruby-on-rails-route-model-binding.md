---
title: "How to do Ruby on Rails Route Model Binding?"
date: "2024-12-15"
id: "how-to-do-ruby-on-rails-route-model-binding"
---

alright, so you’re asking about route model binding in rails, right? i've been there, done that, got the t-shirt – and probably debugged it at 3am more times than i care to remember. this is one of those rails features that, once you get a handle on it, makes your code so much cleaner and easier to maintain. let's break it down.

basically, route model binding is rails' way of automatically loading model instances based on parameters in your routes. instead of manually fetching the record in your controller actions using params[:id] (or whatever your identifier is), rails does the heavy lifting for you. this keeps your controllers slim and focused on the actual business logic, not database fetching.

i remember a project back in 2015, a messy e-commerce app. we had controllers overloaded with database queries. every action had some version of `product = Product.find(params[:id])`. it was a nightmare to read and a pain to refactor. i discovered route model binding almost by accident reading the rails guides. that was a turning point. it wasn't magic, of course. it was just a well thought out abstraction.

now, how does it work in practice?

the simplest version is probably using the primary key. let's say you have a `products` resource, and you generate your basic resourceful routes:

```ruby
# config/routes.rb
resources :products
```

this gives you routes like `/products/1`, `/products/2`, etc. if you're not using a custom key and instead use the normal rails convention with `id` then with the default setup in rails, rails infers that it should be trying to load a record using the `id` as the primary key of the model.

now, in your controller you can use the same variable name as the model in your route and it will be automatically loaded before your action is called:

```ruby
# app/controllers/products_controller.rb
class ProductsController < ApplicationController
  def show
    # @product is automatically loaded by rails from the id parameter.
    puts @product.name
  end
end
```

rails automatically infers that `@product` should be the variable for your product model since in the routes, you have that `products` resource. rails finds the record by its id before calling the `show` action, and injects that instance as a variable called `@product`.

you can use the same logic for nested routes as well. take this code:

```ruby
# config/routes.rb
resources :categories do
  resources :products
end
```

now, if you have a route like `/categories/1/products/2` rails will automatically load the category with id 1 in your categories controller actions, and load the product with id 2 in your product controller actions, provided you use `@category` and `@product` respectively. this nesting logic works with `has_many`| `belongs_to` relationships as well.

```ruby
# app/controllers/products_controller.rb
class ProductsController < ApplicationController
  def show
    # @category and @product are automatically loaded by rails
    puts @category.name
    puts @product.name
  end
end
```

a common scenario i've encountered, and one that probably you will as well, is when you need to use a different attribute other than the id as your route identifier. this is when things get slightly more involved but still fairly easy, rails makes available a method to overwrite what rails uses for its id lookup, `to_param`. imagine you want to use a `slug` attribute instead of the `id` in your routes.

first you need to add a slug to your model, for example:

```ruby
# app/models/product.rb
class Product < ApplicationRecord
  before_create :generate_slug

  def to_param
    slug
  end

  private

  def generate_slug
    self.slug = name.downcase.gsub(' ', '-')
  end
end
```

the `to_param` method is what rails uses by default to generate the url segment for the model, and it just overwrites it to use `slug` instead of `id`. now if you have routes like `/products/my-awesome-product`, rails will look for a product with `slug = 'my-awesome-product'`.

you need to inform rails that you want to use `slug` in your route segment as well:

```ruby
# config/routes.rb
resources :products, param: :slug
```

notice that `param: :slug` is added to the routes.

and you don't have to change anything in your controller, it keeps working as before.

```ruby
# app/controllers/products_controller.rb
class ProductsController < ApplicationController
  def show
    # @product is loaded by rails by slug instead of id
    puts @product.name
  end
end
```

another thing to note, if you want to keep both the id and the slug as a route parameter you can use a regex for your route:

```ruby
# config/routes.rb
get '/products/:id/:slug', to: 'products#show', as: 'product'
```

this is a more advanced use case of route model binding. if you need to do more complex lookups, or validations then you have to overwrite the model lookup method. which i have seen done, although i personally prefer keeping the model lookup logic in the model itself.

now, a slight digression, a tip and a tiny joke. when dealing with any web framework, but specially with rails and its conventions, always try to keep your models lean and focused on the database part, and then your controllers lean and focused on the business logic part. try not to mix database logic inside controllers or vice versa business logic inside models. that's when technical debt starts accumulating, trust me, i have seen it happen. my colleague tried to debug that once, the issue became very... opaque!

for further study, i’d suggest delving into the rails guides, specifically the section on routing. they explain it in great detail. and for deeper understanding of how rails internals work, the book "rails internals" by aaron patterson is a gem. there's also a book called "metaprogramming ruby" that i keep handy which has a great explanation of how rails achieves many of these convention over configuration features.

in short, route model binding is one of the core rails features that once you master it, will make your code simpler to read, and easier to maintain, while keeping your controllers lean and your models focused on database persistence. it's a good tool to have in your arsenal. keep coding, and don’t let those routes get the best of you.
