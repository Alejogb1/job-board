---
title: "How can I replace an embed object array with Rails mongoid?"
date: "2024-12-14"
id: "how-can-i-replace-an-embed-object-array-with-rails-mongoid"
---

alright, so you're looking to swap out an embedded array for something more, let's say, mongoid-y, in your rails app? i've definitely been down that rabbit hole before. it's a fairly common transition, especially as projects grow and your data structures need more finesse. i get it, embedded arrays, while simple at first, can become a pain when you start needing to query or update them independently. let me give you the lowdown based on my personal experiences, no bs, no marketing fluff.

first off, let's talk about *why* you’d want to ditch the basic array. typically it starts with basic requirements that grow into full fledged needs, like:

*   **querying complexity:** you might be finding yourself writing convoluted map-reduce queries or similar mongo operations just to find a specific item within the embedded array, which is terribly inefficient.
*   **updates and data manipulation:** updating a single embedded document within an array can become an exercise in frustration. think about having to fetch the entire parent document, modify the embedded array and then persist the entire thing back. it's ugly, prone to errors and not very performant.
*   **data normalization:** often, the embedded array ends up duplicating data. for example if you are embedding user data in each record that uses it instead of a user collection you will soon find inconsistencies across your data.
*   **scalability challenges:** this kind of setup doesn't scale well when the arrays get very large. the whole document needs to be read and updated every time even if you are changing the smallest thing.

so, yeah, it makes sense to move away from the basic array. but what are our options?

the typical pattern in rails with mongoid is to shift towards using `embeds_many` or `has_many` associations. the choice between them really depends on whether the embedded documents should stand alone or have a strong parent-child relationship to your current model.

`embeds_many` creates nested documents with a parent/child relationship where the embedded model cannot exist without the parent. so, if you delete the parent document the children are gone too, no questions asked. a typical example is an address, you dont usually keep addresses if they have no owner (at least in this specific example). `has_many` creates independent documents that are only associated to the other document through foreign key. that means a user can exist without an address and an address can have many users.

let's start with `embeds_many`, if you are considering embedded resources this makes sense. imagine you have a `product` model which currently has an array of `features`. instead of an array, we'll use an embedded document. here's how you might model it.

```ruby
# models/product.rb
class Product
  include Mongoid::Document
  field :name, type: String
  embeds_many :features, class_name: 'Feature', inverse_of: :product
end

# models/feature.rb
class Feature
  include Mongoid::Document
  field :name, type: String
  field :description, type: String
  embedded_in :product, class_name: 'Product', inverse_of: :features
end
```

notice the `embedded_in` part? this defines the parent-child relationship. the `inverse_of` is very handy too to keep the associations in sync. let's say you had a `feature` and want to find the parent, you can now call `feature.product`. before you would be forced to traverse the database with queries to obtain it.

now, if you want a parent-child relationship but you think your embedded documents should be treated as first class citizens that should live beyond its parent then you would use `has_many` instead. let's take our example further, instead of `features` let's say you have `reviews` instead, like a review for a product, here's how the model would look.

```ruby
# models/product.rb
class Product
  include Mongoid::Document
  field :name, type: String
  has_many :reviews, dependent: :destroy # or dependent: :nullify
end

# models/review.rb
class Review
  include Mongoid::Document
  field :rating, type: Integer
  field :comment, type: String
  belongs_to :product, inverse_of: :reviews
end
```

a couple things changed here. first, we are using `has_many` in product, meaning the reviews are now stored in a different collection. second, we have a `belongs_to` association to the product. and third i added the `dependent: :destroy` clause. the clause handles what to do with reviews when a product is deleted. you can specify `:nullify` if you want to simply remove the `product_id` of the reviews instead of deleting them. and of course, the `inverse_of` clause is there to make querying easier.

now, how would the transition work? this is the trickiest part of the equation and depending on your data volume you might want to take care on how to do this. here's a very simple migration idea:

```ruby
# db/migrate/some_timestamp_migrate_to_associations.rb
class MigrateToAssociations < Mongoid::Migration
  def self.up
    Product.all.each do |product|
      next unless product.respond_to?(:old_features) && product.old_features.is_a?(Array) # Assuming old_features is an array
      product.old_features.each do |old_feature_data|
        product.features.create(name: old_feature_data['name'], description: old_feature_data['description'])
      end
    end
    # if you were using has many it is slightly more complex because it
    # requires creating new instances and setting the relation
    # but i think you get the gist of it, i am not going into that
  end

  def self.down
    # you probably want to remove all the created records
    # you can be smart and remove only the ones created by the migration
    # but that might make this harder to read, so i will just remove all of them
    Product.all.each do |product|
        product.features.destroy_all
    end
  end
end
```

this migration script assumes you have a field called `old_features` that was an array with the old structure. this will iterate through the existing `products` and migrate the `features` into the new `embeds_many` structure, and similarly could have been adapted for `has_many` relation. during the process of migration you need to handle this at scale. the above example is not optimized, since it fetches all products to memory. but there are different ways of tackling this with a more efficient way. i've done this in the past by batch processing in the background to avoid performance degradation on the site. this is where i usually recommend checking out the mongoid documentation, specifically on how to create custom mongoid migrations and how to perform bulk operations with mongo. you can also read "mongodb the definitive guide", it has a good section on schema design patterns.

of course, after this is done you'll need to update the codebase to use the associations instead of your old array. this is a straightforward process of changing the way the data is fetched and modified using mongoid conventions and features. for example, instead of `product.old_features.find{|feature| feature.id == some_id}` you would write something like `product.features.find(some_id)`.

now, let me tell you a story. once i had a situation where i forgot to add the `inverse_of` clause to my model. it was hell to figure out why my data was not being updated properly. it took me an entire afternoon debugging to realize i had made this very very very basic mistake (it is always the basic mistakes that get you, isn't it?).

anyhow, switching from embedded arrays to mongoid associations is a necessary step in the journey of building complex applications, it makes your life easier in the long run. hopefully, this should put you on the *right* track. it might not be a perfect recipe but hopefully it is enough for what you are looking for. definitely hit those mongoid docs though, they are quite comprehensive.
