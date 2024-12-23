---
title: "Why isn't Rails `.build` creating `has_many :options` associations?"
date: "2024-12-23"
id: "why-isnt-rails-build-creating-hasmany-options-associations"
---

Let's dive into this. I remember troubleshooting a similar issue a while back, implementing a complex product configuration system. We were using `has_many :options` in our Rails application, and the `build` method just wasn't behaving as we expected. Specifically, we found that calling `build` on an associated model wasn't creating the actual database records, nor was it populating them in the parent object's association collection. It became clear that a deeper understanding of how Rails handles associations and the nuances of `build` versus `create` was necessary.

The core of the problem usually revolves around understanding the difference between in-memory object manipulation versus persisting records to the database. Let's break this down systematically. The `build` method, within the context of Rails associations, is designed to construct an instance of the associated model in memory; it does not immediately save this instance to the database. It’s a non-persisting action. Essentially, it sets up the relationships for future record creation. The magic of actually persisting the associated records happens when you eventually call `save` on the parent object. If you aren't saving the parent, those built but unsaved child records are lost in the ether.

Now, when you’re working with `has_many` associations and you use `build`, Rails instantiates the child object and also sets the foreign key on the child to match the id of the parent, *if the parent object has been saved* (meaning, it possesses a record id). This relationship is crucial because if the parent is a new record, the foreign key remains nil until you first save the parent. Subsequently, when you save the parent, it triggers the saving of its associated `build` children as well – a cascade if you will. The behavior might appear incorrect if the saving is not propagated accordingly. It might look like nothing is being 'built' but it is happening internally, just not persisted.

To illustrate further, let's examine some hypothetical models and code snippets:

**Example 1: Incorrect Usage (Not Saving the Parent)**

Let's assume we have a `Product` model and an `Option` model, where a `Product` has many `Options`:

```ruby
# models/product.rb
class Product < ApplicationRecord
  has_many :options, dependent: :destroy
end

# models/option.rb
class Option < ApplicationRecord
  belongs_to :product
end

# Example in console
product = Product.new(name: "Awesome Gadget")
product.options.build(name: "Color Option", value: "Red")
product.options.build(name: "Size Option", value: "Large")

puts product.options.count # Output: 2
puts Option.count        # Output: 0 (because we haven't saved)
```

In this scenario, although we’ve `build` two options, they exist only in the in-memory collection of `product.options`, and they do not have a corresponding database record. The count is correct according to the objects existing in memory, but are not yet persistent. The issue in this context is that the `product` record itself hasn't been saved, and without an `id` available on `product` at the point of calling `build`, Rails doesn’t yet know how to correctly set the foreign key for these new options. Subsequently, if you don't persist these records, you won't find the `Options` being persisted. This leads us to...

**Example 2: Correct Usage (Saving the Parent)**

Let’s modify the previous example by saving the `Product` and then we'll see the difference.

```ruby
# console
product = Product.new(name: "Awesome Gadget")
product.options.build(name: "Color Option", value: "Red")
product.options.build(name: "Size Option", value: "Large")
product.save # Save the parent product; this is key!

puts product.options.count # Output: 2
puts Option.count        # Output: 2 (Now records are created)

# To make sure we see that association:
loaded_product = Product.find(product.id)
puts loaded_product.options.count # Output: 2

```

Now, because we've saved the `product` *before* we tried to inspect the `options` association, Rails was able to set up the foreign keys correctly and persist the records for both the product and the associated options. The important part is the product needs to have its id in order for the association to be properly configured before it is saved. The loaded product retrieves the associations properly.

**Example 3: Using `create` instead of `build`**

While not directly addressing why `build` does not persist immediately, let's examine the effect of `create`, which does.

```ruby
# console
product = Product.new(name: "Awesome Gadget")
product.options.create(name: "Color Option", value: "Red")
product.options.create(name: "Size Option", value: "Large")

puts product.options.count # Output: 2
puts Option.count        # Output: 2
# Note the product was saved implicitly due to the call to 'create'
```
In this example, each call to `create` both instantiates and persists the associated `Option`. It’s a direct database operation. The main difference from `build` is that the `create` method saves both the parent and child object immediately to the database if the parent was not saved beforehand. Use this one when you want both the association *and* the persistent state immediately. `build` should only be used when you wish to work on unsaved data in memory before saving it in a batch or transaction.

To deepen your understanding of Rails associations, I’d strongly recommend reading "Agile Web Development with Rails 7" by Sam Ruby et al. This book offers a comprehensive explanation of Rails' ORM and the intricacies of handling associations, which has always proven to be a great resource. Additionally, the official Rails Guides on Active Record Associations are indispensable; refer specifically to the sections dealing with `has_many` and `belongs_to` relationships.

The key takeaway here is that `build` is a non-persisting action. It creates in memory instances of objects that are related via foreign key constraints. Those objects are only persisted if their parent object is also saved, either explicitly, or implicitly as part of a `create` call. If the parent record isn't saved, and therefore lacks an `id` the association won't persist. Understanding the distinction between persisting operations versus building objects is crucial to avoiding confusion when using the Rails ORM. This can often be the source of 'unexpected' behavior and is something I've often found myself advising younger engineers on, over the course of many projects.
