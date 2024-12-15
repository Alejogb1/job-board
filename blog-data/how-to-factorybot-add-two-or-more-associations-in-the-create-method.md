---
title: "How to factory_bot add two or more associations in the create method?"
date: "2024-12-15"
id: "how-to-factorybot-add-two-or-more-associations-in-the-create-method"
---

i see you're having some trouble with factory_bot and creating records with multiple associations. i've definitely been there, staring at the screen wondering why my tests are failing because the data isn't set up the way i expect. it's a pretty common stumbling block when you're getting into more complex testing scenarios, especially with rails apps. let me walk you through how i usually handle this, hopefully it'll clear things up for you.

so, the core problem is that factory_bot's `create` method by default makes one associated record for each association you define. that's great for simple cases but falls short when you need several related items. the key is to leverage factory_bot's `transient` attributes, and the `after(:create)` callback.

let me paint you a picture. a few years back, i was building an e-commerce platform, and i had this `order` model with `line_items`. each `line_item` had a reference to a `product`. a single order could have many line items and each line item could point to a product. i needed to test some scenarios with orders containing multiple items. i initially tried just defining associations and running `create(:order)` but that gave me only one line item. that's not very useful. my tests would not be comprehensive enough to cover different scenarios. so, i dove into the factory_bot documentation.

first, let's break down the factory definition. here's a basic setup for an `order` factory:

```ruby
FactoryBot.define do
  factory :order do
    customer
  end
end

FactoryBot.define do
  factory :line_item do
    order
    product
    quantity { 1 }
  end
end

FactoryBot.define do
  factory :product do
    name { "Test Product" }
    price { 10.00 }
  end
end
```

notice there's no mention of how many `line_items` we want. if you use `create(:order)` you'll get a single line item. what we actually need is a way to tell factory_bot we want multiple line items, and that's where the `transient` and `after(:create)` comes in. `transient` attributes are not stored on the record. they're just variables used during factory creation.

here's how i would modify the order factory to handle multiple line items:

```ruby
FactoryBot.define do
  factory :order do
    customer

    transient do
      line_items_count { 2 }
    end

    after(:create) do |order, evaluator|
      create_list(:line_item, evaluator.line_items_count, order: order)
    end
  end
end

```

this is the key snippet. the `transient` block sets up an attribute called `line_items_count` which defaults to `2`. then, the `after(:create)` block executes *after* the `order` record is created, we use `create_list` method and with `line_items_count` we can create the number of line items defined, in this case we will create 2 line items. inside the block, `evaluator` gives us access to the transient attributes. `create_list` then generates the specified number of `line_item` records and associates each one with the just-created `order` instance.

now you can use it like this:

```ruby
# creates an order with 2 line_items
order = create(:order)
expect(order.line_items.count).to eq(2)

# creates an order with 5 line_items
order = create(:order, line_items_count: 5)
expect(order.line_items.count).to eq(5)
```

let's look into a slightly different case. say you want to customize the created associations. for example, if each `line_item` needed different products, and different quantities. the above example does not allow that level of customization. we can again use the `transient` attribute, but this time passing in an array of values:

```ruby
FactoryBot.define do
  factory :order do
    customer

    transient do
      line_items_data { [{product: create(:product), quantity: 1}, {product: create(:product), quantity: 3}] }
    end

    after(:create) do |order, evaluator|
      evaluator.line_items_data.each do |line_item_data|
        create(:line_item, order: order, product: line_item_data[:product], quantity: line_item_data[:quantity])
      end
    end
  end
end
```

in this revised factory, `line_items_data` is a transient attribute that holds an array of hashes. each hash contains the attributes to customize the creation of each line item. inside the `after(:create)` block we loop through these items and create the line items. we can now use the factory as follows:

```ruby
# creates an order with line items
order = create(:order)
expect(order.line_items.count).to eq(2)
expect(order.line_items.first.quantity).to eq(1)
expect(order.line_items.second.quantity).to eq(3)

# you can easily change or increase number of line items, and their attributes
order = create(:order, line_items_data: [
  {product: create(:product, name: "book"), quantity: 1},
  {product: create(:product, name: "pen"), quantity: 2},
  {product: create(:product, name: "cup"), quantity: 3},
])
expect(order.line_items.count).to eq(3)
expect(order.line_items.first.product.name).to eq("book")
expect(order.line_items.second.product.name).to eq("pen")
expect(order.line_items.third.product.name).to eq("cup")
```

this approach gives you a lot of flexibility for different scenarios, this can be used to create different related items and with the desired attributes for each one. this will help your testing code be more robust.

one last example, sometimes you want to generate a dynamic number of associated records, for example, depending on some calculation, let's say you are dealing with a user with multiple posts, and you want the number of posts to change randomly for each user, here is an example of how that can be done, let's say a post has a user association:

```ruby
FactoryBot.define do
    factory :user do
    name { Faker::Name.name }

    transient do
      posts_count { rand(1..5) } # Random number between 1 and 5 posts
    end

    after(:create) do |user, evaluator|
      create_list(:post, evaluator.posts_count, user: user)
    end
  end
end

FactoryBot.define do
  factory :post do
      user
      title { Faker::Lorem.sentence }
      content { Faker::Lorem.paragraph }
  end
end
```

now, each time we create a user, it will create a random number of posts between 1 and 5. this can be useful in testing, for example, scenarios like a user displaying their posts in their profile. you could then test for different cases based on different number of posts. this adds a nice level of randomness to your testing. i've found that sometimes this gives me coverage of cases that i would have never thought of.

remember, testing is not just about making sure things work, but also making sure they work *correctly* under different circumstances. that randomness will give you the coverage you might be missing.

when i was starting i was spending way too much time trying to figure out the right combination of `build` and `create` and when to use `build_stubbed` or `attributes_for` and other factory bot methods. it took a lot of debugging, reading the official factorybot documentation is key to solve many of these issues.

i encourage you to dive deeper into the factory bot documentation to get a better understanding. specifically pay attention to the `transient` attribute and `after(:create)` callbacks, those 2 are game changers. read "working with rails" by david heinemeier hansson for more context on testing methodologies, as well as testing patterns in rails. for a deep dive on design patterns, i recommend "design patterns: elements of reusable object-oriented software" by erich gamma. there's no substitute for a solid foundation. they are not directly about factory bot but understanding these principles will lead you to more organized, readable, and easily debuggable code, and that includes your testing code.

oh, and one time, i spent 3 hours debugging a factory bot issue. turns out, i had a typo in one of the association names, so i was creating records that did not belong to each other. now i double-check those associations. it's always the little things isn't it? it's funny when you spend hours on something that should have been obvious. but i learned my lesson i guess.

i hope this detailed explanation helps. let me know if you have any more questions, i'm happy to help. good luck.
