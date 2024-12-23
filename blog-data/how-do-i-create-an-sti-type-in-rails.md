---
title: "How do I create an STI type in Rails?"
date: "2024-12-23"
id: "how-do-i-create-an-sti-type-in-rails"
---

Alright,  Single Table Inheritance, or STI, in Rails – it's a topic I've bumped into a fair few times over the years, and it’s usually a case of balancing design elegance with potential pitfalls. The core idea, of course, is to represent a hierarchy of related classes using a single database table. This can be wonderfully convenient for certain scenarios, but it also needs careful consideration. I've seen projects where STI was a perfect fit, and others where it caused more headaches than it solved.

Essentially, in Rails, you define a base class that corresponds to your table, and then create subclasses that inherit from it. Rails uses a special `type` column in your database table to keep track of which subclass an individual row represents.

Now, let's get down to the nuts and bolts of creating an STI type. Imagine, for example, a scenario I faced a few years back working on an e-commerce application. We had various types of promotional items: `Discount`, `FreeShipping`, and `Coupon`. Using STI, we aimed to manage them all in one table, `promotions`.

Here's how we started with the models:

```ruby
# app/models/promotion.rb
class Promotion < ApplicationRecord
  validates :name, presence: true
  # Other common validations and logic
end

# app/models/discount.rb
class Discount < Promotion
  validates :discount_percentage, presence: true, numericality: { greater_than: 0, less_than_or_equal_to: 100 }
  # Discount-specific logic
end

# app/models/free_shipping.rb
class FreeShipping < Promotion
  validates :minimum_order_amount, presence: true, numericality: { greater_than_or_equal_to: 0 }
  # FreeShipping-specific logic
end


# app/models/coupon.rb
class Coupon < Promotion
   validates :code, presence: true, uniqueness: true
   validates :discount_amount, presence: true, numericality: { greater_than: 0 }
   # Coupon-specific logic
end
```

In this setup, `Promotion` is our base class. `Discount`, `FreeShipping`, and `Coupon` inherit from it. Rails automatically adds a `type` column to the `promotions` table, assuming you haven't created that column beforehand, and when you create a `Discount`, `FreeShipping`, or `Coupon` object, the value in the `type` column gets set automatically using the name of the class, in this case, 'Discount', 'FreeShipping' or 'Coupon'. That's how it understands that those records are instances of the correct child class. If you already have data in the `promotions` table, ensure that the `type` column exists; Rails won't infer this for pre-existing databases, and you might have to add it via a migration.

The migration to create the table, before adding any specific columns for individual types, would initially be fairly basic:

```ruby
# db/migrate/YYYYMMDDHHMMSS_create_promotions.rb
class CreatePromotions < ActiveRecord::Migration[7.0]
  def change
    create_table :promotions do |t|
      t.string :name
      t.string :type # This is the crucial STI column
      t.timestamps
    end
  end
end
```

Then in subsequent migrations, you would add the type specific columns (like `discount_percentage`, `minimum_order_amount`, `code`, and `discount_amount`).

A key aspect I found helpful back then, and I still use today, is leveraging the model logic. For example, we can add methods specific to each type:

```ruby
# app/models/discount.rb
class Discount < Promotion
  validates :discount_percentage, presence: true, numericality: { greater_than: 0, less_than_or_equal_to: 100 }

  def apply(price)
    price * (1 - (discount_percentage.to_f / 100))
  end
end

# app/models/free_shipping.rb
class FreeShipping < Promotion
    validates :minimum_order_amount, presence: true, numericality: { greater_than_or_equal_to: 0 }

    def eligible?(order_amount)
      order_amount >= minimum_order_amount
    end
end

# app/models/coupon.rb
class Coupon < Promotion
    validates :code, presence: true, uniqueness: true
    validates :discount_amount, presence: true, numericality: { greater_than: 0 }

    def apply(price)
      price - discount_amount
    end
end
```

In this revised example, the `Discount` model has a method to calculate the discounted price, `FreeShipping` has a method to check the eligibility for free shipping based on order amount, and `Coupon` has a method to apply the coupon discount amount.

Now, creating and querying objects becomes straightforward:

```ruby
discount = Discount.create(name: "Summer Sale", discount_percentage: 20)
free_shipping = FreeShipping.create(name: "Free Shipping over $50", minimum_order_amount: 50)
coupon = Coupon.create(name: "Weekend Special", code: "WEEKEND10", discount_amount: 10)


puts discount.type # Output: Discount
puts free_shipping.type # Output: FreeShipping
puts coupon.type # Output: Coupon


Promotion.all.each do |promotion|
  puts "#{promotion.name} is a #{promotion.type}"
end

# Output: Summer Sale is a Discount
# Output: Free Shipping over $50 is a FreeShipping
# Output: Weekend Special is a Coupon


Discount.all.each do |discount|
  puts "Discount #{discount.name} has percentage #{discount.discount_percentage}"
end
# Output: Discount Summer Sale has percentage 20
```

As you can see, querying and working with these objects is pretty seamless. Rails takes care of the plumbing behind the scenes by using that `type` column to instantiate the correct child class.

It's important to remember the limitations. While STI can be advantageous for simpler hierarchies, as the number of subclasses grows and their specific attributes and logic diverge drastically, an STI structure can lead to a very wide table with many columns only used for specific rows. The model can also become cluttered with validations and methods that apply to a small number of classes. I've experienced the challenges of navigating this complexity first-hand, and I'd advise you to consider alternatives like polymorphic associations or dedicated tables for each class if you anticipate substantial divergence in your subclass requirements.

Regarding further learning, I'd suggest focusing on some core material. For a comprehensive understanding of object-oriented principles that underpin STI, Erich Gamma, Richard Helm, Ralph Johnson, and John Vlissides’s "Design Patterns: Elements of Reusable Object-Oriented Software" is still relevant. For a Rails-specific deep dive, look into the relevant sections of the official Rails documentation, as it is updated regularly to reflect the current practices and recommendations. Also, “Agile Web Development with Rails 7” by David Heinemeier Hansson is a great source to deepen your understanding. These will give you a better grasp of when and how to apply this pattern effectively, and more importantly, when it might not be the best approach.

Finally, keep an eye on performance, specifically when working with large STI tables. Efficient indexing and targeted queries are important. If you have a table with millions of rows and diverse attribute sets for the different types, you might want to rethink your strategy.

So, in short, creating an STI type in Rails is primarily about defining the base class and its subclasses, and letting Rails manage the `type` column. While it’s elegant and convenient for simpler hierarchies, understand its limitations and consider alternatives if your needs become complex. Approach it with care, and it can be a valuable tool.
