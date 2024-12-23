---
title: "How can I validate a field based on a parent's attribute in Rails?"
date: "2024-12-23"
id: "how-can-i-validate-a-field-based-on-a-parents-attribute-in-rails"
---

Let’s tackle this interesting scenario of validating a field based on its parent’s attributes. It's a situation that, if you've been around the rails block for a while, you’ve probably encountered at least once or twice. I remember a project back at StellarTech, a platform for managing complex product configurations, where this very requirement kept cropping up repeatedly. We had nested models—`Product` owning multiple `Component` records—and certain component attributes needed validation based on the selected product type. If we were handling a “premium” product, for instance, a component’s ‘material’ field might have stricter requirements than if it were a ‘standard’ product. This is a common enough use case, and Rails provides several elegant ways to handle it.

Essentially, you’re aiming for conditional validation—validating a child record’s attribute based on the data of its parent record. The most straightforward way involves using custom validation methods. Let's break down the process and look at some specific code examples.

First, I’ll outline the approach, then we’ll jump into specific cases with code. The key concept is to access the parent object within the child's validation method. Rails makes this relatively simple via the associations we define in our models. The logic can be embedded directly in a custom validator or you can use a separate method for cleaner code. We’ll explore both options here.

Let's start with a simple scenario: a `Component` should have a `material` attribute set only if the associated `Product` is of the type “premium.” The `Product` model might look something like this:

```ruby
class Product < ApplicationRecord
  has_many :components, dependent: :destroy
  enum product_type: { standard: 0, premium: 1 }
end
```

And here is the `Component` model with custom validation:

```ruby
class Component < ApplicationRecord
  belongs_to :product

  validate :material_required_for_premium_product

  def material_required_for_premium_product
    if product.premium? && material.blank?
      errors.add(:material, "must be specified for premium products")
    elsif !product.premium? && !material.blank?
      errors.add(:material, "should not be specified for standard products")
    end
  end
end
```

In this example, we use `validate :material_required_for_premium_product` which is the more streamlined approach. The `material_required_for_premium_product` method performs the conditional check and adds the appropriate error to the `errors` collection if the validation fails. We’re able to access the `product` object via the `belongs_to :product` association, and then using rails enum method premium? to determine the product type. I have personally found that this method tends to be simpler and easier to follow when dealing with one or two conditional checks.

Now, consider a slightly more complicated scenario. Let’s say that the `material` attribute should have a specific format only when the `Product` has the attribute ‘manufactured_on’ in the future. For this, let's assume that `manufactured_on` is stored as a date.

Here’s how it might look:

```ruby
class Product < ApplicationRecord
  has_many :components, dependent: :destroy
  attribute :manufactured_on, :date
end

class Component < ApplicationRecord
  belongs_to :product
  validate :material_format_validation

  def material_format_validation
    if product.manufactured_on.present? && product.manufactured_on > Date.today
      unless material =~ /^[A-Z]{3}-\d{3}$/ # example format validation
        errors.add(:material, "must match format 'XXX-999' when product is manufactured in the future")
      end
    end
  end
end
```

Here, we’re checking if `manufactured_on` is a date in the future. If true, then we are applying more specific validations on the `material` field using a regular expression. This is still done within a custom method that is called by the `validate` keyword, but it adds the functionality for future date conditional logic. This illustrates that you’re not restricted to just enums, you can leverage any attribute of the parent to define the validation rules.

Now, let's take an example of using a custom validator. This is often beneficial for more complex validation logic that you might want to reuse across several models or when it involves multiple attributes from the parent and child models. First, you define your custom validator:

```ruby
class MaterialFormatValidator < ActiveModel::Validator
  def validate(record)
    product = record.product

    if product.present? && product.product_type == 'premium' && record.material.present?
      unless record.material =~ /^[A-Z]{2,}-\d{2,}$/
        record.errors.add(:material, "must match a specific format for premium products")
      end
    elsif product.present? && product.product_type == 'standard' && record.material.present?
        unless record.material =~ /^[a-z]{2,}\d{2,}$/
            record.errors.add(:material, "must match a different format for standard products")
        end
    end
  end
end
```
And here is the `Component` model utilizing the custom validator:

```ruby
class Component < ApplicationRecord
  belongs_to :product
  validates_with MaterialFormatValidator
end
```

Here, you encapsulate all the validation logic within a reusable class which is called using `validates_with MaterialFormatValidator`. As you can see, we can also check multiple conditions, for example, validating two different formats based on different product types.

In summary, when validating a field based on its parent's attributes in Rails, the approach centers on accessing the parent object within the child’s validation logic, using either custom validation methods or custom validator classes. For simpler cases, custom methods integrated using the `validate` keyword, as in the first two examples, are often more straightforward and readable. However, when you need to apply complex or reusable logic, leveraging custom validator classes provides better code organization and maintainability. The key is to leverage the existing model relationships and ruby control flows to access all the information you need and construct your validation logic accordingly. Remember, clarity and maintainability are key; choose the approach that best fits your specific context.

For further reading, I'd recommend delving into the 'Active Model Validations' guide in the official Rails documentation. It provides an exhaustive overview of validation techniques. Additionally, "Rails 6 in Action" by Ryan Bigg is an excellent resource for understanding model validations in depth. And finally, to better understand the concept of conditional validation it might be helpful to review the 'Patterns of Enterprise Application Architecture' by Martin Fowler, as it is generally a great architecture overview, and validates well known and widely used patterns in enterprise apps.
