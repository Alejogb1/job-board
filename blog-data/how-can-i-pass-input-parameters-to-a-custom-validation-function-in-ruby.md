---
title: "How can I pass input parameters to a custom validation function in Ruby?"
date: "2024-12-23"
id: "how-can-i-pass-input-parameters-to-a-custom-validation-function-in-ruby"
---

Alright, let's talk about custom validation functions in Ruby and the nuances of parameter passing. I've seen this crop up quite a bit in projects, and there's often a need to go beyond the standard validations that Rails (or other frameworks) offer. The core idea is that you're creating a more specific rule, something that requires more than just checking for the presence of a value or its data type.

The primary challenge, as you've pointed out, is how to feed context or specific parameters into these custom validation checks. A simple method call won't cut it when you need to, for instance, validate a date against a user-defined range or compare a value with another field in the same record. My experience has taught me that relying solely on implicit context can lead to brittle and hard-to-debug code. You really want that explicit control.

The most common approach—and the one I usually lean towards—involves using the `validate` method along with a custom method. This method can then receive the record object itself, allowing it to access other fields, plus any additional parameters we explicitly pass in. Let me elaborate on that with a first example.

**Example 1: Basic Custom Validation with Explicit Parameters**

Imagine you need to ensure a start date is always before the end date for a project record. Here's how you could implement that:

```ruby
class Project < ActiveRecord::Base
  validate :start_date_before_end_date, :on => :create

  def start_date_before_end_date
    return unless start_date && end_date # prevent errors if nil

    if start_date >= end_date
      errors.add(:start_date, "must be before the end date")
    end
  end
end
```

In this initial example, we are not passing explicit parameters; instead, we are using the record's attributes directly. It's basic, but illustrates the initial step. To make it more robust, we could introduce a parameter like a minimum date for the start date:

```ruby
class Project < ActiveRecord::Base
  validate :start_date_within_range

  def start_date_within_range
    min_date = Date.today
    return unless start_date && end_date # prevent errors if nil

    if start_date < min_date
      errors.add(:start_date, "cannot be in the past")
    elsif start_date >= end_date
        errors.add(:start_date, "must be before the end date")
    end
  end
end
```
While the above accomplishes a parameter via a variable within the method, it isn't ideal because it isn't configurable outside the method. Let’s look into using explicit parameter passing.

**Example 2: Using a Block and Configuration**

A slightly more sophisticated way to handle this is to use a block, which opens up the door for even more flexibility and configurability. Blocks let us pass additional data to the validation function. This approach is particularly helpful when you want to configure your validation at the class level:

```ruby
class Project < ActiveRecord::Base
  validate :start_date_within_range, with: -> (project, min_date){
    return unless project.start_date && project.end_date

    if project.start_date < min_date
      project.errors.add(:start_date, "cannot be before #{min_date}")
    elsif project.start_date >= project.end_date
        project.errors.add(:start_date, "must be before the end date")
    end
  }

  def self.set_min_start_date(date)
      @min_start_date = date
  end

  def self.min_start_date
    @min_start_date || Date.today
  end

  def start_date_within_range
      min_date = self.class.min_start_date
      instance_eval(&self.class.validators.find{|v| v.attributes.include?(:start_date_within_range)}.options[:with], self, min_date)
  end
end

Project.set_min_start_date(Date.new(2024, 1, 1))
```
Here, we are passing the method as an argument to the validate method, with the block specified in the `:with` option. We also introduced a class level configuration to provide flexibility on the minimum date, instead of hardcoding it. When the validate method is executed, we locate the correct validator and call its block with the instance of the object as well as the date as parameters.

**Example 3: Parameterized Validator with a Custom Class**

Finally, if you're dealing with more complex validation logic and desire better separation of concerns, creating a custom validator class can be beneficial. This is particularly useful when you find yourself needing similar validation logic in multiple models. The class encapsulates the validation logic, and parameters can be passed during instantiation:

```ruby
class DateRangeValidator < ActiveModel::EachValidator
  def validate_each(record, attribute, value)
      min_date = options[:min_date] || Date.today
      max_date_attribute = options[:max_date_attribute]

      return unless value && record.send(max_date_attribute)

      if value < min_date
        record.errors.add(attribute, "cannot be before #{min_date}")
      elsif value >= record.send(max_date_attribute)
        record.errors.add(attribute, "must be before the #{max_date_attribute}")
      end
    end
end

class Project < ActiveRecord::Base
  validates :start_date, date_range: { min_date: Date.new(2023, 1, 1), max_date_attribute: :end_date }
end
```

In this case, the `DateRangeValidator` is reusable across multiple models. The `validate_each` method receives the record, attribute being validated and its value. It has access to the options hash, where we pass the min date and max date attribute to validate against. This approach offers a clean, declarative way to validate date ranges and is extensible for more attributes and configurations. This separation greatly improves testability and maintainability.

**Further Reading**

For a deeper dive into these concepts, I’d recommend starting with the official Ruby on Rails documentation on Active Record validations. They provide a great baseline understanding of what’s possible out of the box. Additionally, *“Metaprogramming Ruby”* by Paolo Perrotta offers very good insights on the meta-programming involved in techniques like the second and third examples, focusing on building custom validations. Finally, while not directly focused on Ruby validations, *“Refactoring: Improving the Design of Existing Code”* by Martin Fowler can provide a better understanding of how to structure code like this for long-term maintainability.

In summary, passing parameters to custom validation functions in Ruby is achievable using a combination of passing in the record's context and using additional arguments, or more complex techniques using blocks, or dedicated classes. Choose the method that best fits the complexity of your project and your desired maintainability. The goal is to create expressive, testable, and reliable validations to ensure data consistency in your applications. From my experience, being thoughtful about these validation techniques pays off substantially in the long run.
