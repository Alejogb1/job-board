---
title: "How do I create delegated_type associations using Factory Bot/Girl?"
date: "2024-12-23"
id: "how-do-i-create-delegatedtype-associations-using-factory-botgirl"
---

Okay, let's unpack this. Delegated types, eh? Been there, done that – more times than I care to count. It's a fairly common scenario when you're trying to implement polymorphism in your models, and you need your test suite to understand these relationships via something like FactoryBot (or FactoryGirl as it was previously known). I’ve certainly had my share of headaches when getting it to play nicely with complex associations. Let’s break down how to approach this, making sure we cover the necessary ground with some tangible examples.

The crux of the matter here is that delegated types imply an inheritance or polymorphism pattern in your database schema. Imagine a scenario where you have an `attachment` model that can be either an `image` or a `document`. Both `image` and `document` might have distinct attributes, but they are, ultimately, both 'attachments.' The `attachment` model itself typically contains the foreign key and type information. FactoryBot, in its straightforward usage, might struggle with this relationship unless explicitly configured. It's not magic; it’s about instructing the factory on how to correctly create these interconnected records.

Let’s start with what this might look like in our models. Consider this basic setup:

```ruby
# app/models/attachment.rb
class Attachment < ApplicationRecord
  belongs_to :attachable, polymorphic: true
end

# app/models/image.rb
class Image < ApplicationRecord
  has_one :attachment, as: :attachable, dependent: :destroy
end

# app/models/document.rb
class Document < ApplicationRecord
  has_one :attachment, as: :attachable, dependent: :destroy
end
```

In this case, `Attachment` is the polymorphic container and `Image` and `Document` are the polymorphic models. Our goal is to set up FactoryBot such that we can create these relationships seamlessly.

**Example 1: Defining a Specific Type Factory**

The most direct approach is to create individual factories for each of your polymorphic types and a separate factory for the container. This allows for precise control over the generated data. Here's how that might look:

```ruby
# spec/factories/attachments.rb
FactoryBot.define do
  factory :attachment do
    association :attachable, factory: :image
  end

  factory :image do
    filename { 'test_image.jpg' }
    size { 1024 }
  end

  factory :document do
      filename { 'test_document.pdf' }
      page_count { 10 }
  end
end
```

This factory definition allows you to create an `attachment` linked to an `image` by default. If you want to create an attachment for a document you’d need to adjust the `attachable` association. In your test you might write:

```ruby
it "creates an image attachment" do
  attachment = create(:attachment)
  expect(attachment.attachable).to be_a(Image)
  expect(attachment.attachable.filename).to eq('test_image.jpg')
end

it "creates a document attachment" do
  attachment = create(:attachment, attachable: create(:document))
  expect(attachment.attachable).to be_a(Document)
  expect(attachment.attachable.filename).to eq('test_document.pdf')
end
```

This illustrates a key point: You need to be explicit about what type of associated model you are creating. When an `attachable` factory is explicitly provided to the attachment factory, FactoryBot skips the default association definition, creating the attachment based on what was passed.

**Example 2: Using Traits for Different Types**

A more flexible and streamlined way to approach this, particularly if you need to create attachments linked to various types frequently, is using traits. Traits allow you to configure factories differently based on given conditions.

```ruby
# spec/factories/attachments.rb
FactoryBot.define do
  factory :attachment do
    association :attachable, factory: :image

    trait :with_image do
      association :attachable, factory: :image
    end

    trait :with_document do
      association :attachable, factory: :document
    end
  end

  factory :image do
    filename { 'test_image.jpg' }
    size { 1024 }
  end

  factory :document do
    filename { 'test_document.pdf' }
    page_count { 10 }
  end
end
```

With this setup, you can now generate an `attachment` with an associated `image` by default, or with a linked `document`, like so:

```ruby
it "creates an attachment with an image using a trait" do
    attachment = create(:attachment, :with_image)
    expect(attachment.attachable).to be_a(Image)
    expect(attachment.attachable.filename).to eq('test_image.jpg')
end


it "creates an attachment with a document using a trait" do
    attachment = create(:attachment, :with_document)
    expect(attachment.attachable).to be_a(Document)
    expect(attachment.attachable.filename).to eq('test_document.pdf')
end
```

The advantage here is readability and conciseness when specifying the required type of attachment in the test itself.

**Example 3: A More Dynamic Approach Using a Custom After Create Hook**

While traits and individual type-specific factories work well in many cases, there are scenarios where you might want the factory to be a bit more 'intelligent', inferring the related type from a specified parameter. Let’s say, for example, you wish to provide a type string.

```ruby
# spec/factories/attachments.rb
FactoryBot.define do
  factory :attachment do
    transient do
      attachable_type { 'image' }
    end

    after(:build) do |attachment, evaluator|
      case evaluator.attachable_type
        when 'image'
          attachment.attachable = build(:image)
        when 'document'
          attachment.attachable = build(:document)
      end
    end

  end

  factory :image do
    filename { 'test_image.jpg' }
    size { 1024 }
  end

  factory :document do
    filename { 'test_document.pdf' }
    page_count { 10 }
  end
end
```

Here, we are using a transient variable (`attachable_type`). In our `after(:build)` hook, we use the `evaluator` to determine which sub-factory to use, and then build the associated object.

```ruby
it "creates an attachment with a specific type dynamically" do
    attachment = create(:attachment, attachable_type: 'document')
    expect(attachment.attachable).to be_a(Document)
    attachment = create(:attachment, attachable_type: 'image')
    expect(attachment.attachable).to be_a(Image)
end
```

This can be useful when you have a lot of potential delegated types. Just be mindful that using `after(:build)` here will create in memory objects that are not persisted unless you use the `create` method.

**Key Considerations and Further Learning**

These examples demonstrate three ways to handle delegated types with FactoryBot. Choose the one that best fits the complexity of your models and your preferences for readability and maintainability.

For a deeper understanding of object-oriented design patterns, including polymorphism, I recommend exploring "Design Patterns: Elements of Reusable Object-Oriented Software" by Erich Gamma, Richard Helm, Ralph Johnson, and John Vlissides (often referred to as the Gang of Four). It's an essential read for anyone tackling complex code bases. Also, I'd advise digging into the official documentation for FactoryBot, specifically focusing on `association`, `traits`, `transient` and the `after` callbacks – understanding the lifecycle of FactoryBot is paramount. Lastly, look into "Refactoring: Improving the Design of Existing Code" by Martin Fowler for a guide on how to refactor code for better design and testability.

In my experience, a good test setup using FactoryBot isn’t about just getting the tests to pass. It’s about clearly expressing the relationships between your objects and enabling maintainability in the long run. This often means making intelligent choices on factory structure, rather than opting for the simplest initial solution that might not scale well over time. I hope that offers a practical start; delegated types can indeed be tricky, but with the right techniques and understanding, they are far from insurmountable.
