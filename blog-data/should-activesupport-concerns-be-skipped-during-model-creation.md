---
title: "Should ActiveSupport concerns be skipped during model creation?"
date: "2024-12-23"
id: "should-activesupport-concerns-be-skipped-during-model-creation"
---

, let’s talk about ActiveSupport concerns and model creation, a topic that I've certainly spent a good chunk of my career pondering and, at times, debugging. I'll structure this around some of the practical experiences I've encountered. The short answer is: it's not a straightforward "yes" or "no". Skipping ActiveSupport concerns during model creation *can* offer certain advantages, but it also introduces trade-offs that require careful consideration.

My perspective comes from years of maintaining and scaling Ruby on Rails applications, where I’ve seen concerns implemented brilliantly, and also quite disastrously. Let's unpack why that's the case.

First, let's establish what we mean by ActiveSupport concerns. For those less familiar, ActiveSupport::Concern is a module that allows you to package up reusable code and inject it into your models (or any other class that can include it). This is intended to mitigate code duplication and enforce the single responsibility principle, moving specific functionalities out of the core model and into well-defined modules. When done correctly, concerns promote a cleaner, more manageable codebase.

Now, the question of skipping them *during* model creation. What does that mean? Well, I've been in situations where we were programmatically generating a large number of models, sometimes even dynamically based on external configurations. In these scenarios, applying every concern at model creation can become a performance bottleneck, especially if the concerns do not directly impact the initial structure or if they perform heavy calculations. Additionally, we were dealing with scenarios where certain concerns needed to be applied conditionally depending on other factors. Including them all by default was, frankly, overkill, and it needlessly slowed things down. This is where skipping them on creation, and applying them later, becomes a compelling strategy.

For instance, let’s imagine a situation where a concern provides functionality related to user tracking in an application. It might need to interact with various services to retrieve specific user information. If we were to programmatically generate, say, 10,000 models representing different data entries and we apply this concern directly during the initial object instantiation for each of them, the process would take considerable time due to its inherent overhead. Instead, we can create a bare-bones model and attach the user-tracking functionality later, for those models that truly need it. This is also crucial if some models are being created programmatically based on different data sources. They may not all require the same set of concerns, or any concern at all.

Now, this approach isn't without its drawbacks. Skipping concerns at creation introduces a level of indirection. It complicates debugging, and there's a risk that we could inadvertently create models that are missing crucial functionality. It requires careful planning and a strict methodology for applying concerns *after* model creation. This also means we need to handle situations where the concerns need to interact with the model's data, which might not be readily available at the point of concern attachment.

So, here's how I've tackled these problems in the past, and hopefully these code snippets provide some context:

**Example 1: Deferred Concern Application**

This demonstrates how we might create a model instance without a specific concern and then add it conditionally:

```ruby
class BaseDataEntry < ActiveRecord::Base
end

module UserTrackingConcern
  extend ActiveSupport::Concern

  included do
    # Some user tracking related functionality
    def track_user_activity
      puts "Tracking user activity for entry ID: #{self.id}"
    end
  end
end

#create a model with no concern
data_entry = BaseDataEntry.create(data: "some data")
puts "Model created without user tracking concern: #{data_entry.class}"

#conditionally include the concern
data_entry.class.include(UserTrackingConcern)
data_entry.track_user_activity # now has it.
puts "Model has user tracking concern: #{data_entry.class}"
```

**Example 2: Dynamic Concern Selection**

In this scenario, the concerns are applied based on the data itself:

```ruby
class BaseDataEntry < ActiveRecord::Base
end

module DataValidationConcern
  extend ActiveSupport::Concern

  included do
    # Validation logic
    def validate_data
      puts "Data validated for entry ID: #{self.id}"
    end
  end
end

module DataProcessingConcern
    extend ActiveSupport::Concern

    included do
        # Processing logic
        def process_data
            puts "Data processed for entry ID: #{self.id}"
        end
    end
end


data_entry_type_a = BaseDataEntry.create(data: "type_a data", type: "a")
data_entry_type_b = BaseDataEntry.create(data: "type_b data", type: "b")

# Apply concerns conditionally
if data_entry_type_a.type == "a"
  data_entry_type_a.class.include(DataValidationConcern)
end

if data_entry_type_b.type == "b"
    data_entry_type_b.class.include(DataProcessingConcern)
end


data_entry_type_a.validate_data
data_entry_type_b.process_data
puts "Concerns applied based on data: #{data_entry_type_a.class}, #{data_entry_type_b.class}"
```

**Example 3: Concern Application with Callbacks**

This shows how a concern might need to interact with the model's state *after* it's created, and uses an after_create callback to activate the concern functionality. This also shows where not adding a concern at all can make sense if the model doesn't actually need to process the data:

```ruby
class BaseDataEntry < ActiveRecord::Base
    after_create :apply_processing
    attr_accessor :needs_processing

    def apply_processing
      if self.needs_processing
        self.class.include(DataProcessingConcern)
        self.process_data
      end
    end

end

module DataProcessingConcern
    extend ActiveSupport::Concern

    included do
        # Processing logic
        def process_data
            puts "Data processed for entry ID: #{self.id}"
        end
    end
end

data_entry_process = BaseDataEntry.create(data: "needs processing", needs_processing: true)
data_entry_skip_process = BaseDataEntry.create(data: "no need to process", needs_processing: false)

puts "Concerns potentially applied via callback: #{data_entry_process.class}, #{data_entry_skip_process.class}"
```

These examples illustrate the general pattern we've used. The key is to have a clear process for attaching concerns post-creation. This typically involves either some kind of tagging on the created models or a separate service that is responsible for scanning through newly created objects and attaching concerns based on specific criteria.

So, in conclusion, should concerns be skipped during model creation? The answer is nuanced. If you are programmatically generating a large number of models, and not all need all the functionality immediately provided by every concern, it's a performance optimisation worth considering. However, it’s not a free lunch. Deferred concern application adds complexity and requires rigorous testing. We also need to consider the potential for missing important business rules, and how to mitigate the increased complexity this creates. The primary lesson I've learned from dealing with this is that you must be deliberate, with a clear plan on when and how concerns are attached after model creation, to avoid creating technical debt down the line. It's a trade-off, not a universal best practice.

For anyone looking to dive deeper into this, I would recommend studying the ActiveSupport module documentation within the Rails API docs. Furthermore, "Refactoring Ruby" by Martin Fowler provides excellent insights into design patterns and modularity that are crucial for effective use of concerns. Finally, "Design Patterns: Elements of Reusable Object-Oriented Software" by the Gang of Four is a timeless classic that helps you understand the broader context of why techniques like concerns exist in the first place. Understanding the theoretical underpinnings helps ensure your architectural choices are well-informed, even when you diverge from the standard application patterns.
