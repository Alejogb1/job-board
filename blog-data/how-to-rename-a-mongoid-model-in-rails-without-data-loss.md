---
title: "How to rename a Mongoid model in Rails without data loss?"
date: "2024-12-23"
id: "how-to-rename-a-mongoid-model-in-rails-without-data-loss"
---

Alright, let's tackle this one. It’s a scenario I’ve bumped into a few times over the years, usually after the initial design phases of a project where we've had to adjust model names to better reflect the business logic. Renaming a Mongoid model in a Rails application without losing data requires careful planning and execution. It's not as straightforward as a simple rename, since Mongoid, like other odm/orm tools, uses the class name as part of its internal representation. Therefore, simply changing the class name leads to a disconnect between the model definition and the data stored in MongoDB. I'm going to walk you through the steps based on my previous experience, outlining a safe procedure to get this done.

The core issue is that Mongoid typically stores the collection name in MongoDB based on the class name by convention – pluralizing it and converting it to lowercase, unless explicitly specified otherwise. Renaming the model in Rails without any associated database changes would cause your application to look for the data in a newly named collection that doesn't exist, and, equally problematic, it would not recognize the existing collection as the one that belongs to the new model name.

Here’s the approach I’ve found to be most reliable:

**1. Introduce a New Model:**

Instead of immediately renaming the existing model, start by creating a new model with the desired name. This new model class will effectively mirror the data structure of the original one. For example, let’s say we wanted to rename `OldModel` to `NewModel`. Start by creating `app/models/new_model.rb`:

```ruby
# app/models/new_model.rb
class NewModel
  include Mongoid::Document
  include Mongoid::Timestamps

  field :field1, type: String
  field :field2, type: Integer
  # Add all the other fields from the OldModel
end
```

It’s crucial that this `NewModel` has the *exact same fields* as the original model (`OldModel`), in terms of names and data types. The `include Mongoid::Timestamps` and any custom indices must be included. We’re building a shell that matches the old model, but under a new class name. Don't worry about the data location yet, we are getting there.

**2. Ensure Correct Collection Mapping:**

Now, the crucial step. We need to instruct `NewModel` to use the *same collection* as the original `OldModel`. We achieve this by explicitly setting the `collection_name` for `NewModel` in the model definition:

```ruby
# app/models/new_model.rb
class NewModel
  include Mongoid::Document
  include Mongoid::Timestamps

  store_in collection: :old_models

  field :field1, type: String
  field :field2, type: Integer
  # ... same fields as before
end
```

Here, I’ve used `:old_models` assuming the default Mongoid collection name for `OldModel` was `old_models`. You might need to adjust it if it was explicitly customized. If you’re unsure, the quickest check is to open the Rails console (`rails console`) and inspect `OldModel.collection.name`. This gives you the actual name of the collection stored in the database.

**3. Data Migration (and Verification):**

Now that `NewModel` reads from the existing collection, you need to move the application logic from the `OldModel` to the `NewModel` gradually. This can involve slowly transitioning all the places that access or modify data from `OldModel` to now use `NewModel`. This isn't just a find and replace. Start with the places that are easiest and gradually work your way through to the areas where data access is most critical to the application. After updating your code, run tests extensively to verify.

```ruby
# Example data access update
# Old way (avoid this)
@old_records = OldModel.all

# New way (the right way, use NewModel from now on)
@new_records = NewModel.all
```

**4. Gradual Transition & Feature Flags:**

Don't make changes all at once. Consider using feature flags to gradually switch from using the `OldModel` to using the `NewModel` everywhere in your application. This allows you to monitor the transition and quickly revert if any problems arise, all without causing a full app downtime. If you make a mistake, you'll know you haven't fully broken the app but rather introduced a bug in the new model layer. I have personally found the `rollout` gem to be quite handy for implementing feature flags in Rails.

**5. Removing `OldModel` (Carefully):**

Once you are absolutely certain that `NewModel` is the only one being used, and all references to `OldModel` are gone, and all tests pass, then you can *finally* remove `app/models/old_model.rb`. It’s good practice to remove it entirely from the source code instead of commenting it out. After this removal, you need to update the `NewModel` class by removing the `store_in collection: :old_models` line. This sets it back to the Mongoid default of a collection name based on its class name, thereby renaming the collection effectively in the next step.

```ruby
# app/models/new_model.rb (final version)
class NewModel
  include Mongoid::Document
  include Mongoid::Timestamps

  # NO store_in line now

  field :field1, type: String
  field :field2, type: Integer
  # ... same fields as before
end
```

**6. Final Collection Rename (using a migration):**

We now need a single-use migration to rename the collection on the database side. This is done using the MongoDB driver, not through Mongoid directly. We execute this once and then remove the migration file afterwards.

```ruby
# db/migrate/[timestamp]_rename_old_collection.rb
class RenameOldCollection < Mongoid::Migration::Current
  def change
    db = Mongoid::Clients.default
    db.database.command({
      renameCollection: 'old_models',
      to: 'new_models'
    })
  end
end
```

Now, you can execute this migration with `rake db:migrate`. Mongoid uses `rake db:migrate` but it’s important to understand this is a direct database command through the MongoDB driver, not a Mongoid operation. Be absolutely sure of your backups before running this migration.

**7. Post-Rename Cleanup and Verification:**

After running the migration and verifying everything with the application, you should remove the migration file. With this approach, we have successfully renamed the Mongoid model with minimal risk and a clear path for testing and rollback if needed.

This process isn’t fast. It’s designed to be safe and predictable. Patience is key. Avoid shortcuts, as they can result in data corruption or inconsistencies.

**Recommended Resources:**

*   **"MongoDB: The Definitive Guide" by Kristina Chodorow:** This is an excellent general reference book that delves into the core functionalities of MongoDB, providing a solid foundation for understanding the underlying processes that Mongoid leverages.
*   **"Seven Databases in Seven Weeks" by Eric Redmond and Jim Wilson:** A great resource that compares different database systems, including MongoDB, and can improve your intuition about how they work.
*   **MongoDB Documentation:** The official MongoDB documentation is your primary resource for any specific questions or edge cases. It's constantly updated and always the best source for definitive answers.

This methodology has served me well in various projects, and while each circumstance is unique, I have always found that a methodical, step-by-step approach with feature flags and strong testing, pays dividends. I hope this helps you in your own endeavors.
