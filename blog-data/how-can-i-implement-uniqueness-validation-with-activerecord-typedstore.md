---
title: "How can I implement uniqueness validation with activerecord-typedstore?"
date: "2024-12-23"
id: "how-can-i-implement-uniqueness-validation-with-activerecord-typedstore"
---

 Dealing with uniqueness constraints when using `activerecord-typedstore` can indeed present a unique set of challenges. It's not your typical database-level unique index situation, and I've definitely seen teams stumble over this in practice, particularly in larger projects where data integrity is paramount. The key here lies in understanding how `typedstore` stores data and crafting a suitable validation strategy around it.

The core issue is that `typedstore` serializes attributes into a single database column, typically as a json or hstore data type. This means your traditional database unique indexes aren't going to directly work on specific keys within your stored data. You might define a key like 'email' within your typed store, but the database only sees a monolithic string or a json object. Therefore, directly creating an index on the email key isn't feasible; instead, the uniqueness has to be validated on the application level.

My own experiences with this trace back to a now thankfully archived project, where we were modeling user profiles. These profiles were flexible enough to contain all sorts of information that couldn’t be easily modeled with static columns. We initially thought we could just lean on traditional validations. Oh boy, were we wrong. We quickly learned that without a proper strategy for enforcing uniqueness on attributes within the store, data quality suffered significantly. Duplicates started creeping in, which led to cascading issues in other dependent systems.

So, how do we effectively handle this? We need to implement uniqueness validations in our ActiveRecord model. It's not that different from standard ActiveRecord validations but requires adapting our approach for the dynamic nature of the `typedstore`. We will be using model level validations to ensure we enforce uniqueness of the properties stored within our typedstore.

The key strategy involves:

1.  **Defining a Validation Method:** We’ll create a custom validation method that queries the database based on the value of our specific typed store field.
2.  **Handling Different Data Types:** Be mindful of how you are storing the data. If you are using a JSON string or a HSTORE, you might need to use database specific methods to query the stored values.
3.  **Context Matters:** Validate uniqueness depending on the context of your application. You might need to ignore specific records when updating data.

Let me illustrate with three practical scenarios and code examples.

**Scenario 1: Basic Email Uniqueness**

Let's assume you have a User model with a `profile` typedstore which stores email address. We will implement a custom validation method to ensure that the email address is unique across user records.

```ruby
# app/models/user.rb
class User < ApplicationRecord
  include ActiveRecord::TypedStore::Model

  typed_store :profile do |s|
    s.string :email
  end

  validate :email_uniqueness

  private

  def email_uniqueness
    if profile_changed? && profile['email'].present?
      existing_user = User.where("profile ->> 'email' = ?", profile['email']).where.not(id: id).first

      if existing_user
        errors.add(:profile, "email must be unique")
      end
    end
  end
end
```

In this example, we've created a `email_uniqueness` validator that triggers only if the `profile` attribute has changed and if the email address is not blank. This validation performs a query against the database looking for records where a profile's json representation of the "email" key matches the email address stored in the current record's profile. Crucially, we exclude the current record using `where.not(id: id)` when updating the record. If a matching record is found, we add an error.

**Scenario 2: Uniqueness with Complex Queries (Using Hstore)**

In this case, we'll consider a scenario where the email is inside of a nested object inside the typed store using a hstore.

```ruby
# app/models/user.rb
class User < ApplicationRecord
  include ActiveRecord::TypedStore::Model

  typed_store :profile, types: :hstore do |s|
    s.hash :contact
  end

  validate :contact_email_uniqueness

  private

  def contact_email_uniqueness
    if profile_changed? && profile.key?('contact') && profile['contact'].key?('email') && profile['contact']['email'].present?
      existing_user = User.where("profile @> hstore('contact', 'email=>\"#{profile['contact']['email']}\"')").where.not(id: id).first

      if existing_user
        errors.add(:profile, "contact email must be unique")
      end
    end
  end
end
```

Here, we have the `contact` key with a nested `email` key. Since we are using hstore, we can take advantage of the hstore syntax to perform the query, where we look for records that contain the specific key value pair.

**Scenario 3: Scope Uniqueness Validation**

Finally, let’s consider a scenario where uniqueness is scoped within an associated model. Let's assume a product can have attributes stored in typed store, and we want to ensure the same sku is not used within the scope of a company.

```ruby
# app/models/product.rb
class Product < ApplicationRecord
  belongs_to :company
  include ActiveRecord::TypedStore::Model

  typed_store :details do |s|
    s.string :sku
  end

  validate :scoped_sku_uniqueness

  private

  def scoped_sku_uniqueness
    if details_changed? && details['sku'].present?
      existing_product = company.products.where("details ->> 'sku' = ?", details['sku']).where.not(id: id).first

      if existing_product
        errors.add(:details, "sku must be unique within the company")
      end
    end
  end
end
```

This code uses the relationship to ensure the sku is unique within the company. The query gets performed on the associated products, and we still exclude the current record from the search using `where.not(id:id)`.

**Important Considerations:**

*   **Database Type:** The way you query the json attributes within a typed store is dependent on the database. The examples above use PostgreSQL syntax, and you would need to adapt them for other databases such as MySQL or Sqlite.
*   **Indexes:** While you can't directly create a unique index on a typed store key, you *can* add a functional index. This would be an index on an expression like `(profile ->> 'email')`. Consult your database documentation on how to implement those. This could improve query performance significantly if you have a large number of records.
*   **Performance:** Be mindful of query performance. For very large datasets or frequent validations, it might be necessary to explore more specialized techniques and potentially denormalization strategies, or even offloading this validation to a separate asynchronous worker.
*   **Transactions:** Remember that validation happens within the context of a transaction. This prevents race conditions where data is created between the check and the save operations.

**Further Reading:**

*   **PostgreSQL Documentation:** Specifically review the sections on JSON functions and operators and Hstore functions and operators. The knowledge of how to query your data within your database will prove invaluable for validating data that's stored using these types.
*   **The Pragmatic Programmer by Andrew Hunt and David Thomas:** While not specific to activerecord-typedstore, this book offers general guidance on defensive programming and data integrity, which are crucial when implementing such validation scenarios.
*   **"Refactoring Databases" by Scott W. Ambler and Pramod J. Sadalage:** Understanding database refactoring principles is key when your data models become more complex. This will be useful when tackling issues with validation performance in case your system starts to scale up.

In summary, implementing uniqueness validations with `activerecord-typedstore` requires a deliberate approach that recognizes the limitations of standard database indexes and employs application-level validations to enforce data integrity. I have seen my fair share of production outages because of similar issues, so taking the time to implement this correctly is essential. It’s not a silver bullet, but applying this combination of custom validations, database-specific queries, and thoughtful indexing can get you the results you need.
