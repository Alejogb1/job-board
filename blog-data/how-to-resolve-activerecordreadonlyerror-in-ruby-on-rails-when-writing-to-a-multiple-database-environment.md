---
title: "How to resolve ActiveRecord::ReadOnlyError in Ruby on Rails when writing to a multiple database environment?"
date: "2024-12-23"
id: "how-to-resolve-activerecordreadonlyerror-in-ruby-on-rails-when-writing-to-a-multiple-database-environment"
---

Alright, let's talk about `ActiveRecord::ReadOnlyError` in a multi-database rails environment. It’s a situation I've certainly encountered a few times, and it's usually not immediately obvious what’s happening. Back in my days scaling a platform handling both user-generated content and complex transactional data, we definitely ran into this. The short answer is that `ActiveRecord::ReadOnlyError` arises when you attempt to modify a database record via an active record model that is connected to a database configured as read-only. In a multiple database setup, this is fairly common and can be a pain to track down without a systematic approach.

Let's break down what’s happening. Rails, by default, assumes a single database. When you introduce multiple databases – perhaps for scalability or for segregating different types of data – you need to explicitly define which database should be used for each model. The error comes up when you accidentally configure a model to point to a read replica or a connection intended only for reading, and then you try to update or create data through that model. This often manifests in complex applications when data access logic gets interwoven, and a write operation happens on a model that unexpectedly points to the wrong connection.

The core issue revolves around how Rails’ ActiveRecord establishes connections and how these are then mapped to your models. Each model can be set to use a specific connection, which is declared in the model itself, usually within the `establish_connection` block or through more modern rails connection-handling practices using the `connected_to` directive. Failing to specify this or specifying a read-only connection is often the culprit.

Let’s dive into some examples, and I'll try to show you how this looks both when it goes wrong and how to rectify the issue.

**Example 1: The Case of the Read-Only User Model**

Imagine you have a `User` model that is mistakenly configured to point to a read-only replica. This is often caused by a misconfiguration or accidentally pointing to the database's read replica instead of the main write master database.

```ruby
# app/models/user.rb
class User < ApplicationRecord
  # This is where the error originates, if this is a read-only replica
  # and not the main database:
  establish_connection :read_only_replica 
end

# Example of the attempted write operation which throws the error
begin
  user = User.find(1)
  user.email = "newemail@example.com"
  user.save! #This will raise ActiveRecord::ReadOnlyError
rescue ActiveRecord::ReadOnlyError => e
  puts "Caught: #{e.message}"
end
```

In the above example, the `establish_connection` call directs all queries and actions on the `User` model towards a connection named `:read_only_replica`. If that database connection is indeed a read replica, any attempt to save the modified user record will result in an `ActiveRecord::ReadOnlyError`. That particular setup was a headache for us until we actually dug into what the models were connecting to via the `database.yml` settings and in the respective model files.

**Example 2: Correctly Connecting to a Write Database**

Here's the fix. You must ensure your `User` model (or any model needing write access) is configured to connect to the database where writes are allowed.

```ruby
# app/models/user.rb
class User < ApplicationRecord
  establish_connection :primary_database  # Correct connection!
end


begin
  user = User.find(1)
  user.email = "newemail@example.com"
  user.save!  # This should now succeed, if primary_database is a writable database
  puts "User updated successfully"
rescue ActiveRecord::ReadOnlyError => e
  puts "Caught error again: #{e.message}"
end

```

In this revision, I changed the connection to `:primary_database`, which should point to your main database that allows write operations. This, seemingly simple, adjustment resolves the issue when you correctly point the model to the correct connection in the configuration, which usually sits within `config/database.yml`. This is a critical step that's often overlooked, especially in more complex applications.

**Example 3: Using `connected_to` for Dynamic Connection Handling**

Modern Rails offers the `connected_to` method, which provides a more flexible way to handle connections, especially within a block, for very specific cases where you want to route some operations to one connection and the rest to another. It's also helpful for scoping database access with a fine-grained control.

```ruby
# This scenario might be where some logging needs to go to a specific
# audit logging database while the user data needs to write to the primary.
class AuditLog < ApplicationRecord
  establish_connection :audit_logs
end

class User < ApplicationRecord
  establish_connection :primary_database # Correct connection!
end

def update_user_and_audit_log(user_id, new_email)
  User.find(user_id).tap do |user|
   
     AuditLog.connected_to(database: :audit_logs) do
       AuditLog.create!(message: "User id #{user_id} updated email to: #{new_email}")
     end

    user.email = new_email
    user.save! # User model connects to primary_database for updates
  end
end

begin
  update_user_and_audit_log(1, "new_email_2@example.com")
  puts "User updated, and audit log created successfully"
rescue ActiveRecord::ReadOnlyError => e
  puts "Caught error: #{e.message}"
end
```

Here, the `connected_to(database: :audit_logs)` block scopes the create operation for the `AuditLog` model to use the `audit_logs` connection, while the `User` model, configured through `establish_connection` uses its own primary database connection, demonstrating how you can orchestrate write operations across different databases within the same flow of operations. This was particularly useful for us when migrating data and ensuring our audit trails were correct while also handling data with correct transactionality.

**Key Takeaways and How to Diagnose the Issue**

When debugging an `ActiveRecord::ReadOnlyError`, follow these steps:

1.  **Inspect the Model’s Connection:** Use `ModelName.connection_config` to see what connection settings are used by the specific model. Double-check the connection details in your `config/database.yml` file, or environment variable settings for database access. Make sure they point to a write-capable database.
2.  **Trace the Error:** Pinpoint exactly where the error occurs within your code. Stack traces are your friends.
3.  **Verify the Database Connection:** Ensure the connection specified for the model is intended for both reads and writes. Review your infrastructure setup to confirm the intended use of each database connection.
4.  **Use `connected_to`:** Leverage `connected_to` for more granular control of database access within blocks of code. This is particularly useful when specific data access needs to go to different databases.
5.  **Review Your Data Flow:** Especially in more complex applications, data can be read and then modified across different services. Make sure you're understanding the full path your data is taking, to know when write access should be attempted.

For further learning, I recommend you look into "Agile Web Development with Rails 7" for detailed information on Rails configurations, and "Designing Data-Intensive Applications" by Martin Kleppmann, which provides a broader context on database architecture and the challenges of distributed systems, of which multi-database setups are an example. Understanding the database architectural concerns is key. Another resource you might find insightful is "Patterns of Enterprise Application Architecture" by Martin Fowler, which offers a more abstract approach but is helpful to organize your domain logic and models.

In my experience, meticulous review of model configurations and connection settings is paramount. This will eventually lead to a solid and less error prone multi-database rails setup. The key is often not just coding the solutions, but understanding why the setup itself behaves the way it does. It really boils down to the connections themselves.
