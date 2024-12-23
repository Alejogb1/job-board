---
title: "Why does `db:migrate` modify enum types in `schema.rb`, causing `db:schema:load` to fail?"
date: "2024-12-23"
id: "why-does-dbmigrate-modify-enum-types-in-schemarb-causing-dbschemaload-to-fail"
---

Okay, let’s tackle this one. I’ve seen this particular scenario play out a few times, usually late at night when deadlines are looming and the last thing anyone needs is a schema load failing. It's a frustrating experience that stems from a combination of how Rails handles enums and how `db:migrate` and `db:schema:load` function. Let’s break down why this happens, and more importantly, how to address it effectively.

The core issue lies in the representation of enum types in your Rails application’s `schema.rb` file versus their actual definition in your database. When you create an enum column using Rails migrations, you’re essentially instructing your database (typically postgresql, mysql or similar) to create a specific custom type to constrain the data within that column to a predefined set of strings. The migration itself executes sql directly to create that custom type.

However, `schema.rb`, which is a schema definition based on the current structure of your database, does not store the detailed *definition* of these custom types. It stores a simplified representation, often just as a string representing the enum's name and a list of allowed values. This simplification is designed to make `schema.rb` more readable and portable across different database systems, and it avoids embedding database-specific sql syntax. This is generally a good thing for database agnostic setups.

The problem comes into play when you subsequently modify the allowed values of your enum, or you modify something that forces a type regeneration, for instance, renaming the enum type itself. `db:migrate` correctly updates your database with the modified type. The issue then surfaces when you run `db:schema:dump` and it regenerates `schema.rb`. It writes the *current* state of the enum, the new values, over the old. Critically though, it doesn’t understand that the existing type needs to be dropped before recreated, and that’s where the fundamental mismatch begins. This is because a schema load uses the `schema.rb` file to create or alter database tables and types from a clean slate and doesn't attempt to diff the schema.

So, later, when you attempt `db:schema:load`, it reads `schema.rb`, which contains the modified, updated enum definition, and attempts to create it, and fails, because the named type already exists. The schema loader doesn't attempt to update the already existing custom enum type, it tries to recreate a named custom type, which it can’t. It does not compare existing types before trying to create, and instead just blindly tries to create the types.

This creates a situation where `schema.rb` is out of sync with the database's underlying structure, especially for custom types like enums.

Let’s illustrate with some examples and code. Assume we have a `status` enum field on an `orders` table, and we've been dealing with this for a while.

**Initial migration (simplified for clarity):**

```ruby
# db/migrate/20231027000000_create_orders.rb
class CreateOrders < ActiveRecord::Migration[7.0]
  def change
    create_table :orders do |t|
      t.string :status # simplified representation, imagine a custom enum type is made here under the hood
      t.timestamps
    end
    execute <<~SQL
       CREATE TYPE order_status AS ENUM ('pending', 'processing', 'shipped');
    SQL
    change_column :orders, :status, :order_status, using: 'status::order_status'
  end
end
```

This would generate an entry in `schema.rb` that, roughly, would look something like this:

```ruby
# db/schema.rb
  create_table "orders", force: :cascade do |t|
    t.string "status"
    t.datetime "created_at", null: false
    t.datetime "updated_at", null: false
  end
```

Note that the enum is represented simply as a string column. Now, let’s say we need to add a new status “cancelled”.

**Subsequent migration that modifies the enum:**

```ruby
# db/migrate/20231027000001_add_cancelled_status.rb
class AddCancelledStatus < ActiveRecord::Migration[7.0]
  def change
      execute <<~SQL
        ALTER TYPE order_status ADD VALUE 'cancelled';
      SQL
  end
end
```

After `db:migrate`, `schema.rb` is updated, and will now reflect a different column type, as Rails now knows about the `order_status` type, and will include that information. `db:schema:dump` is run behind the scenes, and it will add information for all custom types into the `schema.rb` file:

```ruby
 # db/schema.rb
  create_table "orders", force: :cascade do |t|
      t.string "status"
    t.datetime "created_at", null: false
    t.datetime "updated_at", null: false
  end
  
  create_enum "order_status", ["pending", "processing", "shipped", "cancelled"]

```
Now, if you run `db:schema:load`, it will fail, because `order_status` already exists. This is because `db:schema:load` tries to run `create_enum "order_status", ["pending", "processing", "shipped", "cancelled"]` and this instruction doesn’t account for the type already existing.

Here’s a simplified code snippet to see this in action:

```ruby
# Suppose the database already has the type order_status (pending, processing, shipped)
# this code would fail in reality, if the database has the enum.
ActiveRecord::Migration.create_enum("order_status", ["pending", "processing", "shipped", "cancelled"])
# this fails because of the type being already existing
```

**Solution:**

The most reliable way to avoid this issue is to be very careful when modifying existing enums. The most reliable way is to either (a) drop the existing type and create a new one, or (b) if supported, alter the existing type. The first method is not ideal as you lose all data. The second method is also not ideal, as it can be very cumbersome, but is more practical if you have data. If you are just starting, the simplest way is to always drop and recreate, as you won’t have data integrity issues to worry about.

Here's the safer way when you *must* change the enum values, using the drop/recreate method, keeping data integrity in mind:

```ruby
# db/migrate/20231027000002_alter_order_status_safely.rb
class AlterOrderStatusSafely < ActiveRecord::Migration[7.0]
  def change
    reversible do |dir|
      dir.up do
        #1. Create a temporary column for the new type
        add_column :orders, :temp_status, :string
        execute "ALTER TYPE order_status RENAME TO old_order_status;" #rename the old type
        
        execute <<~SQL
            CREATE TYPE order_status AS ENUM ('pending', 'processing', 'shipped', 'cancelled');
        SQL
        
        execute "UPDATE orders SET temp_status = status::text;" # Copy the existing data to the temporary column
        execute "ALTER TABLE orders ALTER COLUMN status TYPE order_status USING temp_status::order_status" # Change the old column to the new type.
        remove_column :orders, :temp_status #remove the temporary column

        execute "DROP TYPE old_order_status;" #drop the old type
      end
      dir.down do
          # This part needs more careful consideration depending on how you handle rollbacks and what data you want to preserve
          add_column :orders, :temp_status, :string # add the temporary column again
           execute "ALTER TYPE order_status RENAME TO new_order_status;"#rename the new type to temporary name
           execute <<~SQL
            CREATE TYPE order_status AS ENUM ('pending', 'processing', 'shipped');
        SQL # recreate the old type
         execute "UPDATE orders SET temp_status = status::text;" # copy data to temp column
        execute "ALTER TABLE orders ALTER COLUMN status TYPE order_status USING temp_status::order_status"
        remove_column :orders, :temp_status # remove temp column

        execute "DROP TYPE new_order_status;" #drop the renamed new type.
      end
    end
  end
end
```
This approach involves a more manual process, including temporarily storing the column as text, and is a better approach to preserving data integrity, instead of relying on just schema.rb and the default behaviors. Note that this could be drastically simplified if data integrity was not a concern.

The `reversible` block helps to handle rollbacks, although it would be prudent to be cautious and test your rollbacks extensively, since it is easy to end up with data loss in the `down` direction.

**Recommendation:**

For a deep dive into database schema management, I'd recommend reading up on relational database theory. Elmasri and Navathe's "Fundamentals of Database Systems" provides an excellent foundation on relational concepts and schema design. Also, diving into documentation of your specific database, such as PostgreSQL's documentation on `enum` types, will greatly aid your understanding. For practical Rails usage, the official Rails guides on migrations and schema management are indispensable. Be aware, that while these explain the general usage patterns of migrations, schema.rb and rails, they may not go into the details that one might encounter in real world, advanced scenarios.

To summarize, understanding how `db:migrate` interacts with custom database types like enums and the role of `schema.rb` is crucial. While seemingly straightforward, a naive approach to enum modifications can lead to schema load failures. It’s always safer to explicitly manage type changes, including dropping and recreating if you can, or alter the existing type if you must. Data loss is never acceptable in production situations, so testing, both for migrations *and rollbacks*, is always advised. This is one of those seemingly simple areas where deeper database knowledge can make a huge difference, especially under pressure.
