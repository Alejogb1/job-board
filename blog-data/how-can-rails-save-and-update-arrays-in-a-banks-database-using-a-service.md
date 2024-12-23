---
title: "How can Rails save and update arrays in a bank's database using a service?"
date: "2024-12-23"
id: "how-can-rails-save-and-update-arrays-in-a-banks-database-using-a-service"
---

Okay, let’s tackle this. I've seen this particular challenge crop up in a few contexts, most notably during my time working on a legacy financial system, ironically not *too* long ago. We had to migrate from a clunky, pre-relational database to a Postgres system using Rails, and handling arrays efficiently was key. It's not as straightforward as simple scalar data, but with the right approach, it becomes manageable.

The core problem revolves around how Rails, with its ActiveRecord ORM, interacts with database fields that hold arrays. Postgres, thankfully, handles arrays natively which allows us to go beyond simple single-value columns. Using a service layer on top of this isn't just good practice; it's often essential for managing the complexity of data manipulation and business logic, especially when dealing with sensitive financial information.

The basic challenge here is that Rails, out of the box, doesn't intuitively know how to handle an array of values within a single column. You might try assigning an array to an attribute directly, but you’ll quickly discover that ActiveRecord treats this as a serialized object rather than individual elements within a database-level array. This isn't efficient for searching, filtering or updates. So, how do we bridge this gap?

First, we need to define our models correctly and leverage postgres array types. Consider a scenario where we're storing customer account transaction identifiers. I’ve named it ‘transaction_ids’. The following is how the migration and model should ideally look:

```ruby
# migration
class CreateCustomerAccounts < ActiveRecord::Migration[7.0]
  def change
    create_table :customer_accounts do |t|
      t.string :customer_name
      t.integer :account_number
      t.integer :transaction_ids, array: true, default: []
      t.timestamps
    end
  end
end

# model
class CustomerAccount < ApplicationRecord
  validates :customer_name, presence: true
  validates :account_number, presence: true, uniqueness: true
end
```
Notice the crucial part: `t.integer :transaction_ids, array: true, default: []`. This declares a column named `transaction_ids` that is a PostgreSQL array of integers. The `default: []` part sets the initial value to an empty array if no value is provided on record creation, preventing null value related issues. With this setup, you can store and manipulate transaction IDs directly as a database array.

Now, let’s introduce the service layer. Here's a sample service that deals with both saving and updating transactions:

```ruby
# service
class CustomerAccountService
  def self.add_transaction(account_number, transaction_id)
    account = CustomerAccount.find_by(account_number: account_number)
    raise ArgumentError, "Account not found" unless account

    account.transaction_ids << transaction_id
    account.save!

    account # returns the updated record
  end

  def self.update_transactions(account_number, new_transaction_ids)
    account = CustomerAccount.find_by(account_number: account_number)
    raise ArgumentError, "Account not found" unless account

    account.transaction_ids = new_transaction_ids
    account.save!

    account # returns the updated record
  end

   def self.clear_transactions(account_number)
    account = CustomerAccount.find_by(account_number: account_number)
    raise ArgumentError, "Account not found" unless account
    
    account.transaction_ids = []
    account.save!
    account # returns the updated record

  end

end
```
In this service, the `add_transaction` method appends a transaction ID to the existing array. The important thing to note here is the use of `<<` which effectively modifies the existing array in memory and then is saved by active record. The `update_transactions` method replaces the existing array with a new array. Using `save!` will ensure data consistency, and will raise an exception if the update fails due to any validations set in the model. I've added the `clear_transactions` method as well, as that also often is required.

Here's an example of how you would use this service in a controller:

```ruby
#controller
class CustomerAccountsController < ApplicationController

  def add_transaction
      begin
        @account = CustomerAccountService.add_transaction(params[:account_number], params[:transaction_id].to_i)
        render json: @account, status: :ok
      rescue ArgumentError => e
        render json: { error: e.message }, status: :not_found
      rescue ActiveRecord::RecordInvalid => e
        render json: { errors: e.record.errors }, status: :unprocessable_entity
      end
  end

  def update_transactions
    begin
       new_transaction_ids = params[:transaction_ids].map(&:to_i)
      @account = CustomerAccountService.update_transactions(params[:account_number],new_transaction_ids )
      render json: @account, status: :ok
    rescue ArgumentError => e
      render json: { error: e.message }, status: :not_found
    rescue ActiveRecord::RecordInvalid => e
      render json: { errors: e.record.errors }, status: :unprocessable_entity
    end
  end

  def clear_transactions
    begin
      @account = CustomerAccountService.clear_transactions(params[:account_number])
      render json: @account, status: :ok
    rescue ArgumentError => e
      render json: { error: e.message }, status: :not_found
    rescue ActiveRecord::RecordInvalid => e
        render json: { errors: e.record.errors }, status: :unprocessable_entity
    end
  end

end

```
The controller here handles the API requests, calls the corresponding service layer, and returns the output to the client. This layer of separation is essential to ensure clean code architecture and allow for easier testing. The controller actions are concise and focused on request handling and the service encapsulates all the database interactions and array manipulations. The `rescue` blocks are crucial for exception handling, and ensure the api provides appropriate responses to the caller, should anything go wrong.

Remember, this approach leverages postgres's native support for arrays. This allows for a good degree of efficiency, and when indexed properly, makes searching within these arrays feasible. It's critical, therefore, to always check how well a query is performing, especially as array sizes grow. Tools like `EXPLAIN ANALYZE` in psql can be your friend here.

To further deepen your understanding, consider diving into the Postgres documentation, particularly sections on array types and index strategies. For Ruby on Rails specific considerations, Michael Hartl’s *Ruby on Rails Tutorial* covers the ORM deeply and would be a good general starting point. Further, the book *Understanding Databases* by Markus Winand can help in optimizing your database queries in a wider context, even though it’s not Rails specific. Finally, if you want a deeper understanding of database performance, *Database Internals* by Alex Petrov is excellent resource, although fairly advanced.

This isn't the only approach; using a relational table for a many-to-many relationship can be an alternative, but that often involves extra JOIN operations which can be more costly. Using database arrays can provide performance gains in many use cases, but it should be evaluated on a case-by-case basis based on data access patterns and data volume. In my experience, array columns have proven to be quite effective for storing things like associated IDs, as long as they don't grow to unmanageable sizes.

In conclusion, with the right models, service layer and awareness of database specific optimizations, we can manipulate database arrays with Rails efficiently, allowing for complex data storage within the relational model, while ensuring maintainable code.
