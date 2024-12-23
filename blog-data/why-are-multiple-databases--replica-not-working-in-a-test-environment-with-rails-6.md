---
title: "Why are multiple databases / replica not working in a test environment with Rails 6?"
date: "2024-12-23"
id: "why-are-multiple-databases--replica-not-working-in-a-test-environment-with-rails-6"
---

Alright, let's unpack this. I've definitely seen this issue crop up more than a few times over the years, particularly when teams are scaling up their Rails applications and introducing more complex database setups. The problem of multiple database replicas not functioning correctly in a test environment with Rails 6, while often seemingly baffling, typically boils down to a few common culprits related to how Rails manages database connections and configurations in different environments.

It’s not uncommon to start with a setup that works perfectly in development, hitting one central database, but then things get hairy when we try to replicate that for testing or staging – especially with replicas. In my experience, I've found the key is almost always in the fine-grained details of connection management and how Rails' test environment treats those connections.

One of the main problems I've encountered stems from the way Rails handles database configurations. In your `database.yml` file, you're probably specifying different connection details for the `development`, `test`, and potentially `production` environments. The 'test' configuration, more often than not, uses an in-memory sqlite database, or a separate test database in postgres or MySQL. However, when you're attempting to incorporate database replicas, this configuration needs more specific handling, especially if you’re aiming to mirror a production-like setup as closely as possible.

The first, and perhaps most critical mistake, I've observed teams make is to not fully specify connection roles. Rails 6, thankfully, has the concept of connection roles built in, but if your configuration isn't making use of them, you’re in for a world of pain when dealing with replicas. You need to explicitly state which connections are meant for writing, and which are read-only replicas. Without this distinction, all connections could end up trying to write, and thus, your 'replicas' won't be serving their purpose. It’s a recipe for inconsistent data, test failures, and all kinds of headaches.

Here’s a basic example of how your `database.yml` might *incorrectly* look when you're not fully leveraging connection roles, especially for replicas, and the test environment configuration. This will highlight the issue where the 'test' environment could be inadvertently pointing to the main database or is just using an in memory database, instead of a replica or multiple databases that is similar to production.

```yaml
default: &default
  adapter: postgresql
  encoding: unicode
  pool: <%= ENV.fetch("RAILS_MAX_THREADS") { 5 } %>

development:
  <<: *default
  database: my_app_development
  username: user_dev
  password: password_dev

test:
  <<: *default
  database: my_app_test
  username: user_test
  password: password_test
```

Now, consider what needs to be added to correctly implement connection roles for a primary database and a read replica. This will be particularly relevant for testing a setup that is like production. This is crucial, especially in scenarios where the test suite needs to verify read operations from the replica and write operations to the primary.

```yaml
default: &default
  adapter: postgresql
  encoding: unicode
  pool: <%= ENV.fetch("RAILS_MAX_THREADS") { 5 } %>

development:
  <<: *default
  database: my_app_development
  username: user_dev
  password: password_dev
  role: writing

test:
  primary:
    <<: *default
    database: my_app_test_primary
    username: user_test
    password: password_test
    role: writing
  replica_1:
    <<: *default
    database: my_app_test_replica_1
    username: user_test
    password: password_test
    role: reading
  replica_2:
    <<: *default
    database: my_app_test_replica_2
    username: user_test
    password: password_test
    role: reading
```

As you can see, in the second example we explicitly define roles for each connection in the test environment. `primary` is designated for writes and the two `replica` connections for reads. Without this structure, Rails doesn't know which connection to use for writing or reading, and might default to writing everywhere, which, needless to say, defeats the purpose of using read replicas.

Another aspect frequently overlooked is the interaction with Rails' transactional tests. When a test suite runs in transactional mode (the default in Rails), database operations are wrapped within a transaction that is rolled back at the end of each test. This works well for a single database, but when you’re dealing with multiple database connections, these transaction are per connection. So, a write operation to the `primary` database will *not* automatically appear on your `replica`, at least not until the transaction commits. That is if they are set up to work in an asynchronous way. This can lead to tests passing incorrectly.

Here's a simplified snippet of a test, and the issue it demonstrates:

```ruby
# Assume we have an ActiveRecord model 'User'
class User < ApplicationRecord; end

# In test/models/user_test.rb
require 'test_helper'

class UserTest < ActiveSupport::TestCase
  def setup
    ActiveRecord::Base.connected_to(role: :writing) do
      User.create!(name: "testuser") # write to the primary
    end
  end
  test "user is created and readable" do
    ActiveRecord::Base.connected_to(role: :reading) do
      user = User.find_by(name: "testuser") # try reading from replica
      assert_not_nil user # this could fail sometimes
    end
  end
end
```

In this case, the test can fail because, the data written to the primary is not immediately available on the read replica due to replication lag. In a test environment, where consistency expectations are stricter, you might need to explicitly instruct your test to wait until the data is available on the replica (or bypass the replica altogether in tests). This will vary depending on the replication method, whether that is async or sync, and which database software is being used. There are strategies to ensure the primary and secondary data is in sync within test cases.

To mitigate this, you might consider configuring your tests to bypass the read replicas or write to it as well in the test environment.

Here's a modified version where we bypass the replica, or set up the test data directly on the replica.

```ruby
# Assume we have an ActiveRecord model 'User'
class User < ApplicationRecord; end

# In test/models/user_test.rb
require 'test_helper'

class UserTest < ActiveSupport::TestCase
  def setup
    ActiveRecord::Base.connected_to(role: :writing) do
      User.create!(name: "testuser") # write to the primary
    end
    # A solution could also be to write to the replicas as well, and query from them as well.
    # ActiveRecord::Base.connected_to(role: :reading) do
    #  User.create!(name: "testuser")
    # end
  end
  test "user is created and readable" do
    ActiveRecord::Base.connected_to(role: :writing) do # Read directly from the primary in the test
      user = User.find_by(name: "testuser")
      assert_not_nil user
    end
  end
end
```

By bypassing the replica in this test, you ensure that you're querying directly from the source where the data was written, removing any inconsistencies due to replication delays or transaction rollbacks. Alternatively, you can set up the test data in the replicas as well. This method can help for some cases, but it's a different testing strategy.

In terms of resources, I'd strongly suggest diving into *'Designing Data-Intensive Applications' by Martin Kleppmann.* It offers a very thorough overview of databases, including topics like replication, consistency, and transactions, which are critical for understanding the complexities you're encountering. Furthermore, the official Rails documentation on multiple databases and connection management provides up-to-date guidance on configuration specifics and available options. Also consider reading articles and blog posts about how to effectively set up tests with replicas. This is generally a case where you have to learn from others that have tackled similar problems.

The key here is not to treat your database connections as an afterthought in a test environment but to actively manage how your tests are interacting with them. Understanding connection roles, the transactional behavior of Rails, and the specifics of your replication setup are vital. It’s a journey that requires careful setup and testing and can be quite frustrating initially, but with the right approach, it is a manageable process. I hope that these examples and recommendations provide some direction in resolving your issue.
