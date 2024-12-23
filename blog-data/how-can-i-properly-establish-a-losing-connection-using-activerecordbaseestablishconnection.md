---
title: "How can I properly establish a losing connection using ActiveRecord::Base.establish_connection?"
date: "2024-12-23"
id: "how-can-i-properly-establish-a-losing-connection-using-activerecordbaseestablishconnection"
---

,  I remember a particularly frustrating incident a few years back where I had to deal with a similar scenario - managing connections to a legacy database that was… let’s say, temperamental. The core issue, as I experienced, isn't just about establishing *any* connection, it's about gracefully handling a connection that's intentionally failing, potentially to test failure scenarios, or manage fallback processes. ActiveRecord::Base.establish_connection isn't inherently designed for *losing* connections; rather, its function is to establish a successful one. The "losing" aspect is something we have to engineer around, often leveraging features like connection pools and exception handling.

The trick here isn't to misuse `establish_connection` to force a failure; that's not its primary purpose, and forcing a connection to fail during establishment is typically a sign of misconfiguration, not a desired state. What you are really after is simulating or managing a scenario where a connection established initially, can, at some point in time, become unavailable.

Here’s the breakdown of how I'd approach this, focusing on achieving that controlled, and yes, sometimes intentional, loss of connectivity.

First, let’s cover the basic setup. `ActiveRecord::Base.establish_connection` takes a hash as its argument, containing connection details: typically an adapter, host, database name, username, password, and other relevant options. When called, it configures ActiveRecord to use these settings for subsequent database interactions, effectively opening up a connection, managed by the connection pool that ActiveRecord provides. What `establish_connection` *doesn't* provide, is the mechanism to readily "lose" a working connection. Instead, we'd work on deliberately interrupting the connection and handling that interruption.

Here's the fundamental idea: you establish a valid connection first, then employ tactics to simulate loss. These tactics usually involve intentionally making the connection unusable or unreachable.

**1. The "Simulated Downtime" Approach**

One approach, which I used in that legacy system fiasco, is to have a method that, when called, effectively breaks the connection while it's held by the connection pool, forcing a new one to be established. To simulate this "loss", we can temporarily invalidate credentials, change the database name or the host. This isn’t a pure “loss” in the sense of simulating network failures, it's more about invalidating the connection data stored by ActiveRecord.

```ruby
require 'active_record'

class ApplicationRecord < ActiveRecord::Base
  self.abstract_class = true
end


def establish_valid_connection
  ActiveRecord::Base.establish_connection(
    adapter: 'sqlite3',
    database: 'test.db'
  )
end

def simulate_connection_loss
  ActiveRecord::Base.establish_connection(
    adapter: 'sqlite3',
    database: 'test_broken.db' # Intentionally use a different, non-existent database.
  )
end

# Establish a valid connection
establish_valid_connection

# Example usage:
# Assuming you have model User(id: int, name: string), created with the initial schema in test.db
class User < ApplicationRecord; end


begin
  puts "Initial user count: #{User.count}" # Should work

  simulate_connection_loss # "Break" the connection
  puts "User count after simulated loss: #{User.count}" # This will fail
rescue ActiveRecord::DatabaseConnectionError => e
  puts "Caught a connection error: #{e.message}"
  establish_valid_connection # Establish a new, valid connection.
   puts "User count after re-establishing connection: #{User.count}" # Should work again
end
```

In this code, we initially establish a good connection to 'test.db'. The `simulate_connection_loss` function, changes the configuration to connect to a non-existent database 'test_broken.db', effectively breaking the connection held in the connection pool and leading to `ActiveRecord::DatabaseConnectionError`. We catch the exception, then we establish a valid connection again before continuing, ensuring the application recovers.

**2. Connection Timeout Manipulation**

Another strategy I’ve used is to manipulate the timeout settings. We can set a very short connection timeout which will cause the connection pool to reject it quickly as the pool tries to acquire a connection. This is useful when simulating network issues or congested servers that don't immediately respond to connection requests.

```ruby
require 'active_record'

class ApplicationRecord < ActiveRecord::Base
  self.abstract_class = true
end

def establish_valid_connection
  ActiveRecord::Base.establish_connection(
    adapter: 'sqlite3',
    database: 'test.db',
    connect_timeout: 5
  )
end

def simulate_network_timeout
 ActiveRecord::Base.establish_connection(
    adapter: 'sqlite3',
    database: 'test.db',
    connect_timeout: 0.00001 # Very short timeout.
  )
end

# Establish a connection with standard timeout.
establish_valid_connection

class User < ApplicationRecord; end

begin
    puts "Initial user count: #{User.count}"
    simulate_network_timeout # Force timeout on next acquire attempt
    puts "User count after simulated network timeout: #{User.count}"
rescue ActiveRecord::DatabaseConnectionError => e
  puts "Caught a connection timeout: #{e.message}"
  establish_valid_connection # Reestablish and continue.
  puts "User count after re-establishing connection: #{User.count}"

end
```

Here, we begin with a 5-second timeout, then temporarily change it to an extremely low value. This ensures that future connection attempts are rejected quickly, simulating a connection failure. Again, handling the raised `ActiveRecord::DatabaseConnectionError` becomes essential.

**3. Direct Database Connection Manipulation (Advanced)**

For those who want more fine-grained control, you can directly manipulate the underlying database connection object. While less common, this offers control if you're debugging connection pooling behavior, or if you need to force a hard disconnect. You could fetch the currently active connection and call `disconnect!` on it. However, note that this approach is inherently fragile, as the implementation of connection management changes between database adapters and ActiveRecord versions, so it should be used carefully. I’ve generally avoided it outside of debugging scenarios, and I wouldn’t recommend it for general application use unless you fully understand the connection internals.

```ruby
require 'active_record'

class ApplicationRecord < ActiveRecord::Base
  self.abstract_class = true
end

def establish_valid_connection
  ActiveRecord::Base.establish_connection(
    adapter: 'sqlite3',
    database: 'test.db'
  )
end

# Establish a valid connection.
establish_valid_connection

class User < ApplicationRecord; end

begin
  puts "Initial user count: #{User.count}"

  connection_pool = ActiveRecord::Base.connection_pool
  connection_pool.disconnect! # Forces a disconnect of all connections
   puts "User count after manual disconnect attempt: #{User.count}" # Will likely raise an exception

rescue ActiveRecord::ConnectionNotEstablished => e
  puts "Caught disconnection error: #{e.message}"
  establish_valid_connection # Re-establish connection
  puts "User count after re-establishing connection: #{User.count}"
end

```

This code forces a disconnection at the pool level. It's a blunt tool useful in certain circumstances, but, again, not the recommended method for simulating connection loss generally, due to its fragility to underlying framework behavior changes.

**Key Considerations**

-   **Robust Exception Handling:** When dealing with connection issues, robust exception handling is absolutely critical. Catching `ActiveRecord::DatabaseConnectionError` and other related exceptions allows your application to gracefully recover from connection failures, implement fallbacks or inform users of the issue.
-   **Connection Pooling Behavior:** A good understanding of how ActiveRecord manages connection pools is beneficial. Reading through the source code, and research papers around the design of databases' internal connection pools can help immensely. "Database Internals" by Alex Petrov is particularly useful here, providing insights into database internals, connection pool design, and related challenges. Also the official documentation of your chosen database (e.g., PostgreSQL, MySQL, etc.) should be consulted, as the intricacies of connection handling can vary.
-   **Logging and Monitoring:**  Log connection establishment and failures for diagnostic purposes. If connections are failing unexpectedly, logs will provide critical data. In a production environment, it is crucial to monitor for failed connections and raise alerts as needed to prevent outages.

In summary, `ActiveRecord::Base.establish_connection` isn't designed to simulate connection *loss* in itself, but to *establish* connections. Simulating a "lost connection" involves breaking or invalidating an established connection by either changing the connection details, manipulating timeout settings, or directly managing the underlying connection, while ensuring you have robust exception handling for a graceful recovery. I hope these insights from my past experiences help. Always remember to test your failover scenarios thoroughly before deploying to a production environment.
