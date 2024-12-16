---
title: "How to combine ActiveRecord models and sort by shared properties?"
date: "2024-12-16"
id: "how-to-combine-activerecord-models-and-sort-by-shared-properties"
---

Okay, let's tackle this one. It’s a challenge I've encountered multiple times over the years, usually when dealing with complex data visualizations or consolidated reporting. The core issue, as I see it, is how to efficiently combine information from different ActiveRecord models and then sort the resulting data by a common attribute that might not exist in all models individually. It requires a bit of careful planning and, frankly, a willingness to move beyond the usual ActiveRecord query patterns.

My personal experience stems from a project where we had a system tracking both ‘events’ and ‘tasks’, each with its own specific model and set of properties. However, we needed to present a chronological timeline sorted by a combined timestamp, even though the timestamp was called something different in each table (e.g., `created_at` for events, and `due_date` for tasks). ActiveRecord's built-in methods aren't ideally suited for this kind of cross-model sorting, so we had to get creative.

Here’s the breakdown of how I typically approach this, along with some illustrative code examples.

The fundamental problem isn’t just combining the models – it's normalizing the shared attribute before we even think about sorting. This involves mapping the diverse properties onto a common representation, essentially creating a unified data structure.

Here’s the first example illustrating how to do this with Ruby:

```ruby
  def self.combined_and_sorted_timeline(limit: 20)
    events = Event.order(created_at: :desc).limit(limit)
    tasks = Task.order(due_date: :desc).limit(limit)

    combined_items = []
    events.each do |event|
      combined_items << {
        item_type: 'event',
        timestamp: event.created_at,
        item: event
      }
    end
    tasks.each do |task|
      combined_items << {
        item_type: 'task',
        timestamp: task.due_date,
        item: task
      }
    end

    combined_items.sort_by! { |item| item[:timestamp] }.reverse!.take(limit)
  end
```

This snippet does a couple of important things. First, it fetches a limited number of `Event` and `Task` records ordered by their respective time attributes (`created_at` and `due_date`). Then, it maps each record into a hash containing the `item_type`, a unified `timestamp`, and the original record. Finally, it combines the arrays, sorts them by the new timestamp, and then takes the specified limit which has been reverse sorted for the desired ordering.

Now, this solution is functional, but it can be inefficient, especially with larger datasets. We're loading all the data into memory and then sorting it, which can be a performance bottleneck.

Let's move on to an example that uses a more scalable approach, leveraging a database-level `UNION` operation. This is the second snippet:

```ruby
  def self.combined_and_sorted_timeline_sql(limit: 20)
   sql = <<-SQL
      (SELECT
        'event' as item_type,
        created_at as timestamp,
        id
      FROM events
      ORDER BY created_at DESC
      LIMIT #{limit})
      UNION ALL
      (SELECT
        'task' as item_type,
        due_date as timestamp,
        id
      FROM tasks
      ORDER BY due_date DESC
      LIMIT #{limit})
      ORDER BY timestamp DESC
      LIMIT #{limit}
    SQL
    results = ActiveRecord::Base.connection.execute(sql)

    results.map do |row|
      if row['item_type'] == 'event'
        {item_type: 'event', item: Event.find(row['id']), timestamp: row['timestamp']}
      elsif row['item_type'] == 'task'
        {item_type: 'task', item: Task.find(row['id']), timestamp: row['timestamp']}
      end
    end
  end
```

Here, we construct raw SQL to perform the `UNION`. This approach pushes the sorting and limiting down to the database, which is almost always more efficient than doing it in application code for large datasets. We create a single result set including a unified 'timestamp' and 'id', ordered appropriately, and capped by limit. Then we query individual models based on id and type before returning an array of hashes.

This approach significantly reduces the memory footprint. It can still be further optimized using database specific features if needed, but this provides a good balance between clarity and efficiency for many scenarios. The key here is understanding that offloading database operations is usually preferable.

Now, neither of these solutions is perfect for all circumstances. For truly massive datasets, or if you require further flexibility in sorting, there are more advanced techniques. Let's explore a third snippet that provides a more granular approach:

```ruby
  def self.combined_and_sorted_timeline_page(limit: 20, offset: 0, item_type_filter: nil)
     sql_subqueries = []
    sql_subqueries << "(SELECT 'event' AS item_type, created_at AS timestamp, id FROM events ORDER BY created_at DESC LIMIT #{limit})" unless item_type_filter == 'task'
    sql_subqueries << "(SELECT 'task' AS item_type, due_date AS timestamp, id FROM tasks ORDER BY due_date DESC LIMIT #{limit})" unless item_type_filter == 'event'

    union_query = sql_subqueries.join(" UNION ALL ")

    sql = <<-SQL
      SELECT *
      FROM (
        #{union_query}
      ) AS combined_results
      ORDER BY timestamp DESC
      LIMIT #{limit} OFFSET #{offset}
    SQL

    results = ActiveRecord::Base.connection.execute(sql)

    results.map do |row|
      if row['item_type'] == 'event'
        {item_type: 'event', item: Event.find(row['id']), timestamp: row['timestamp']}
      elsif row['item_type'] == 'task'
        {item_type: 'task', item: Task.find(row['id']), timestamp: row['timestamp']}
      end
    end
  end
```

This final snippet adds the possibility of filtering item types and includes a pagination scheme using offset and limits. This shows how to add greater flexibility on top of the previous approaches. It highlights how to leverage the database to its fullest extent to achieve more complex queries.

For further study in this area, I'd recommend delving into "SQL and Relational Theory" by C.J. Date. This book is a foundational text on database theory and provides a deeper understanding of how relational databases work, which is essential for writing optimal queries, especially when dealing with `UNION` and similar operations. Also, "Understanding Database Performance" by Ron Teitel should be considered as it focuses on the practical implications of database design choices and query optimization techniques.

In summary, combining ActiveRecord models and sorting by shared properties is about more than just concatenating arrays. It requires mapping diverse attributes to a common representation, making smart use of database features, and picking the right approach for the specific problem at hand. The examples provided here should serve as a solid foundation for building more robust and scalable applications. And always, keep in mind the trade-offs of each approach and choose the method that best fits your particular use case and dataset size.
