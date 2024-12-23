---
title: "How can I combine ActiveRecord models and sort by shared properties?"
date: "2024-12-23"
id: "how-can-i-combine-activerecord-models-and-sort-by-shared-properties"
---

Alright,  It's a scenario I've bumped into more times than I'd like to recall, particularly when dealing with complex data structures where multiple models share common attributes. The core challenge is efficient sorting, and ActiveRecord, while powerful, doesn't natively provide this out of the box, at least not in the way one might initially expect.

The issue stems from the fundamental structure of a relational database and the way ActiveRecord abstracts interactions with it. When you’re working with disparate models, each model often exists within its own table, with its own unique set of columns (and sometimes, shared ones). ActiveRecord, operating primarily on a per-table basis, doesn’t inherently understand how to perform sorting on attributes that exist across multiple different tables. We have to facilitate that ourselves.

To achieve the desired sorting across different ActiveRecord models based on shared properties, we essentially need to bring those disparate datasets together into a form that can be sorted by a single attribute. There are several strategies we can deploy here, and the optimal choice often depends on factors like the specific database (postgres, mysql, sqlite, etc.), the volume of data, the performance requirements, and the complexity of the data relationships. Let's explore a few of them with code examples.

**Strategy 1: Combining Results into a Single Array and Sorting in Memory**

This method is straightforward and involves fetching records from multiple models, combining them into a single array, and then sorting the array in memory using Ruby's sorting capabilities. It’s often the simplest approach initially but can become less efficient when dealing with large datasets.

Here's the conceptual code, assuming you have two models, `Article` and `BlogPost`, both possessing a `created_at` attribute:

```ruby
def combined_and_sort_in_memory
  articles = Article.all.to_a
  blog_posts = BlogPost.all.to_a

  combined = articles + blog_posts

  sorted_combined = combined.sort_by(&:created_at).reverse # Sort by date, descending

  sorted_combined
end
```
*Explanation:*

*   We start by retrieving all the records from each model and convert the ActiveRecord relations to plain arrays using `.to_a`.
*   The two arrays are merged into a single array, `combined`.
*   We sort the combined array based on the `created_at` attribute of each object in descending order. Notice the use of the `&:` operator here, which provides a concise shorthand for calling methods on elements within an array.
*   The sorted result is then returned.

This approach works nicely with small or moderate datasets, but if you're dealing with thousands or millions of records, this will strain server memory, and become inefficient, especially if pagination is needed. Performance will quickly deteriorate as your dataset increases.

**Strategy 2: Using SQL UNION and Active Record's `from` Method**

A more performant approach, especially beneficial for large datasets, is to leverage the database's capabilities using a SQL `UNION` operation. This allows the database to do the heavy lifting, specifically the sorting, before the data is even returned to Rails. We can use `ActiveRecord::Base.from` to accomplish this.
This strategy pushes the sorting workload to the database server itself, which is usually optimized for these operations.

```ruby
def sort_with_sql_union
    article_sql = Article.select('id, title, created_at, "Article" as type').to_sql
    blog_post_sql = BlogPost.select('id, title, created_at, "BlogPost" as type').to_sql

    union_sql = "(#{article_sql}) UNION ALL (#{blog_post_sql})"

    CombinedResult = Struct.new(:id, :title, :created_at, :type)

    ActiveRecord::Base.connection.exec_query("SELECT id, title, created_at, type FROM #{union_sql} ORDER BY created_at DESC").map { |row| CombinedResult.new(*row.values) }
end

```

*Explanation:*

*   We generate custom sql queries to fetch only necessary columns from each model, also adding a `type` column so we can distinguish between Article and Blogpost instances.
*   The two queries are combined using SQL's `UNION ALL`, which appends the results of the second query to the first. It is also important to note that we explicitly select the same columns from both tables and use `UNION ALL` rather than `UNION` as `UNION` will remove duplicate records.
*   We use `ActiveRecord::Base.connection.exec_query` to execute the final SQL query. Note that this returns a result set, but not a standard ActiveRecord Relation. We also wrap the results into a custom Struct, in this case, `CombinedResult` for further manipulation.

This approach is considerably faster than the in-memory sort, particularly for larger datasets, as the database server is generally much more efficient at sorting than Rails. It’s important to note that we need to manually define a struct to hold the results which is usually the price we pay for raw SQL queries and gives us great flexibility.

**Strategy 3: Polymorphic Associations and a Common Table for Shared Data**

For scenarios where shared properties are very frequently used and have a more complex relationship, a more elegant long term approach might be to create a common table that stores shared data along with a polymorphic association. This approach is best when the application design allows us to be more structured.

Let's illustrate with the creation of a `Content` model to hold shared attributes and polymorphic associations:

```ruby
class Content < ApplicationRecord
    belongs_to :contentable, polymorphic: true
end

class Article < ApplicationRecord
    has_one :content, as: :contentable, dependent: :destroy
    accepts_nested_attributes_for :content
end

class BlogPost < ApplicationRecord
   has_one :content, as: :contentable, dependent: :destroy
   accepts_nested_attributes_for :content
end
```

*Explanation:*

*   We define the `Content` model, which contains shared properties like `created_at`, and a polymorphic association via `contentable`.
*   Both the `Article` and `BlogPost` models now have a `content` association.
*   The `accepts_nested_attributes_for :content` enables us to create new `Content` entries through the respective models.

This facilitates more structured data storage as well as simplified sorting, albeit at the cost of introducing more tables to manage:
```ruby
def sort_with_polymorphic_association
    Content.order(created_at: :desc).includes(:contentable)
end
```
*Explanation:*

*   We leverage the now centralized `Content` model to sort by the `created_at` column directly
*   We include the `contentable` association so we can access the associated model (article or blog post).

This approach is more suitable when dealing with shared columns such as `created_at`, `updated_at`, and other general attributes which are used by various models as this consolidates common data and sorting logic to a central place in the application.

**Additional Considerations**

When making your decision on the approach you will be using, consider the following:

1.  **Database Indexes:** Ensure that your sorting columns are indexed in your database tables. Indexes can dramatically improve query performance for sorting and filtering operations, especially on large tables.

2.  **Pagination:** When using large datasets, always implement pagination to fetch the results in batches. It reduces server load and avoids memory issues.

3.  **Caching:** If the dataset doesn't change frequently, consider caching the results, especially for sorted data. This can further improve performance and responsiveness.

**Recommended Resources:**

*   **“SQL and Relational Theory” by C.J. Date**: Provides a deep understanding of relational database concepts, crucial for efficient SQL usage.
*   **“Database System Concepts” by Abraham Silberschatz, Henry F. Korth, and S. Sudarshan:** A comprehensive resource that covers the underpinnings of database management systems.
*  **The Official ActiveRecord documentation**: The official documentation for ActiveRecord offers insights into its inner workings, which will prove useful when you are trying to optimise operations involving this library.

In my experience, a combination of these techniques is often necessary for creating highly efficient and scalable applications. The "correct" approach depends on the particular circumstances and you will need to weigh the pros and cons of each approach in your specific application. Always begin with the most straightforward method, which would likely be combining results and sorting in memory, and then progressively enhance it if and when performance demands it.
