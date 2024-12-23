---
title: "Why is sorting failing on a non-database column in Rails 6.1.4?"
date: "2024-12-23"
id: "why-is-sorting-failing-on-a-non-database-column-in-rails-614"
---

Alright, let's unpack this sorting issue on a non-database column in Rails 6.1.4. I've seen this particular problem crop up enough times across various projects to have a pretty solid grasp on its typical causes and solutions. It often stems from a fundamental misunderstanding of how ActiveRecord interacts with data and how sorting is generally handled, especially when custom, non-database attributes get involved. This issue isn’t necessarily a bug in Rails, but rather a consequence of how we need to approach sorting based on properties that don’t directly exist as database columns.

The core problem lies in ActiveRecord's `order` method. This method is designed to translate SQL `ORDER BY` clauses, working directly with columns present in the database schema. If you try to use `order` with an attribute that isn't mapped to a database column, it won’t know how to construct the corresponding SQL query and thus, it's unable to achieve the sort on the specified attribute. Essentially, you are asking the database to sort on something it doesn’t understand.

Let’s say, for instance, you have a `Product` model with a non-database attribute calculated dynamically, perhaps based on some business logic or data retrieval. Something like an `average_rating` attribute determined by related `Review` records. Let’s look at some code snippets to put this into perspective.

**Scenario 1: Direct Attempt to use `order` on a Non-Database Attribute (Fails)**

```ruby
# app/models/product.rb
class Product < ApplicationRecord
  has_many :reviews

  def average_rating
    return 0 if reviews.empty?
    reviews.average(:rating).to_f
  end
end

# products_controller.rb
def index
    @products = Product.order(average_rating: :desc) # This will raise an error
end
```
This will raise a `ActiveRecord::StatementInvalid` error or some variation thereof. Rails doesn't know how to generate the SQL to sort by `average_rating`, since it’s not a column on the `products` table. The error messaging will often hint towards this by indicating the column does not exist in the table being queried.

**Scenario 2: Sorting in Ruby After Fetching Records**

This is the approach that resolves the situation, where we load all the records and sort in memory.
```ruby
# products_controller.rb
def index
  @products = Product.all.sort_by { |product| -product.average_rating } # sorting in ruby
end
```
This will correctly sort the products by `average_rating` in descending order. Here’s the key: we’re first fetching *all* records from the database using `Product.all`, and then we are sorting them within ruby. Note the use of the unary minus sign (`-`) in front of `product.average_rating` for descending order sorting.

While this approach will work, it's not generally advisable for larger datasets due to the significant performance implications of pulling all records into memory before sorting. For a few thousand products, it may be fine. However, for datasets with tens of thousands, hundreds of thousands or millions of records it will be detrimental. You'll find your app slowing down considerably.

**Scenario 3: Using a Join and a database computed value for sorting**

This is a more complicated approach. It requires that you create a query that creates a sortable value in the query using SQL. In this case we will use the database `AVG()` function to provide a value for sorting.
```ruby
# products_controller.rb
def index
  @products = Product.joins(:reviews)
                      .group("products.id")
                      .select("products.*, AVG(reviews.rating) as average_rating")
                      .order("average_rating DESC")

end
```
Here, we use `joins` to establish the relationship to the `reviews` table, `group` to ensure our average calculations are correct for each product (assuming each product can have multiple reviews), we `select` all the `products` columns and a computed average rating value using `AVG()`, and finally sort by the derived average rating column `average_rating`. We’ve moved the sorting from Ruby-level code into the SQL statement, which is significantly more efficient for larger tables. This pushes the sort to the database engine.

This query is not trivial, but is extremely powerful for performance. The downside is the increased complexity. It assumes the table relationships are setup correctly.

**Key Considerations and Recommendations**

1.  **Performance:** As demonstrated, sorting large datasets in memory is inefficient. If you expect to have many records and your sort logic is relatively simple, try to perform the sorting within the database engine using SQL queries.

2.  **Complexity vs. Efficiency:** Weigh the performance gains of more complex SQL queries against the increase in code complexity and maintainability. It's not always necessary to use complex sql for small datasets. Use the solution that best balances your teams knowledge and the need for performance.

3.  **Materialized Views:** For cases where the calculation of a sortable attribute is computationally expensive or involves complex relationships, consider using materialized views. These are essentially pre-calculated tables that can be indexed and queried efficiently, thus providing performance improvements. This can be an advantage when the database computations become too complex to efficiently create via direct joins.

4.  **Query Objects and Scopes:** For complex sorting logic that you might need in several different places, it’s better to encapsulate the logic into query objects or model scopes. This will help with maintainability and reduce duplication.

5.  **Data Structure:** Always consider the data structure itself. Could it be designed differently to make sorting easier? Maybe a denormalized view is an option to speed up your queries.

**Further Reading**

For deeper understanding, I recommend delving into some key resources:

*   **"Database Internals" by Alex Petrov:** While this book provides a deep dive into database internals, it is invaluable for understanding how database engines perform operations, including sorting. This background knowledge helps in creating efficient queries.
*   **"SQL and Relational Theory: How to Write Accurate SQL Code" by C.J. Date:** Date's book offers a solid theoretical foundation for working with SQL. A good grasp of relational theory helps with developing efficient SQL queries, and designing appropriate database structures.
*   **The Rails Guide on Active Record Querying:** A thorough review of the official Rails documentation is fundamental. The “Active Record Querying” section specifically delves into topics like joins, scopes, and efficient query construction.
*   **Database Performance Books by Jonathan Lewis and Cary Millsap** These two authors are well known as performance experts for Oracle. Even though their books do not directly relate to ActiveRecord, their principles are widely applicable to all databases and will help in better database usage.

In essence, the challenge you're encountering with sorting on a non-database column in Rails 6.1.4 isn’t unique, and with some careful planning and query construction, it can be solved efficiently. Remember, always strive to perform your sorts at the database level when dealing with substantial datasets, and keep your queries as optimized and readable as possible. Good luck!
