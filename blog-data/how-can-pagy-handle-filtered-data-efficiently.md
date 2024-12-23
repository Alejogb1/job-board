---
title: "How can Pagy handle filtered data efficiently?"
date: "2024-12-23"
id: "how-can-pagy-handle-filtered-data-efficiently"
---

Alright, let’s delve into this. I’ve seen this exact scenario pop up more times than I can count, particularly when building data-heavy applications. The challenge of efficiently handling filtered data with pagination is a common one, and *Pagy*, as a gem, provides some robust tools to address it. It’s not just about slapping a pagination bar on a page; it's about ensuring the queries remain performant even when filters get complex. My experience with this has led me to a few go-to strategies, and I'll walk you through them, showing you some code snippets along the way.

The core problem here is that without careful handling, each page request could trigger a database query that re-applies all the filters *again*. This is inefficient and can quickly bog down an application when dealing with large datasets. The trick is to leverage the capabilities of your database system, specifically with techniques like scope chaining in rails, to effectively manage the filtering *before* paginating.

Let’s start with an example. Assume we have a model called `Product` and users can filter these products by `category` and `price_range`. Without any optimization, a naive implementation might look like this within a controller action, which is really where the majority of these issues appear.

```ruby
def index
  @products = Product.all

  if params[:category].present?
    @products = @products.where(category: params[:category])
  end

  if params[:min_price].present? && params[:max_price].present?
    @products = @products.where(price: params[:min_price]..params[:max_price])
  end

  @pagy, @products = pagy(@products)
end
```

This looks straightforward, and it functions. However, as the filter logic becomes more intricate, you'll realize that each time we apply a new filter condition, it’s essentially constructing a new query. This can be problematic as the conditions can overlap, and the query plan could become less optimized. A better approach is to encapsulate the filtering logic within the model itself using scopes. This improves readability and also enhances the reusability of our filtering logic. It's a far more efficient and maintainable structure.

Here's how we could refactor the `Product` model:

```ruby
# app/models/product.rb
class Product < ApplicationRecord
  scope :by_category, -> (category) { where(category: category) if category.present? }
  scope :by_price_range, -> (min_price, max_price) { where(price: min_price..max_price) if min_price.present? && max_price.present? }
end
```

And here’s how the controller action would now look:

```ruby
def index
  @products = Product.all.by_category(params[:category]).by_price_range(params[:min_price], params[:max_price])
  @pagy, @products = pagy(@products)
end
```

Notice how the controller action is considerably cleaner and focuses solely on data retrieval and pagy integration. The filtering responsibilities reside in the `Product` model where they belong. This chaining approach not only improves readability and organization but also takes advantage of ActiveRecord’s lazy evaluation. Queries are constructed efficiently and are only executed when `pagy()` method initiates the database retrieval. This approach enhances performance, as only the necessary conditions are used to build the specific query, and it's not re-applying or recreating the conditions each time a page is requested.

The *next level* of efficiency comes when integrating this with indexed database columns. If the columns used for filtering (`category`, `price`) are appropriately indexed, queries become much faster. If you are not familiar with database indexing, "Database Internals: A Deep Dive Into How Databases Work" by Alex Petrov is an excellent reference for understanding these concepts.

Now, there are cases where filters aren’t just simple equality conditions or ranges. You might need more elaborate filtering such as searching across multiple fields or doing full-text searches. In such scenarios, it's crucial to leverage database-specific search features. Here's an example using a simple text search using a scope:

```ruby
# app/models/product.rb
class Product < ApplicationRecord
    scope :search, -> (query) {
      if query.present?
        where("name LIKE ? OR description LIKE ?", "%#{query}%", "%#{query}%")
      end
    }
  # existing scopes are still here
  scope :by_category, -> (category) { where(category: category) if category.present? }
  scope :by_price_range, -> (min_price, max_price) { where(price: min_price..max_price) if min_price.present? && max_price.present? }
end

```
And our controller action:

```ruby
def index
  @products = Product.all.by_category(params[:category]).by_price_range(params[:min_price], params[:max_price]).search(params[:query])
  @pagy, @products = pagy(@products)
end
```

This particular example utilizes simple `LIKE` queries. In a real-world application, I’d strongly advise to use database features like full-text indexes (such as PostgreSQL’s full text search capabilities) for such requirements. Books like “PostgreSQL Query Optimization” by Deborah L. Wilson provide thorough understanding of such techniques.

Another point that's frequently missed is the count optimization provided by *Pagy*. When `pagy` is called, it performs two database queries: one to retrieve the paginated records and another to determine the total record count for rendering pagination links. Sometimes this count query might have performance overhead particularly if your filtering involves subqueries. *Pagy* has an optional count parameter to address that issue when the total count of records is available via other means, such as with `count()` on a cached association if your conditions permit.

Let’s consider a scenario where we are applying several filters on an associated model. Retrieving the total count by simply running `count` on the resulting scope could be slow. If we assume we are filtering `Order` records which have `line_items`, and we are filtering by a product attribute on the `line_items`:

```ruby
# app/models/order.rb
class Order < ApplicationRecord
  has_many :line_items
  scope :with_product_name, -> (query) {
    joins(:line_items).where('line_items.product_name like ?', "%#{query}%") if query.present?
  }
  #other scopes as well
end
```

And our controller action:

```ruby
def index
  filtered_orders = Order.all.with_product_name(params[:product_name]).where(order_date: 1.year.ago..Date.today)
  @pagy, @orders = pagy(filtered_orders)
end
```

In this scenario, `Pagy` would trigger a separate count query with the applied `JOIN` and `where` conditions again. That could be problematic for performance if that scope is significantly slower than simply fetching the IDs.

Here’s a better approach if your situation allows it, where you are able to retrieve the total count via a more efficient manner like a `count` directly on the relation before using the conditions, using the `count` parameter. I have used variations of this successfully in multiple projects:

```ruby
def index
  filtered_orders = Order.all.with_product_name(params[:product_name]).where(order_date: 1.year.ago..Date.today)
    count = filtered_orders.count(:all) # efficient way to get the count
    @pagy, @orders = pagy(filtered_orders, count: count)
end
```

Here, the `count: count` tells *Pagy* to use our precalculated count instead of performing a second query. This optimization is particularly crucial when dealing with complex joins or subqueries within the filter.

In summary, handling filtered data efficiently with *Pagy* involves more than just paginating an ActiveRecord collection. The key is leveraging database features and optimizing your query construction, especially by using scopes in your models. Remember the importance of database indexing and using the optional `count` argument of the `pagy` method to reduce the number of queries when possible. By combining these techniques, you can ensure a smooth user experience with performant pagination even under complex filtering scenarios. And that, fundamentally, is what keeps any application humming.
