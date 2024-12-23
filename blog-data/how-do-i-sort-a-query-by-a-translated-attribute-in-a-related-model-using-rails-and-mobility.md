---
title: "How do I sort a query by a translated attribute in a related model using Rails and Mobility?"
date: "2024-12-23"
id: "how-do-i-sort-a-query-by-a-translated-attribute-in-a-related-model-using-rails-and-mobility"
---

Alright,  I remember dealing with a similar issue back when I was optimizing a multilingual e-commerce platform – sorting products by their translated names was a performance bottleneck. It turned out to be less straightforward than it initially seemed, especially when dealing with complex relational structures in Rails and the intricacies of the `mobility` gem.

The crux of the matter is that when you're dealing with translated attributes, like product names in different languages, you can't directly apply standard SQL sorting. The database sees those translations as separate entries in a join table (or similar, depending on your `mobility` backend), not as a single, easily sortable field. So, we need a way to tell the database, "look at the appropriate translation for each record in the primary table and then sort based on those values."

Here's a technical walkthrough of how I'd approach this problem, along with code examples that you can adapt:

**The Challenge: Bridging Translation and Database Sorting**

The core problem arises from the fact that the translation lives outside the primary model table. Imagine you have a `Product` model, and its `name` attribute is translated using `mobility`. This typically creates another table, let's say `product_translations`, which maps product ids to translations for different locales. When you execute a basic `Product.order(:name)` query, the database looks for a `name` column directly in the `products` table; it doesn't implicitly know how to look up the translated names in another table. We must craft a SQL query that does the join *and* selects the relevant translated name based on the current locale.

**Solution: Joining and Conditional Ordering**

The most efficient method I’ve found involves utilizing `joins`, `where`, and conditional `order` statements within ActiveRecord. We effectively guide the SQL to retrieve the appropriate translated value before applying the sort. `Mobility` itself doesn't handle the sorting part directly; it provides the infrastructure for managing translations, but the query logic is our responsibility.

**Code Snippets: Example Implementations**

Let's work with a simplified model, where `Product` has a translated `name` attribute, and we aim to sort a set of products by their names in the currently set locale. The models are defined as such:

```ruby
class Product < ApplicationRecord
  extend Mobility
  translates :name
  has_many :product_categories
  has_many :categories, through: :product_categories
end

class Category < ApplicationRecord
  has_many :product_categories
  has_many :products, through: :product_categories
end

class ProductCategory < ApplicationRecord
  belongs_to :product
  belongs_to :category
end

# In your migration (example)
# t.string :name # in products table
# t.string :name # in product_translations table
```

*   **Snippet 1: Sorting by translated product name**

    This demonstrates how we can sort products by their translated names in the currently set locale (using the assumption that `I18n.locale` is already set).

    ```ruby
    def sorted_products_by_name
      Product.joins(:translations)
            .where(mobility_translations: { locale: I18n.locale.to_s, translatable_attribute: 'name'})
            .order('mobility_translations.value')
    end

    # Example usage
    I18n.locale = :en
    english_products = sorted_products_by_name
    puts "Products sorted by english name: #{english_products.pluck(:name)}"
    I18n.locale = :fr
    french_products = sorted_products_by_name
    puts "Products sorted by french name: #{french_products.pluck(:name)}"

    ```

    Here, we're leveraging `mobility_translations` table, which is how `Mobility` typically stores its translations. Note that I'm using `mobility_translations.value` to order since that field generally contains the actual translated string. The `where` clause ensures that only translations corresponding to the current `I18n.locale` and the attribute `name` are being considered.

*   **Snippet 2: Sorting by translated category name**

    Now, let’s suppose you want to sort *categories*, and you'd like to order them based on their translated names.

    ```ruby
    def sorted_categories_by_name
     Category.joins(:translations)
             .where(mobility_translations: { locale: I18n.locale.to_s, translatable_attribute: 'name'})
             .order('mobility_translations.value')
    end

     # Example usage
    I18n.locale = :en
    english_categories = sorted_categories_by_name
    puts "Categories sorted by english name: #{english_categories.pluck(:name)}"

    I18n.locale = :fr
    french_categories = sorted_categories_by_name
    puts "Categories sorted by french name: #{french_categories.pluck(:name)}"
    ```
    This is very similar to the prior example, but we apply it to the `Category` model, which also has translations. This demonstrates the consistency and reusability of the approach.

*   **Snippet 3: Sorting products by translated category name**

    This last example is slightly more complex, and illustrates how we can use it to sort products by their category names. This demonstrates the power of relational sorting with translations.

    ```ruby
    def sorted_products_by_translated_category_name
        Product.joins(product_categories: {category: :translations})
            .where(mobility_translations: { locale: I18n.locale.to_s, translatable_attribute: 'name'})
            .order('mobility_translations.value')
    end

    # Example usage
    I18n.locale = :en
    english_products = sorted_products_by_translated_category_name
    puts "Products sorted by english category name: #{english_products.map{|p| p.categories.first.name}}"
    I18n.locale = :fr
    french_products = sorted_products_by_translated_category_name
    puts "Products sorted by french category name: #{french_products.map{|p| p.categories.first.name}}"
    ```

     This snippet builds upon the previous ones but adds a join through the intermediate `product_categories` table. This is where the query complexity increases. We start with products, join product categories, then join the category translations table. We filter for the locale as before and then sort based on the translated value of the category.
**Important Considerations and Further Reading**

*   **Performance:** While the above techniques work, these queries can become slow as the scale increases. Caching strategies, indexing the `mobility_translations` table on `locale` and `translatable_attribute`, or even leveraging materialized views in PostgreSQL, can be useful tools for optimization.
*   **Different Mobility Backends:** The query syntax might vary slightly depending on which backend you're using with `mobility` (e.g., key-value store, table). Adapt the joins and where clauses accordingly to target the correct storage mechanism.
*   **Null Handling:** Consider edge cases, especially when there are missing translations. You might need to introduce specific null sorting conditions to handle those scenarios gracefully, to ensure that missing translations don’t impact the overall sort order.
*   **Complex Ordering:** For cases requiring multi-level ordering, you can expand the `order` clause. Also, consider the performance when complex `order by` clauses are introduced.

**Recommendations for Further Study:**

*   **"SQL Performance Explained" by Markus Winand:** This book offers a deep dive into database performance tuning, covering indexing, query plan analysis, and other essential aspects for optimization. Understanding how the database processes your SQL queries will help in writing optimized sorting logic.
*   **"Active Record Query Interface" section in the Rails Guides:** Familiarize yourself with the intricacies of building efficient queries with ActiveRecord. This will help you adapt and expand on the basic examples shown above.
*   **The documentation for the `mobility` gem itself:** The `mobility` gem's documentation is crucial for understanding how translations are stored and accessed using your selected backend.

In my experience, it's important to start with clear understanding how joins and where clauses work, then test each approach carefully in a staging environment before applying it to production. These techniques should provide a solid starting point for tackling such complex scenarios. Let me know if you have any follow-up questions!
