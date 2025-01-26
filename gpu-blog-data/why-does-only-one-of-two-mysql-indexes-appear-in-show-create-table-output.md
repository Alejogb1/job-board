---
title: "Why does only one of two MySQL indexes appear in SHOW CREATE TABLE output?"
date: "2025-01-26"
id: "why-does-only-one-of-two-mysql-indexes-appear-in-show-create-table-output"
---

In my experience managing large MySQL databases, I've frequently encountered situations where only one of two seemingly viable indexes appears in the `SHOW CREATE TABLE` output. This often leads to confusion, particularly when both indexes appear to cover similar or identical sets of columns. The root cause is not the absence of the index itself, but how MySQL handles redundant or partially redundant indexes to optimize query performance and minimize storage overhead.

Specifically, the `SHOW CREATE TABLE` statement reveals the *defined* indexes of a table, not necessarily all the indexes actually *used* by the query optimizer. MySQL might internally consider additional indexes during query execution planning, but these do not always become explicitly visible in this output.

The primary reason for this disparity relates to MySQL's implicit index prefixing behavior and its optimization strategies. When an index's columns are a prefix of another index, it’s often considered redundant. The optimizer can often use the larger index in place of the smaller one with no significant performance penalty, effectively making the smaller index redundant from a query processing perspective. In many cases, MySQL will not reveal the smaller redundant index because it's not the primary index being leveraged by the optimizer. The decision to omit an index from the `SHOW CREATE TABLE` output, even when it exists, comes down to redundancy and the primary index used for the CREATE statement's output.

Let's examine a scenario with a table named `products`:

```sql
CREATE TABLE products (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    category_id INT NOT NULL,
    supplier_id INT NOT NULL,
    price DECIMAL(10, 2) NOT NULL,
    INDEX idx_category (category_id),
    INDEX idx_category_supplier (category_id, supplier_id)
);

```

In this table, `idx_category` indexes only the `category_id` column, while `idx_category_supplier` indexes `category_id` and `supplier_id`. If I were to run `SHOW CREATE TABLE products;`, I would most likely see `idx_category_supplier` but *not* `idx_category`. This is because any query using `idx_category` can effectively leverage `idx_category_supplier` as the leading column in the latter index matches the first column in the smaller index. The query optimizer does not need to maintain the redundant `idx_category`.

Here's a second example demonstrating this further.

```sql
CREATE TABLE users (
    user_id INT AUTO_INCREMENT PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    city VARCHAR(100),
    INDEX idx_email (email),
    INDEX idx_email_first_last (email, first_name, last_name)
);
```

Similar to the first example, running `SHOW CREATE TABLE users;` will likely display `idx_email_first_last` but not `idx_email`.  Although `idx_email` *exists* as a valid index, it is considered a prefix of `idx_email_first_last`. The query optimizer can efficiently use the larger composite index for queries that target just the `email` column, negating the need for a separate, dedicated single-column index. Note that the primary key will also be displayed.

It is important to understand that, even if one index isn't visible in the `SHOW CREATE TABLE` output, it still *exists*. You can verify this using `SHOW INDEX FROM users;`, which displays *all* the indexes defined for a table, including those considered redundant in `SHOW CREATE TABLE` output.

Finally, let us consider the case where the columns of the two indexes have different orders.

```sql
CREATE TABLE orders (
    order_id INT AUTO_INCREMENT PRIMARY KEY,
    customer_id INT NOT NULL,
    order_date DATE NOT NULL,
    product_id INT NOT NULL,
    INDEX idx_customer_date (customer_id, order_date),
    INDEX idx_date_customer (order_date, customer_id)
);
```

Here, `idx_customer_date` indexes columns `customer_id` and `order_date` in that order, while `idx_date_customer` reverses the order.  In this instance, both indexes would likely appear in the output of `SHOW CREATE TABLE orders;` as they are not prefixes of each other.  While the query optimizer might still internally select one over the other based on the query at hand, neither index is wholly redundant in the way that `idx_category` was to `idx_category_supplier`. Each index would be beneficial depending on the WHERE clause.

Understanding the difference between defined indexes and the indexes that are deemed ‘useful’ in query optimization is critical for effective database design and troubleshooting. `SHOW CREATE TABLE` provides information on the table definition, not necessarily a comprehensive list of all potentially available indexes. You need to consult with `SHOW INDEX` to get that full perspective.

To improve index management, I would recommend these learning resources. First, explore official MySQL documentation covering index types, indexing best practices, and query optimization. Next, I would suggest reviewing resources discussing compound or multi-column indexes and their relationship to prefix indexes. These typically cover implicit prefixing and cardinality. Finally, research the MySQL query optimizer's EXPLAIN output. Understanding how it determines index use will provide the most nuanced understanding of index relevance. Understanding all these concepts is vital to understanding why you might see just one index when you have created multiple that seem like they should be displayed.
