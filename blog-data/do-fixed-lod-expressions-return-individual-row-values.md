---
title: "Do fixed LOD expressions return individual row values?"
date: "2024-12-23"
id: "do-fixed-lod-expressions-return-individual-row-values"
---

, let's unpack this. Before we dive into the specifics, it's worth stating that questions about Level of Detail (LOD) expressions, especially fixed ones, are common and sometimes a source of misunderstanding, even after years of working with them. I've personally spent countless hours debugging visualizations where the expected behavior didn't quite match the reality, specifically regarding fixed LODs and row-level interactions. So, let me break down whether fixed LOD expressions return individual row values, addressing common misconceptions along the way.

Fundamentally, the answer isn't a simple yes or no. It depends heavily on *how* the fixed LOD is structured and *what* calculations it contains within its scope. A fixed LOD expression, in its core functionality, is designed to compute a value *at a specified level of detail*, independent of the level of detail in the visualization or the row-level context of the data. That’s crucial. It means it's not inherently a row-by-row operation in the way you might expect from, say, a simple aggregate calculation within a table.

Here's the key to understanding this: fixed LODs are defined by dimensions stated *after* the `FIXED` keyword. Think of it as "compute this value *for* these dimensions." The value computed is then associated with all rows that match those dimensions, not necessarily at the individual row level *if the dimensions you use do not granularly define rows.* This means that if the fixed dimensions are a subset of the dimensions presented in the visualization, the result is applied to every row in the subset (and potentially duplicated) because it's calculating a value related to a more aggregated set of data. Now, if you use *all* available dimensions, and if you have unique rows at all dimensions, then *yes* you are effectively dealing with row-level values.

To clarify, consider three scenarios, each demonstrating a different approach and outcome:

**Scenario 1: Fixed LOD with Aggregated Output**

Imagine we're working with sales data, and we want to find the maximum sales amount per customer, irrespective of the specific product they purchased. We have a table with columns like `Customer ID`, `Product Name`, and `Sales Amount`. Here's how we'd approach it with a fixed LOD:

```sql
-- Assuming a data platform that uses a SQL-like syntax compatible with LOD operations
-- This is a conceptual representation; specifics may vary across tools.

CREATE TEMP TABLE sales_data (
    customer_id VARCHAR(50),
    product_name VARCHAR(100),
    sales_amount DECIMAL(10, 2)
);

INSERT INTO sales_data (customer_id, product_name, sales_amount)
VALUES
('Cust1', 'Product A', 100),
('Cust1', 'Product B', 150),
('Cust2', 'Product A', 200),
('Cust2', 'Product C', 100),
('Cust3', 'Product B', 50),
('Cust3', 'Product D', 75);

-- The fixed LOD calculation. Note this isn't actual SQL but a conceptual representation for explanation.
-- In practice this syntax could be different based on the specific tool
SELECT
    *,
    (
        SELECT max(sales_amount)
        FROM sales_data AS subquery
        WHERE subquery.customer_id = mainquery.customer_id
    ) AS max_sales_per_customer
FROM sales_data AS mainquery;

-- In a data visualization tool, this would typically look something like
-- {FIXED [Customer ID] : MAX([Sales Amount])}
```
In this case, the `max_sales_per_customer` calculated within the fixed LOD will *not* represent individual row values for the `sales_amount`. Instead, it’s returning the maximum sales value *for each customer* and then applies that maximum value to every row corresponding to that customer. Observe how the `MAX(sales_amount)` is calculated *at the customer level*. The fixed LOD here is not intended to be row specific to sales amount but rather a customer level measure.

**Scenario 2: Fixed LOD with Row-Level Detail using all Dimensions**

Now, let’s say we want to create a calculation that's inherently tied to each specific row in the dataset. Assuming our dataset includes all dimensions that uniquely identify each row, we use these dimensions in the `FIXED` clause:

```sql
-- Assuming a data platform that uses a SQL-like syntax compatible with LOD operations
-- This is a conceptual representation; specifics may vary across tools.

CREATE TEMP TABLE order_details (
    order_id INT,
    product_id INT,
    quantity INT,
    price DECIMAL(10, 2)
);

INSERT INTO order_details (order_id, product_id, quantity, price)
VALUES
(1, 101, 2, 10.00),
(1, 102, 1, 20.00),
(2, 101, 1, 10.00),
(3, 103, 3, 15.00),
(3, 102, 1, 20.00);


SELECT
    *,
    (
        SELECT quantity * price
        FROM order_details AS subquery
        WHERE subquery.order_id = mainquery.order_id
        AND subquery.product_id = mainquery.product_id

    ) AS line_total
FROM order_details AS mainquery;
-- In a data visualization tool, this would typically look something like
-- {FIXED [order_id], [product_id] : [quantity] * [price] }

```
In this scenario, because we’re fixing on `order_id` and `product_id` – which we assume uniquely identify each line item — the calculated `line_total` *does* effectively return a row-specific value because that's how the `FIXED` calculation was defined. It sums up quantities in case the same product is in multiple rows, but the calculation is indeed specific to the combination of `order_id` and `product_id`. If the level of detail specified in the `FIXED` expression is the same as the row-level detail of the source data, then the FIXED expression will return a row-level result.

**Scenario 3: Conditional Row-Level Values within Fixed LOD**

Let's take the second example and modify it with conditional logic:

```sql
-- Assuming a data platform that uses a SQL-like syntax compatible with LOD operations
-- This is a conceptual representation; specifics may vary across tools.

CREATE TEMP TABLE order_details (
    order_id INT,
    product_id INT,
    quantity INT,
    price DECIMAL(10, 2),
    discount DECIMAL(5,2)
);

INSERT INTO order_details (order_id, product_id, quantity, price, discount)
VALUES
(1, 101, 2, 10.00, 0.10),
(1, 102, 1, 20.00, 0.05),
(2, 101, 1, 10.00, 0.00),
(3, 103, 3, 15.00, 0.20),
(3, 102, 1, 20.00, 0.00);



SELECT
    *,
    (
      SELECT
       CASE
          WHEN discount > 0 THEN (quantity * price) * (1 - discount)
          ELSE quantity * price
          END
      FROM order_details AS subquery
      WHERE subquery.order_id = mainquery.order_id
      AND subquery.product_id = mainquery.product_id

    ) AS discounted_line_total
FROM order_details AS mainquery;

-- In a data visualization tool, this would typically look something like
-- {FIXED [order_id], [product_id] : IF [discount] > 0 THEN [quantity] * [price] * (1-[discount]) ELSE [quantity] * [price] END }
```
In this instance, we introduce a discount and conditionally apply it within the fixed LOD. While still fixed at `order_id` and `product_id` granularity, the calculation now incorporates conditional row-specific logic. The result, `discounted_line_total`, is specific to each row but is calculated using logic that depends on the values *within* that row, not as an aggregation across multiple rows. The key takeaway is that the dimension list determines the granularity of the LOD calculation, while the actual logic within the calculation is specific to that calculated group (in this case a row since we use all available dimensions to define a unique row).

In summary, fixed LOD expressions don't inherently return individual row values in a simplistic one-to-one manner. Their output depends entirely on the defined dimensions in the `FIXED` clause. If those dimensions result in a single aggregated value being applied to multiple rows (e.g. the maximum sales amount per customer), then it isn't a row-by-row operation. If all dimensions which uniquely identify each row are included, then it is effectively row level. It's crucial to understand this distinction to leverage the true power and flexibility of LOD calculations. If you’re looking to deepen your knowledge, I'd highly recommend reading up on the concepts outlined in *The Definitive Guide to Tableau LOD Expressions* by Joshua Milligan, which delves much more into the nuances of the different types of LOD expressions and their use cases. For a more theoretical background, consider delving into literature on multi-dimensional databases or OLAP (Online Analytical Processing).

Hope this helps clarify things for you.
