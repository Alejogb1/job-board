---
title: "Do fixed LOD expressions return individual row values?"
date: "2025-01-30"
id: "do-fixed-lod-expressions-return-individual-row-values"
---
In my experience, fixed Level of Detail (LOD) expressions, in the context of data analysis tools like Tableau, do not directly return individual row values in the same way that row-level calculations do. Instead, they compute an aggregated value at a specified level of granularity, and that aggregated result is then *attached* to each row that matches that specified granularity. Understanding this distinction is crucial when constructing complex analyses and anticipating the output of such calculations.

The key to understanding this behavior lies in the 'fixed' clause itself. Unlike other LOD expressions that are computed based on the visualization's current dimensions or are relative to specific subsets, fixed LOD expressions calculate an aggregation irrespective of the dimensions in the view. They perform this aggregation at the level specified by the dimensions declared within the `FIXED()` syntax. For each row, the tool checks the specified dimensions to determine the group to which the row belongs. It then assigns the aggregated value for that group to the corresponding row. Thus, while it appears they return a value *for* each row, the value itself is based on an aggregated calculation. Therefore, a fixed LOD expression does not return a distinct row-level result but a repeated aggregated calculation at the given level.

This distinction becomes particularly important when we think about how aggregations and filtering interact with fixed LODs. If you were to use a fixed LOD expression to find the total sales per customer, that total sales value would be the same value across all records for the customer, even if the data at hand included multiple transactions. The value wouldn't reflect the sales of the individual transaction. It’s a constant for a grouping defined at the fixed dimension level. Consequently, they behave almost like attributes or properties at the defined dimension level, allowing for comparisons across the data without the distortion that would occur from mixing row-level detail and aggregated calculations.

Here are three practical scenarios that illustrate the behavior of `FIXED` LOD expressions, coupled with commentary on the resulting output:

**Example 1: Average Sales Per Region**

Assume a dataset containing sales data with `Order ID`, `Region`, and `Sales`. We want to find the average sales per region and attach this result to each row.

```
// Tableau Calculation
{FIXED [Region] : AVG([Sales])}
```

**Commentary:**
This fixed LOD expression, `FIXED [Region] : AVG([Sales])`, calculates the average sales for each region. In effect, Tableau first computes `AVG([Sales])` for *every* region in the dataset. If, for example, the “East” region has an average sales of $500, then every single row associated with the “East” region will have the value $500 as the result of this expression, regardless of individual order sales within that region. This value persists even if `Order ID` is present in the view or if records are displayed in a detail table, illustrating that the output is not row-level. If the goal was to find the average of the individual sales, a simple `AVG([Sales])` would be appropriate. However, this fixed LOD provides regional aggregate context, rather than detail about each order.

**Example 2: Identifying Products Above Average Sales (by Category)**

Now consider a dataset with `Product Name`, `Category`, and `Sales`. We aim to determine which products have sales above their category average.

```
// Tableau Calculation 1 - Category Average Sales
{FIXED [Category] : AVG([Sales])}

// Tableau Calculation 2 - Sales Compared To Category Average
[Sales] > [Category Average Sales]
```

**Commentary:**
Here, we use a two-calculation approach. First, `FIXED [Category] : AVG([Sales])` computes the average sales *per* category, attaching this category average value to *every* record associated with that category, mimicking a property of the category on all of its related product rows. Second, we create a boolean field, `[Sales] > [Category Average Sales]`, which compares each product’s individual sales value to that attached average. The result will return `true` for products above the category average and `false` otherwise. This example underscores how fixed LOD expressions don’t operate at the row level, instead, they distribute an aggregate computed at the fixed level, enabling complex comparative logic. It showcases how this behavior allows us to assess each row's data within a specific group's broader context.

**Example 3: Percentage of Total Sales per Customer**

Consider customer data containing `Customer Name`, `Order ID`, and `Sales`. We seek to calculate each customer's percentage of total sales.

```
// Tableau Calculation 1 - Total Sales
{FIXED : SUM([Sales])}

// Tableau Calculation 2 - Customer Sales
{FIXED [Customer Name] : SUM([Sales])}

// Tableau Calculation 3 - Percentage of Total
([Customer Sales] / [Total Sales])
```
**Commentary:**
This uses three LOD expressions.  The first, `FIXED : SUM([Sales])`, calculates the overall total sales. Because there are no dimensions declared within the `FIXED()` clause, the aggregation calculates the total for *all* rows in the entire dataset. This total sales value is applied to every record in the data. Second, `FIXED [Customer Name] : SUM([Sales])`, sums all sales within each customer. Finally, the ratio of `[Customer Sales]` over the `[Total Sales]` delivers each customer’s percentage contribution to overall sales. The fact that both totals are derived by fixed LODs showcases the behavior of applying an aggregated value to all records at different dimension levels, not at row level. The key point is that although all records will receive both `[Total Sales]` and `[Customer Sales]` values, these totals are not based on individual transactions.

To solidify your understanding of LOD expressions further, consider focusing on the following resources, specifically in documentation and training materials, from the vendors or educators of the tool you utilize:

*   **Official Documentation:** Thorough documentation from the tool's provider offers specific syntax nuances and detailed explanations for edge cases. Pay close attention to sections discussing Level of Detail expressions, aggregation, and data context.
*   **Online Tutorials:** Platforms offering in-depth video tutorials often demonstrate different scenarios where LOD expressions provide practical value. These can help visualize the mechanics and impacts of `FIXED` LOD calculations.
*   **Sample Workbooks:** Experiment with pre-built workbooks that employ fixed LOD expressions in a variety of analysis contexts. This hands-on approach allows for examination of calculations and their direct impact on the data within the visualizations.
*   **Community Forums:** Participation in online forums or communities surrounding your data analysis tool provides access to solutions to specific challenges and nuanced understandings of edge cases from other practitioners who may have experience with LODs that you have not personally encountered.

In conclusion, while fixed LOD expressions may seem to return a value for each row, this value is the result of an aggregation computed at the specified level and then applied across rows sharing that grouping. The values are not independently calculated at a row level. This understanding is essential for the accurate use and interpretation of fixed LODs and enables complex analysis by allowing for comparison of rows within the context of calculated aggregate groupings.
