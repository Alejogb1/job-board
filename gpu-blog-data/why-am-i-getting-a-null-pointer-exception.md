---
title: "Why am I getting a null pointer exception when accessing _aggs elements in a scripted metric aggregation?"
date: "2025-01-30"
id: "why-am-i-getting-a-null-pointer-exception"
---
Null pointer exceptions when accessing `_aggs` within a scripted metric aggregation in Elasticsearch typically stem from a fundamental misunderstanding of how aggregations are processed and how the `_aggs` variable is populated during script execution. Specifically, the `_aggs` object within a script isn’t a static mirror of all defined aggregations; it’s a context-sensitive representation populated *only* if the metric aggregation itself is nested within another aggregation that *provides* the context for those named aggregations. My experience, particularly while building analytical dashboards for telemetry data, has consistently revealed this pitfall.

The crux of the issue lies in the scope and timing of aggregation results. Metric aggregations, such as `sum`, `avg`, `min`, `max`, and custom scripted metrics, operate directly on the document stream that matches the query. They compute aggregate values based on this document stream. The `_aggs` variable, on the other hand, only becomes meaningful when an aggregation hierarchy is established. That hierarchy is built by nesting aggregations: a parent aggregation that groups data, and then a child aggregation within each of the parent groups that processes its own data within that grouping. Only then will the `_aggs` data be present at the child level. Without this nesting, or if the metric aggregation is placed as a top-level aggregation, `_aggs` will not be populated and accessing any element within it will inevitably trigger a null pointer exception. Think of `_aggs` as the bridge providing the results of aggregations higher in the hierarchy. If there is no hierarchy, there's no bridge and trying to cross it will naturally cause an error.

Let's examine a few concrete examples to illustrate this point and provide solutions.

**Example 1: Incorrect Usage (Leading to a Null Pointer Exception)**

Imagine you have an index containing sales data, each document representing a single transaction. You want to calculate the total price of items sold and additionally perform a post-processing step that adds a fixed bonus based on the sum of a specific field in another aggregation (which is *incorrectly* applied as a peer aggregation).

Here is a sample query that will result in a NullPointerException:

```json
{
  "size": 0,
  "aggs": {
    "total_sales": {
      "sum": {
        "field": "price"
      }
    },
    "bonus_calc": {
      "scripted_metric": {
        "init_script": "state.bonus = 0",
        "map_script": "state.bonus += 10;",
        "combine_script": "return state.bonus;",
        "reduce_script": "return states.sum(s -> s);"
      },
      "aggs": {
       "nested_total_sales": {
          "sum":{
             "field": "price"
             }
           }
       },
        "params":{
              "bonus_amount": "100",
              "total_sales": "_aggs.total_sales.value"
           }
       }
  }
}
```

**Commentary:**

The problem here is that "bonus\_calc" is on the same level as "total\_sales". The `_aggs.total_sales` path is invalid in this context because `bonus_calc` is not nested *within* an aggregation that would allow access to the result of a peer aggregation. The `params` map is evaluated during the script generation and the `_aggs` object is a runtime construct and is not present in the compilation phase. This does not cause the error at the compilation stage of the query, but it will throw a NullPointerException at runtime, because the `_aggs` object will not exist. Also notice, I've nested an aggregation named `nested_total_sales`, which *does not* solve the underlying problem here, because it does not provide the correct parent/child relationship necessary for the script's context.

**Example 2: Correct Usage (Nested Aggregation)**

To fix the previous example, we need to structure the aggregation correctly by using a bucket aggregation as the parent that provides the context for `_aggs`. For example, the following query will successfully execute:

```json
{
    "size": 0,
    "aggs": {
      "by_category": {
        "terms": {
          "field": "category"
        },
        "aggs": {
            "total_sales": {
              "sum": {
                "field": "price"
              }
           },
          "bonus_calc": {
            "scripted_metric": {
                "init_script": "state.bonus = 0",
                "map_script": "state.bonus += 10;",
                "combine_script": "return state.bonus;",
                "reduce_script": "return states.sum(s -> s);"
              },
             "params": {
               "bonus_amount": "100",
               "total_sales": "_aggs.total_sales.value"
              }
          }
        }
      }
    }
}
```

**Commentary:**

Here, I've introduced a `terms` aggregation on the "category" field, named "by\_category", which creates buckets for each category. Importantly, the "total\_sales" aggregation and the "bonus\_calc" scripted metric aggregation are nested *inside* each category bucket. Now, within the `params` section of "bonus\_calc", `_aggs.total_sales.value` is valid because we're operating within the scope of a particular category's results. Elasticsearch calculates the `total_sales` sum for each category first, and then exposes the result to the `bonus_calc` aggregation *within* each category. The `_aggs` object will be populated within the context of this nested script. This resolves the NullPointerException. It is key that both `total_sales` and `bonus_calc` are under the same bucket.

**Example 3: Correct Usage (Using _doc for Accessing Document Values)**

If, instead of accessing other aggregations, the script needs to operate on individual document fields, the correct method is to use `doc['field_name'].value`, not `_aggs`. For example, let's say that the bonus needs to be added only when a product has the "premium" status, and that status is a field in each document:

```json
{
    "size": 0,
    "aggs": {
       "total_sales":{
         "sum": {
           "field": "price"
         }
       },
       "premium_bonus_calc": {
          "scripted_metric": {
              "init_script": "state.bonus = 0",
              "map_script": "if (doc['status'].value == 'premium'){ state.bonus += 10} ",
              "combine_script": "return state.bonus;",
              "reduce_script": "return states.sum(s -> s);"
             }
        }
    }
}
```

**Commentary:**

Here, the `map_script` directly accesses the "status" field of each document using `doc['status'].value`.  There is no reliance on `_aggs` because no other aggregations are needed to calculate this value, as the data is contained within each document being evaluated. This avoids the NullPointerException issue altogether by not relying on the populated `_aggs` object. Also note that if the field is a keyword, `doc['status'].value == "premium"` can also be used rather than `doc['status'].value.equals("premium")`.

**Recommendations and Best Practices:**

To avoid future null pointer exceptions and other related problems in scripted metric aggregations, the following points are crucial to keep in mind. First and foremost, always be aware of aggregation hierarchy. Ensure that scripted metric aggregations that need to access `_aggs` are correctly nested within a parent aggregation, such as a `terms`, `date_histogram`, or `range` aggregation. Also, remember that the `_aggs` object is context dependent. Only aggregations within the scope of a particular grouping (defined by the parent aggregation) can be accessed using `_aggs`.

Second, thoroughly test your script logic in a development or testing environment before deploying it to production.  The script debugging feature in Elasticsearch is essential here for identifying and resolving such issues. If your script depends on external variables, use the params parameter. This is especially useful when passing the values from parent aggregations for use in child aggregations via the `_aggs` variable.

Finally, avoid complex logic within your scripts if possible.  Often, simple logic within a scripted metric aggregation suffices, and more complex logic is more manageable by using other constructs, such as post processing scripts or other data processing tools if required. If you need to access the fields of individual documents in the script, use the `doc['field_name'].value` format; do not attempt to access them via the `_aggs` object because they will be undefined. By carefully considering the aggregation context and script implementation details, these errors can be readily avoided, which will lead to more reliable, and efficient queries.
