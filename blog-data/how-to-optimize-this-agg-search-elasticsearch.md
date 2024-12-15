---
title: "How to optimize this agg search elasticsearch?"
date: "2024-12-15"
id: "how-to-optimize-this-agg-search-elasticsearch"
---

alright, so, optimizing an aggregate search in elasticsearch, eh? i’ve been down that rabbit hole more times than i care to count. it’s one of those things that seems simple on the surface, but can quickly spiral into a performance nightmare if not handled correctly. i'll walk you through some things i've learned, mostly from painful experience, and how to make those aggregations sing.

let’s start with the most common pitfalls. when people talk about slow elasticsearch queries, aggregations are often the culprit. it's not elasticsearch's fault per se, it’s usually how we're asking it to work. the biggest issue i see is asking for too much data in the aggregation process, meaning processing the whole index instead of a filtered selection and processing aggregations on fields that are not optimized for it.

the first thing to consider is the scope of your aggregation. are you really trying to aggregate across your entire index or could you narrow that down significantly? elasticsearch’s sweet spot is fast searches, and using it efficiently involves filtering as early as possible. let's say we have an index of user activity logs, and you're trying to aggregate the total number of sessions per day. before you even think about aggregations, make sure you have a good `query` portion of the request that will narrow down your data selection.

for example, instead of this:

```json
{
  "size": 0,
  "aggs": {
    "sessions_per_day": {
      "date_histogram": {
        "field": "session_start_time",
        "calendar_interval": "day"
      }
    }
  }
}
```

you could do something like this:

```json
{
  "size": 0,
  "query": {
    "bool": {
      "filter": [
        {
          "range": {
            "session_start_time": {
              "gte": "now-30d/d",
              "lt": "now/d"
             }
          }
        }
      ]
     }
   },
  "aggs": {
    "sessions_per_day": {
      "date_histogram": {
        "field": "session_start_time",
        "calendar_interval": "day"
      }
    }
  }
}
```

the `query` portion now limits the aggregation to the last 30 days. this drastically reduces the amount of data that elasticsearch has to work with, making the aggregation much faster. in my early days, i used to fire off aggregations across the entire indices, wondering why things were so slow, you live and you learn i guess.

another critical optimization is field data types. the performance of aggregations heavily depends on whether the field you're aggregating on is indexed properly. for example, if you're aggregating on a text field that's analyzed, elasticsearch will have to do a lot more work than if it were a `keyword` field. the `keyword` type stores the entire field content verbatim, which is great for aggregation and sorting.

consider this, if you have something like user names and you are doing an aggregation over them but the field is a `text`, it would first tokenize the words and store them indexed, so you will lose the full term and also the terms you are aggregating on will probably be wrong, a better approach here is defining a multi field over that field for aggregation purposes:

```json
"mappings": {
  "properties": {
    "user_name": {
       "type": "text",
      "fields": {
        "keyword":{
           "type":"keyword",
           "ignore_above": 256
        }
      }
    }
  }
}
```

and then use the `user_name.keyword` field in your aggregation query instead of just `user_name`

```json
{
  "size": 0,
  "aggs": {
    "users_count": {
      "terms": {
        "field": "user_name.keyword",
          "size":10
      }
    }
  }
}
```

i remember a situation where i was trying to aggregate user locations using a text field that was also tokenized for searching by location. the aggregation was so slow, it made my machine feel like it was from the stone age. switching to a multi-field with `keyword` for aggregation made it performant enough to do its job. i guess not every field needs to be analyzed for text searching, it's important to know when to use one or the other.

going a step further, pre-aggregating your data before indexing can be a big win if you know the exact aggregations you are going to need. instead of performing aggregations on the fly, you can create separate indices with aggregated data that are optimized for those specific queries. think of it as a caching mechanism, but at the index level. for instance, if you need daily totals of some values, create an index specifically for storing that daily aggregation, and use a batch process that pre-calculates it and inserts it into the index, rather than performing those aggregation at query time.

for example if you need total daily sales by product from an index, consider creating an index with the schema below:

```json
"mappings": {
  "properties": {
    "day": {
      "type": "date",
      "format": "yyyy-MM-dd"
    },
    "product_id": {
      "type": "keyword"
    },
    "total_sales": {
        "type":"double"
    }
  }
}
```

and then you can perform the following aggregation directly on this index which should be way faster than aggregating over a large raw index of transactions:

```json
{
  "size": 0,
  "aggs": {
    "total_sales_by_product":{
      "terms": {
        "field": "product_id",
        "size": 10
      },
      "aggs":{
        "total_sales_amount":{
          "sum":{
            "field": "total_sales"
          }
        }
      }
    }
  }
}
```

i had a client once that was struggling with real-time dashboards, and the aggregations were just taking too long, even after all the usual optimizations. we moved to pre-aggregating hourly summaries and it solved the problem by magnitudes. it added a little complexity to our data pipelines, but the performance boost was undeniable.

also, be mindful of the aggregations you choose to use. some aggregations are just more computationally expensive than others. `cardinality` aggregations, for example, are more expensive because they involve estimating the number of unique values in a field. avoid them unless you absolutely need that level of accuracy, and when you do need it, use the `precision_threshold` parameter to trade off some accuracy for performance, this parameter controls the amount of memory allocated and the precision of the estimate, lower values will result in faster more inexact estimations.

if you're dealing with large data volumes, consider using the `composite` aggregation to page through your results, composite aggregations allow the pagination of large amounts of data without losing state, this can improve drastically the overall performance since you do not need to aggregate the whole data set at once.

here's another tip – when you are aggregating over a large data set with many different filters, you might want to explore using cached filters to reuse those filters across different aggregations. elasticsearch filter caching can often lead to significant improvements. consider the use of `named` filters so you can use the same filters in different parts of the request to improve performance.

in my experience, tuning aggregations in elasticsearch is rarely a one-size-fits-all solution. it's usually a matter of identifying the bottlenecks, tweaking data types, and possibly restructuring the way your data is stored. if i had to summarize it all, it would be something like: reduce the scope as early as possible, choose the correct data types, pre-aggregate if possible and know your aggregations well.

as for resources, i wouldn't bother with random blog posts too much. instead, i highly recommend checking out “elasticsearch: the definitive guide” by clinton gormley and zachary tong. it's an excellent resource for understanding how aggregations work at a lower level, which is something i had to learn the hard way. also, read the official elasticsearch documentation on aggregations. it's really well-written and very in-depth. it goes into detail about all the available aggregation types, their parameters, and performance considerations. these resources will give you a solid background to tackle any complex aggregation scenario.

oh and one last thing, and probably one of the best advices you'll ever get in your life: always run your queries in elasticsearch profiler or the slow query logs to see where time is being spent in your queries, never assume what is slow, instead get the facts directly from elasticsearch, that's how we should work in this profession, don't guess measure.
