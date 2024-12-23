---
title: "elasticsearch get distinct values field query?"
date: "2024-12-13"
id: "elasticsearch-get-distinct-values-field-query"
---

 so you're after distinct values from a field in Elasticsearch cool I've been there done that got the t-shirt a few times over lets break this down its simpler than you might think but like everything theres a right way and a bunch of wrong ways especially when you start scaling things up

First off when you say "distinct values" we're talking about unique values right avoiding duplicates Its a common ask when you're trying to figure out what options you have in a field whether its for filtering or building aggregations or anything else really

Now you could try just looping through all the documents and building a set on the application side but trust me been there done that it's a total performance killer and not something you want to do even if you have like "small" indices it will catch up to you eventually especially as your datasets grow or your application traffic increases I saw one system crawl to a halt when a junior dev tried that approach and we had to spend a whole night fixing it it was not fun

Elasticsearch has aggregations that are specifically designed for this and they are way more efficient than fetching the whole dataset to client side they run within the engine and send you only the result you are looking for So thats where we are going to be focusing here

The aggregation we want is called "terms" and it's like the Swiss army knife for this kind of stuff It gives you a count of how many times each value occurs but since all we are asking for is distinct values we don't have to use the count part we just have to use the buckets and thats it

So in a nutshell you send a query to Elasticsearch that has a terms aggregation for the field you want It'll give you back a list of unique values and you can work with that in your application logic Its really that simple

Here is the simplest form of it as a curl request assuming you are hitting your es on localhost:9200 with index name `my_index`:

```bash
curl -X GET "localhost:9200/my_index/_search?pretty" -H 'Content-Type: application/json' -d'
{
  "size": 0,
  "aggs": {
    "unique_values": {
      "terms": {
        "field": "your_field_name"
      }
    }
  }
}
'
```

So thats a basic example to retrieve the unique values of the field named `your_field_name` replace this placeholder with your specific field name The `"size": 0` part is crucial because we don't need the actual documents so we ask Elasticsearch not to give them to us This reduces the data being transferred over the wire and its always a good idea to try to get back as little data as possible from Elasticsearch if you do not need it its one of the low hanging fruits for performance improvement

The response will look something like this:

```json
{
  "took": 5,
  "timed_out": false,
  "_shards": {
    "total": 1,
    "successful": 1,
    "skipped": 0,
    "failed": 0
  },
  "hits": {
    "total": {
      "value": 1000,
      "relation": "eq"
    },
    "max_score": null,
    "hits": []
  },
  "aggregations": {
    "unique_values": {
      "doc_count_error_upper_bound": 0,
      "sum_other_doc_count": 0,
      "buckets": [
        {
          "key": "value1",
          "doc_count": 100
        },
        {
          "key": "value2",
          "doc_count": 150
        },
		{
          "key": "value3",
          "doc_count": 50
        }
        // ... other unique values
      ]
    }
  }
}
```

You'll find your unique values under `"aggregations.unique_values.buckets"` the `doc_count` here tells you how many times that particular value appeared but you were after unique values so what you should care for are the key's value in the `buckets` array each is unique value found for the given field. We're not using the `doc_count` part so you can safely ignore it.

But what if you have a lot of unique values like thousands or even millions? The default terms aggregation might return only a limited number of results (usually the top 10) you have to use the `size` parameter to increase this limit. However be careful to not ask for extremely high numbers because that can also affect performance as Elasticsearch has to allocate resources for handling that. You have to have a trade off between performance and functionality

If you know the max number of values is for example 100 then you can use something like this:

```bash
curl -X GET "localhost:9200/my_index/_search?pretty" -H 'Content-Type: application/json' -d'
{
  "size": 0,
  "aggs": {
    "unique_values": {
      "terms": {
        "field": "your_field_name",
        "size": 100
      }
    }
  }
}
'
```

This will try to get the top 100 distinct values and if you know your dataset well this should work fine for you and this method should suffice if your dataset does not have more unique values than you are requesting.

Now what if your field contains multiple values? Elasticsearch by default treats fields as an array so if you have a document with the field value as `["value1", "value2"]` that will be accounted as having those both values not a single value containing those and it will correctly return the unique values if both of those values appear in other documents independently. This is usually how things should work with no additional configurations.

But what if you want only unique values from a nested field you might need to work with the `nested` aggregation to first handle the nested values correctly here is an example:

```bash
curl -X GET "localhost:9200/my_index/_search?pretty" -H 'Content-Type: application/json' -d'
{
  "size": 0,
  "aggs": {
    "nested_values": {
      "nested": {
        "path": "nested_field"
      },
      "aggs": {
        "unique_nested_values": {
          "terms": {
            "field": "nested_field.inner_field"
          }
        }
      }
    }
  }
}
'
```

Replace `nested_field` with the path to your nested field and `nested_field.inner_field` to the specific field within the nested document that contains the value you want you might need to adjust `nested_field.inner_field` according to your field mappings. This nested aggregation allows you to properly get the distinct value of the nested field.

I remember one time we were trying to get the unique tags on a system and we had nested values inside a custom mapping it was a real pain to figure out what was going on it took a few hours of debugging to realise we had to use the nested aggregation in addition to the terms aggregation we were using to obtain the unique tags.

You might be thinking what is a good resource to learn more about Elasticsearch I would recommend to take a look at the official Elasticsearch documentation its probably the most complete resource out there and they are constantly updating it. You can find it on the Elastic website its a very extensive piece of documentation. There is also the "Elasticsearch: The Definitive Guide" book by Clinton Gormley and Zachary Tong that gives you a deep dive on Elasticsearch if you are more into books. There are plenty of blog posts and tutorials out there but nothing really beats the official documentation and the definitive guide book.

One more thing to consider for performance is field mapping in Elasticsearch if the field you are querying is `not_analyzed` its going to perform way better than a field that is being analysed and tokenized. If you do not need any full text search on that particular field make sure it is configured as `not_analyzed` in the field mapping so Elasticsearch can treat it as a simple value that is stored as a single entity rather than a set of tokens this is going to give you the most optimal performance since we are trying to extract distinct values not trying to perform searches on it.

And there you have it using terms aggregation to get distinct values from your Elasticsearch index. Just make sure to use `size:0` and configure `size` field if you know the max size of your unique values and be aware of nested values when using nested aggregations. One time I was debugging a slow query in a production environment and after two hours I realised I was querying an analyzed field for distinct values without need when I have the exact version of it as `not_analyzed` it was silly and I wasted a lot of my precious time.

Oh and before I forget a good programmer once said "why do programmers prefer dark mode because light attracts bugs!" Anyway I hope that helps good luck and let me know if you have any other questions.
