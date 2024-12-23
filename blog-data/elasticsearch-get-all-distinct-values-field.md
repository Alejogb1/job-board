---
title: "elasticsearch get all distinct values field?"
date: "2024-12-13"
id: "elasticsearch-get-all-distinct-values-field"
---

 so you're asking about how to get all the distinct values of a field in Elasticsearch got it been there done that several times lets dive in I remember back in my early days working on a massive data ingestion project for a social media platform oh man the things we tried the sheer amount of data being pumped in was insane anyway one of the first things we realized was how crucial it was to be able to quickly and efficiently get unique values for various fields it was like trying to find a specific grain of sand on a beach but way way more complex

So lets get this out of the way right now elasticsearch isnt a relational database you cant just fire off a `SELECT DISTINCT` query you need to think a little differently its more like a sophisticated document store where everything is essentially json at the core its all about how you index your data and how you query it later we're gonna talk aggregations its your best friend here

**First Try Terms Aggregation**

The basic way to do this in ES is using the `terms` aggregation this aggregation will give you a count of each unique value in a specified field and the values themselves you get the data and the count its the most common approach for this problem and has been my go to for years

```json
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
```

This is the simple version replace `"your_field_name"` with your actual field name and thats it the size is 0 cause we dont want to see any document hit just the aggregate results.  The response will give you the buckets for each term in that field and the counts associated with them if you don't need the counts you can ignore them but you get them for free with this request type of request. Easy enough right

**What about Size Limits**

 so maybe you have a field with a ton of distinct values like tags for example or usernames maybe even ip addresses if you dont set the size limit it defaults to 10 and you'll miss unique values from the tail end this is a common gotcha that I have seen even myself fallen for numerous times so we need to set the size parameter explicitly in terms to be sure that it lists ALL unique values in a field.

```json
{
  "size": 0,
  "aggs": {
    "unique_values": {
      "terms": {
        "field": "your_field_name",
        "size": 10000
      }
    }
  }
}
```

Now with `size: 10000` we are requesting the first 10000 terms I have seen people use even 100000 but be aware that this can cause issues if you are dealing with huge numbers of unique values since it can make the query slower. In addition to this you need to keep in mind that if you have more than the set size the response will be truncated this is important and easy to miss if you are testing with small data sets so make sure to check if your response is complete for your use case this happened to me on an old project when I missed a bunch of unique IPs due to not setting the size which led to days of debugging and figuring that it was a simple size configuration

**The Cardianlity Aggregation**

Ok so you have huge amount of unique values in a field you might not need to know every single value but just the count of the distinct values and this is where the `cardinality` aggregation comes in handy. I used this once when I needed just to know number of distinct user actions in a data streaming platform for a weekly report i didnt need to know the actions just how many were used

```json
{
  "size": 0,
  "aggs": {
    "unique_count": {
      "cardinality": {
        "field": "your_field_name"
      }
    }
  }
}
```

This is super useful it gives a fast approximate count of the unique values for a specific field without needing to extract all the unique values themselves. The `cardinality` aggregation uses a HyperLogLog++ algorithm for efficient counting this algorithm works with a certain degree of approximation but it is super fast and efficient for large datasets.

**Best Practices**

*   **Field Data Type**: Make sure that your field type is set correctly in ES. Using keyword fields is important for terms aggregation its faster and more appropriate for distinct searches if you have the field as a text type you might need to use a subfield called `.keyword` or map the field again as a keyword which you should generally avoid.
*   **Size Limits**: Watch out for size limits especially with the `terms` aggregation if you have lots of unique values it is crucial that you account for the `size` parameter or you will not get all the unique values.
*   **Performance**: For really large data sets you need to optimize your search and aggregation its a complex system that needs to be well tuned but thats another story for another day.
*   **Data Modeling**: Before you even get to the search/aggregation phase make sure that you have modeled your data correctly. The right data model can save you hours of query writing and optimization.

**Going Further**

I'm not going to go into the rabbit hole of all the options you have in ES aggregations but if you really want to up your game take a look at the official ES documentation or this book i read a while back "Elasticsearch in Action" by Radu Gheorghiu et al it explains the inner workings of aggregations and how you can fine tune them.

**Final Notes**

So basically there are different paths to get all distinct values in Elasticsearch each with its own particular use cases. The `terms` aggregation for getting the values and their counts the `cardinality` aggregation for just getting the number and `size` for preventing incomplete results. I think this is enough to get you going just a friendly reminder to not forget to check the size limits that have cost me some time over my career which I remember that like it was yesterday I had to rebuild an index due to having a wrong size value. This is the reality of development sometimes and also if you are reading this from the future, you probably are still getting incomplete results because of this silly mistake. So good luck and happy searching! I'm out now gotta go fix another bug lol.
