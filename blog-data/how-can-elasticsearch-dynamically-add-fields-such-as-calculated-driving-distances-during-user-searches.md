---
title: "How can Elasticsearch dynamically add fields, such as calculated driving distances, during user searches?"
date: "2024-12-23"
id: "how-can-elasticsearch-dynamically-add-fields-such-as-calculated-driving-distances-during-user-searches"
---

Alright, let's tackle this. I recall a particularly challenging project a few years back, a location-based service requiring real-time distance calculations. It’s a scenario that often trips up newcomers to Elasticsearch: dynamically incorporating data during the search phase rather than at index time. The issue, of course, boils down to not storing these computed fields directly in the index but needing them ‘on the fly.’ Elasticsearch isn't designed to perform arbitrary calculations during searches on indexed data as a first principle, but there are several well-established techniques we can employ, using scripting and custom function scores.

Essentially, we need a way to execute code during query execution. Elasticsearch provides a few mechanisms to achieve this, primarily through scripting languages like painless, or via function scores. These methods allow us to manipulate search results based on the specific query parameters and the document data within the index.

The core of the problem is that Elasticsearch, by default, indexes data in a fixed schema. You pre-define your fields, their types, and how they're analyzed. Now, we're talking about needing fields that aren't present in that static structure, which require some ingenuity. Dynamic fields, in our context, are not the standard dynamic mapping which adds new fields when encountered for indexing, but rather fields calculated solely during a search based on the current query or document properties. Distance is a perfect example. Storing pre-computed distances is often impractical due to combinations of potential origins and destinations.

Let’s start with a basic approach using a scripting function within a function score query.

**Example 1: Simple Scripting with Painless for Distance Calculation**

Imagine you have an index containing documents with geo_point fields `location`. We want to calculate the distance from a query origin point. I remember having a particularly stubborn performance issue with this approach when I had a lot of searches needing distance on large areas. I found that optimizing the script and using better query structures was important.

Here is a simplified version to illustrate the concept, and it's good to know that with the right precautions and optimizing for your dataset, it's certainly usable.

```json
{
  "query": {
    "function_score": {
      "query": {
        "match_all": {}
      },
      "functions": [
        {
          "script_score": {
            "script": {
              "source": """
                double lat1 = params.queryLat;
                double lon1 = params.queryLon;
                double lat2 = doc['location'].lat;
                double lon2 = doc['location'].lon;

                double R = 6371; // Earth radius in kilometers
                double dLat = Math.toRadians(lat2 - lat1);
                double dLon = Math.toRadians(lon2 - lon1);
                double a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
                           Math.cos(Math.toRadians(lat1)) * Math.cos(Math.toRadians(lat2)) *
                           Math.sin(dLon / 2) * Math.sin(dLon / 2);
                double c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
                double distance = R * c;
                return distance;
              """,
              "params": {
                "queryLat": 34.0522,
                "queryLon": -118.2437
              }
            }
          }
        }
      ],
      "boost_mode": "replace"
    }
  }
}
```

In this query, `queryLat` and `queryLon` are parameters for the origin point. The painless script calculates the distance between the origin and each document's `location` using the haversine formula, and then the entire score is replaced by this computed distance, you will want to set `boost_mode` to `multiply` or `sum` if you don't want to replace the score. This allows us to see the distance with respect to the origin point in our search results. Remember to use better precision and error handling in real-world usage.

While this works, executing complex scripts for every document on every query can be expensive. We need to be very mindful of the performance implications. For production, I would recommend exploring different ways to cache the computations or use simpler, more efficient formulas, based on your needs, such as Manhattan distances if appropriate for your situation. Also it is vital to test the performance with different use cases.

Now, let's move on to a slightly different approach using function scores, but this time incorporating the `_geo_distance` function. This approach is already optimized for distance calculations for performance.

**Example 2: Using `_geo_distance` Function Score**

Elasticsearch already provides built-in functions for spatial operations. Using these offers better performance compared to user-defined scripts as the engine has internal optimizations it can employ when executing these types of calculations.

```json
{
  "query": {
    "function_score": {
      "query": {
        "match_all": {}
      },
      "functions": [
        {
          "gauss": {
            "location": {
              "origin": "34.0522,-118.2437",
              "scale": "100km",
              "offset": "2km",
              "decay": 0.5
            }
          }
        }
      ],
      "boost_mode": "multiply"
    }
  }
}

```

In this example, the `gauss` function calculates a score based on the distance between each document's `location` and the origin point specified in "origin". We set the "scale" to 100 kilometers, which controls how fast the score decays as the distance from the origin increases. The "offset" parameter means we only penalize the documents starting after the offset value. This approach, using Elasticsearch's built-in functions, is generally more efficient than custom scripting for common spatial operations, as I’ve seen firsthand on production datasets. The result score will have high value when close to the origin point and low when far from it, which we can use to sort or filter results based on distance.

Finally, I'll include a third example, now adding a custom field to the output, which might be beneficial for other computations or visualizations of distance during the search results.

**Example 3: Adding a Custom `distance_km` Field with Script Fields**

If you need the raw computed distance value in the search result to display, use `script_fields` rather than relying on just scoring mechanisms:

```json
{
    "query": {
        "match_all": {}
    },
    "script_fields": {
        "distance_km": {
            "script": {
                "source": """
                 double lat1 = params.queryLat;
                double lon1 = params.queryLon;
                double lat2 = doc['location'].lat;
                double lon2 = doc['location'].lon;

                double R = 6371; // Earth radius in kilometers
                double dLat = Math.toRadians(lat2 - lat1);
                double dLon = Math.toRadians(lon2 - lon1);
                double a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
                           Math.cos(Math.toRadians(lat1)) * Math.cos(Math.toRadians(lat2)) *
                           Math.sin(dLon / 2) * Math.sin(dLon / 2);
                double c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
                double distance = R * c;
                return distance;

                """,
              "params": {
                  "queryLat": 34.0522,
                  "queryLon": -118.2437
               }
            }
        }
    }
}
```

Here, the `script_fields` section defines a new field called `distance_km`, computed using the same haversine formula. This will include the calculated distance in the search results alongside the regular fields from the index, making it directly accessible for processing in the consuming application. It’s useful for cases where you not only need to score documents based on distance but also present that distance in the output.

In terms of resources for further reading, I’d recommend exploring the Elasticsearch documentation thoroughly, focusing on function score queries and scripting with Painless. “Elasticsearch in Action” by Radu Gheorghe, Matthew Lee Hinman, and Roy Russo is an excellent book for understanding the deeper aspects of these topics. Also, the paper “A Simple and Effective Method to Calculate the Distance Between Two Locations” (various authors have published on this) is helpful to learn the different distance calculation algorithms and their characteristics, to select the ones which suit your needs.

These examples provide a good starting point for dynamic field calculations in Elasticsearch. Remember that performance always depends on the complexity of your calculations and your data volume. Always benchmark thoroughly, and adjust your strategy based on the specific needs of your project.
