---
title: "How do I calculate a final script score from an Elasticsearch function score query?"
date: "2024-12-23"
id: "how-do-i-calculate-a-final-script-score-from-an-elasticsearch-function-score-query"
---

Alright, let's talk about script scores within Elasticsearch's function score queries. I've certainly spent my fair share of time grappling with this, and I recall vividly a project a few years back involving a large content recommendation engine where we needed extremely nuanced scoring logic. We moved away from static weights quite rapidly. Getting those final scores to behave exactly as we intended required a pretty solid understanding of how the function score query operates and how to best leverage scripts within it.

The core concept here, for anyone coming across it, is that Elasticsearch’s function score query allows you to go beyond basic relevance scores. You can apply custom scoring functions based on a document’s fields, various factors, and, yes, even custom scripts. The script score function specifically lets you inject arbitrary groovy, painless, or lucene expressions directly into the scoring calculation. The end result, or the “final script score” as you phrased it, is a composite score derived from several operations. It's crucial to understand this composition: you start with the base query score (what would be returned if you didn't use function score), then you apply whatever modifiers you've defined within the function score query, which can include one or more script scores.

The function score query structure typically involves a `query` field, which dictates the documents that are selected initially, and a `functions` array that describes how the scores for those documents are to be modified. Each item in this array contains a function type (such as `script_score`), an optional `filter`, and, in the case of `script_score`, the `script` block. It’s the *execution of this `script` that directly affects the document’s final score*.

The formula at the heart of the process can be, broadly, interpreted as this:

`FinalScore = BaseQueryScore * (1 + functionScoreModifier_1 + functionScoreModifier_2 + … + functionScoreModifier_n)`

where the `functionScoreModifier_i` is the score produced by the i-th function in the functions array. Script scores add to or change this in very particular and often very powerful ways.

Here are some concrete examples based on scenarios I've seen:

**Example 1: Simple Field-Based Boost with Groovy**

Let’s say you have an index of articles, and you want to boost articles that are more recent. This isn't solely based on date but rather a non-linear function that makes recent articles *significantly* more relevant. We can achieve this using a groovy script and a timestamp.

```json
{
  "query": {
    "match": {
      "content": "search term"
    }
  },
  "rescore": {
    "window_size": 100,
    "query": {
       "score_mode": "multiply",
       "function_score": {
         "functions": [
            {
              "script_score": {
                "script": {
                  "lang": "groovy",
                  "source": "double days_since = (System.currentTimeMillis() - doc['publish_date'].date.getTime()) / (1000 * 60 * 60 * 24); return 1.0 + Math.max(0, 1.0/(days_since * days_since)) * 10; "
                }
              }
            }
         ],
         "boost_mode": "multiply"
       }
     }
   }
}
```

In this example, the initial query searches for the “search term” in the content field. The script calculates the number of days since the `publish_date`. Then, it returns a score that boosts more recent documents. Notice the use of `rescore` to apply the scoring within the top 100 results of the base query, which keeps the rescore process efficient. This is a useful optimization to consider, especially on large result sets. The `score_mode` is important here, dictating how each function’s result is combined, in this case `multiply`, and the script contributes a multiplicative factor to the final score.

**Example 2: Using Painless with Parameters**

Imagine you want to boost products based on a combination of their popularity and their review score. Painless, the recommended scripting language for Elasticsearch from version 5 onward, is a great fit here due to its safety and efficiency. This time let's also inject the current date to make a time decay, just like the previous groovy example, only a bit more elegant.

```json
{
    "query": {
      "match": {
        "description": "search term"
      }
    },
   "rescore": {
    "window_size": 100,
    "query": {
       "score_mode": "multiply",
       "function_score": {
            "functions": [
               {
                  "script_score": {
                    "script": {
                      "lang": "painless",
                      "params": {
                         "decayFactor": 0.1,
                         "maxBoost": 5,
                         "currentDateMillis" : "2024-03-08T12:00:00.000Z"
                      },
                      "source": """
                      double daysSince = (params.currentDateMillis.toInstant().toEpochMilli() - doc['publish_date'].date.toInstant().toEpochMilli()) / (1000 * 60 * 60 * 24);

                      double popularityBoost = doc['popularity'].value * params.decayFactor;

                      double reviewBoost = doc['review_score'].value * 0.8;

                      double timeBoost = params.maxBoost / (1 + Math.exp(daysSince / 20));

                      return 1 + (popularityBoost + reviewBoost + timeBoost);
                      """
                    }
                  }
               }
           ],
         "boost_mode": "multiply"
        }
      }
    }
  }
```

Here, we are defining parameters passed into the painless script. This keeps the script flexible, making adjustments easier without having to alter the query’s text. We're using a time decay, the review score, and the `popularity` field of the document to construct a more sophisticated score. This approach is highly powerful for tuning relevance in complex scenarios. Note the use of epoch time for date comparisons to ensure numerical calculations are accurate.

**Example 3: Using Lucene Expressions for Optimized Calculations**

For even faster execution, especially when dealing with numerical data, you can leverage lucene expressions. These are compiled directly into the lucene engine, making them extremely efficient. This is quite beneficial if your script logic can be expressed as a lucene expression. Let’s assume we have two fields, `fieldA` and `fieldB`, and we want to apply a simple mathematical operation.

```json
{
  "query": {
    "match_all": {}
  },
    "rescore": {
    "window_size": 100,
    "query": {
       "score_mode": "multiply",
      "function_score": {
        "functions": [
          {
            "script_score": {
              "script": {
                "lang": "expression",
                "source": "1 + log10(fieldA + fieldB)"
              }
            }
          }
        ],
         "boost_mode": "multiply"
      }
    }
  }
}

```

This example, while simple, highlights the effectiveness of lucene expressions when your score logic is fundamentally mathematical. The `source` here uses the expression language syntax directly. Lucene expressions come with some constraints (not as flexible as painless), but when suitable, they provide the best performance.

**Key Considerations:**

*   **Performance:** Script execution has a performance cost. It's very important to keep the script as lightweight as possible. Avoid complex logic inside the script when you can perform that logic beforehand. If using painless, try to simplify as much as possible. For number crunching use lucene expressions when possible, as they run directly in lucene.
*   **Testing:** Thorough testing is essential when incorporating custom scripts into your scoring mechanism. Always have a comprehensive test suite, verifying that the final scores behave as expected for a variety of cases.
*   **Parameterization:** When scripts become even moderately complex, use parameters to make them easier to adjust and maintain without changing the entire query.
*   **Security:** Pay special attention when enabling groovy scripts (if using pre-Elasticsearch 7.x), as they can expose security vulnerabilities if not handled with caution. Painless, designed with safety in mind, is preferred.

For deeper understanding, the official Elasticsearch documentation on the function score query and scripting is a crucial resource. For further study into scripting performance I recommend “Lucene in Action”, 2nd edition by Michael McCandless, Erik Hatcher and Otis Gospodnetic. For a more academic paper that could help with scoring functions, look into the original "BM25" paper, *Okapi at TREC-3*, by Stephen E. Robertson, Steve Walker, and Micheline Hancock-Beaulieu, which laid the groundwork for much modern search ranking. These will provide a solid foundation for understanding the nuances and practical application of function score queries and script scoring in Elasticsearch.
