---
title: "How to access LUIS recognizer results with verbose set to true?"
date: "2024-12-16"
id: "how-to-access-luis-recognizer-results-with-verbose-set-to-true"
---

Okay, let’s tackle this. I've spent my share of time wrangling with LUIS, especially when the need arises to extract granular details from its analysis. Specifically, accessing those verbose results can be crucial for complex conversational flows, and I've definitely been there, scratching my head over the structure of the JSON it throws back. The key, as with many things in development, lies in understanding the precise shape of the data and how to best process it programmatically.

The ‘verbose’ parameter in LUIS API calls, when set to true, fundamentally alters the structure of the response you receive. Instead of getting just the top-scoring intent and a condensed list of entities, you’re provided with a more detailed breakdown. This includes scores for all intents, confidence levels for multiple entity recognitions (including those that might be overlapping or ambiguous), and additional metadata that can prove incredibly valuable. Without verbose mode, you're essentially only getting the highlights reel. With it, you get the full director's cut, and that makes a difference.

In my past experience, I encountered a scenario where I needed to handle conversational disambiguation. The user might say something that could potentially trigger multiple intents. If the confidence scores for these intents were close, I'd need to examine the complete intent list to design a clarification dialogue, something that isn't feasible with the default condensed LUIS response. This is where verbose mode, combined with careful code, becomes essential. I remember initially trying to rely on just the top intent and was regularly running into frustrating dead ends in the conversation.

Let's delve into how you’d actually pull this information apart. The verbose response, as you likely know, is a JSON payload. It can look somewhat intimidating, particularly if you are coming from working only with the brief response. It includes a ‘query’ field (your original input), ‘prediction’ which contains 'topIntent', 'intents', and 'entities'. Crucially, 'intents' is an array of intent results, and each intent contains a score. The 'entities' part can be very nuanced with numerous recognized entities, with differing levels of confidence depending on how ambiguous the language is.

Here are some code examples, focusing on common languages you'd typically use when interfacing with LUIS:

**Example 1: Python**

This snippet demonstrates how to extract all intent scores, a useful step if you need to perform a threshold check.

```python
import json

def parse_luis_verbose_response(json_response):
    """Parses a LUIS verbose response and extracts all intent scores."""
    data = json.loads(json_response)
    if 'prediction' in data and 'intents' in data['prediction']:
        intents = data['prediction']['intents']
        intent_scores = {}
        for intent, details in intents.items():
            intent_scores[intent] = details['score']
        return intent_scores
    else:
        return None

# Assume json_response_string is the raw JSON string obtained from the LUIS API
json_response_string = """
{
 "query": "book a flight to london tomorrow",
  "prediction": {
   "topIntent": "BookFlight",
   "intents": {
    "BookFlight": {
      "score": 0.9856564
    },
    "CancelFlight": {
     "score": 0.0122345
    },
    "CheckFlightStatus": {
        "score": 0.0021091
    }
   },
   "entities": [
      {
        "entity": "london",
        "type": "Destination",
        "startIndex": 15,
        "endIndex": 21,
        "resolution": {
         "values": [
          "London"
         ]
        },
          "score": 0.99999
      },
        {
        "entity": "tomorrow",
        "type": "DateTime",
        "startIndex": 22,
        "endIndex": 30,
        "resolution": {
          "values": [
            {
              "timex": "2024-06-11",
              "type": "date"
            }
          ]
        },
        "score": 0.99999
      }
    ]
  }
}
"""

intent_scores = parse_luis_verbose_response(json_response_string)

if intent_scores:
    for intent, score in intent_scores.items():
        print(f"Intent: {intent}, Score: {score}")
else:
    print("Could not parse intent scores")
```

**Example 2: Javascript (Node.js)**

This example focuses on extracting a specific entity, accounting for verbose data structure, and is a common scenario when processing LUIS results.

```javascript
function parseLuisVerboseEntities(jsonResponse) {
    try {
        const data = JSON.parse(jsonResponse);
         if (data.prediction && data.prediction.entities) {
            const destinationEntities = data.prediction.entities.filter(entity => entity.type === 'Destination');
            if (destinationEntities && destinationEntities.length > 0){
              //if there are multiple we will just take the first as a simplification
              return destinationEntities[0];
            }
            return null;

          } else {
                return null;
            }
        } catch (e) {
             console.error("error:",e)
        return null;
    }
}

const jsonResponseString = `{
 "query": "book a flight to paris tomorrow and then to rome",
  "prediction": {
   "topIntent": "BookFlight",
   "intents": {
    "BookFlight": {
      "score": 0.9856564
    },
    "CancelFlight": {
     "score": 0.0122345
    },
    "CheckFlightStatus": {
        "score": 0.0021091
    }
   },
   "entities": [
      {
        "entity": "paris",
        "type": "Destination",
        "startIndex": 15,
        "endIndex": 21,
        "resolution": {
         "values": [
          "Paris"
         ]
        },
         "score": 0.99999
      },
        {
        "entity": "tomorrow",
        "type": "DateTime",
        "startIndex": 22,
        "endIndex": 30,
         "resolution": {
          "values": [
            {
              "timex": "2024-06-11",
              "type": "date"
            }
          ]
        },
        "score": 0.99999
      },
      {
        "entity": "rome",
        "type": "Destination",
        "startIndex": 41,
        "endIndex": 45,
        "resolution": {
         "values": [
          "Rome"
         ]
        },
       "score": 0.99999
      }
    ]
  }
}`;

const destination = parseLuisVerboseEntities(jsonResponseString);
if (destination)
    console.log("Found Destination:", destination);
else
    console.log("Destination not found.");

```

**Example 3: C#**

This C# example, in a similar vein to python, will pull all of the entities and display the extracted entity and associated types.

```csharp
using System;
using System.Collections.Generic;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

public class LuisEntity {
    public string entity {get; set;}
    public string type {get; set;}
    public double score {get; set;}
}

public class LuisResponseParser {

    public static List<LuisEntity> parseVerboseEntities(string jsonResponse) {

        try {
             var data = JObject.Parse(jsonResponse);

            if (data["prediction"] != null && data["prediction"]["entities"] != null) {
                var entitiesArray = (JArray)data["prediction"]["entities"];

                var entityList = new List<LuisEntity>();

                 foreach (JObject entityObj in entitiesArray)
                {
                    LuisEntity entity = new LuisEntity
                    {
                        entity = entityObj["entity"].ToString(),
                        type = entityObj["type"].ToString(),
                        score = (double)entityObj["score"]

                    };
                     entityList.Add(entity);
                }
                return entityList;

                }
             else {
                return null;
            }

        } catch (JsonReaderException ex) {
                Console.WriteLine($"Error parsing JSON: {ex.Message}");
                return null;
        }

    }


    public static void Main(string[] args)
        {
           string jsonResponseString = @"{
             ""query"": ""book a flight to london tomorrow"",
              ""prediction"": {
               ""topIntent"": ""BookFlight"",
               ""intents"": {
                ""BookFlight"": {
                  ""score"": 0.9856564
                },
                ""CancelFlight"": {
                 ""score"": 0.0122345
                },
                ""CheckFlightStatus"": {
                    ""score"": 0.0021091
                }
               },
               ""entities"": [
                  {
                    ""entity"": ""london"",
                    ""type"": ""Destination"",
                    ""startIndex"": 15,
                    ""endIndex"": 21,
                    ""resolution"": {
                     ""values"": [
                      ""London""
                     ]
                    },
                    ""score"": 0.99999
                  },
                    {
                    ""entity"": ""tomorrow"",
                    ""type"": ""DateTime"",
                    ""startIndex"": 22,
                    ""endIndex"": 30,
                     ""resolution"": {
                      ""values"": [
                        {
                          ""timex"": ""2024-06-11"",
                          ""type"": ""date""
                        }
                      ]
                    },
                    ""score"": 0.99999
                  }
                ]
              }
            }";

          var entities = parseVerboseEntities(jsonResponseString);

            if (entities!=null)
            {
                foreach (var entity in entities)
                {
                Console.WriteLine($"Entity: {entity.entity}, Type: {entity.type}, Score: {entity.score}");
                }
            }
            else
            {
                Console.WriteLine("No Entities Found");
            }


         }
}
```

These examples highlight the core logic involved in parsing LUIS verbose responses, showing how you can tailor your code to the data structure provided. I recommend that you familiarize yourself with the LUIS API documentation thoroughly, as the specifics can sometimes change. Further, examining related academic papers in the field of natural language understanding and intent classification can greatly enhance your skills. Look into papers focusing on statistical models used in NLU and intent detection; this will improve your understanding of the internal workings of the models. Finally, I'd suggest exploring advanced books on natural language processing, and practical implementations to deepen your knowledge in building more sophisticated conversational systems, particularly those that focus on extracting information from textual data. Understanding the data structures, which verbose mode provides, will allow you to extract the detail required to move beyond simple applications and build more advanced solutions.
