---
title: "How do I access LUIS recognizer results with verbose set to true?"
date: "2024-12-23"
id: "how-do-i-access-luis-recognizer-results-with-verbose-set-to-true"
---

Alright, let's talk about accessing LUIS (Language Understanding Intelligent Service) recognizer results when you've set the `verbose` option to true. It's a common point of confusion, and I’ve certainly spent my fair share of time debugging it. The default, non-verbose output can feel pretty limited once you start needing more nuanced information about what LUIS is picking up. I remember back in the early days of building a conversational AI for a customer support system, we were getting some bizarre intent classifications and it took us a while to understand the underlying causes. Setting `verbose` to true was a huge help, but accessing the data properly was the next hurdle.

When the `verbose` flag is enabled, the JSON response from LUIS includes significantly more data than when it’s set to false. This data is immensely useful for troubleshooting, feature engineering, and implementing fallback mechanisms. Think of it as unlocking a more detailed audit trail of LUIS’s processing. Instead of just seeing the top intent and a few entities, you get a breakdown of all intents and their scores, which utterances contributed to each score, the predicted entities, resolutions, and more. It’s a goldmine but it requires a more specific approach to access the results.

The key thing to understand is that with `verbose=true`, your top-level response doesn't directly hold the structured intent and entity information in the same way it does with `verbose=false`. Instead, it contains a list of utterances, each with its own corresponding processing results. This means that, conceptually, you’re looking at a 1:n mapping between user input and LUIS analysis. There’s an array named `utterances`, and inside each `utterance` object you’ll find the meat of the matter: `intents` (an array of intent scores), `entities` (an array of recognized entities with various associated details), and other potentially important pieces of information like the `compositeEntities`.

Let's illustrate with a few practical examples using javascript, which is pretty typical in this space. Assume we've got the raw LUIS response stored as a JSON object named `luisResponse`.

**Example 1: Extracting the Top Intent with Verbose Output**

```javascript
function getTopIntentVerbose(luisResponse) {
    if (!luisResponse || !luisResponse.utterances || luisResponse.utterances.length === 0) {
        return null;
    }

    // Assuming we only care about the first utterance processed.
    const firstUtterance = luisResponse.utterances[0];

     if (!firstUtterance.intents || firstUtterance.intents.length === 0){
         return null;
     }

    //sort intents in descending order of their score.
    const sortedIntents = firstUtterance.intents.sort((a,b) => b.score-a.score);

    return sortedIntents[0].intent;
}

// Example Usage
// Assume luisResponse is your actual LUIS response with verbose=true
const topIntent = getTopIntentVerbose(luisResponse);
if (topIntent) {
  console.log("Top Intent:", topIntent);
} else {
  console.log("Could not determine top intent");
}
```

This snippet shows how to access the top intent. We first grab the first utterance from the array, then sort the array of intents based on their scores and retrieve the first one. It demonstrates that you don't directly access `topScoringIntent`. Instead, you work your way through the `utterances` array, and within each utterance, the `intents` array, ordering them by score to find the top one.

**Example 2: Extracting Entities with Details**

```javascript
function getEntitiesVerbose(luisResponse) {
    if (!luisResponse || !luisResponse.utterances || luisResponse.utterances.length === 0) {
      return [];
    }

  const firstUtterance = luisResponse.utterances[0];
    if (!firstUtterance.entities || firstUtterance.entities.length === 0) {
        return [];
    }

    const entities = firstUtterance.entities.map(entity => {
      return {
        entity: entity.entity,
        type: entity.type,
        resolution: entity.resolution,
        startIndex: entity.startIndex,
        endIndex: entity.endIndex
      }
    });


    return entities;
}

// Example Usage
const extractedEntities = getEntitiesVerbose(luisResponse);
if(extractedEntities.length > 0){
    console.log("Extracted Entities", extractedEntities);
}else{
    console.log("No entities extracted");
}
```

This example focuses on entities. It shows how to navigate the verbose response to pull out individual entities along with their associated resolution, which contains further normalized data. You'll notice how you obtain more than just the name of an entity. The `startIndex` and `endIndex` can be important if you need to precisely locate and manipulate the entity within the original user input string.

**Example 3: Handling Composite Entities**

```javascript
function getCompositeEntitiesVerbose(luisResponse) {
  if (!luisResponse || !luisResponse.utterances || luisResponse.utterances.length === 0) {
        return [];
  }
    const firstUtterance = luisResponse.utterances[0];
     if (!firstUtterance.compositeEntities || firstUtterance.compositeEntities.length === 0) {
        return [];
    }

    const compositeEntities = firstUtterance.compositeEntities.map(entity => {
        return {
            entity: entity.value,
            children: entity.children.map(child => {
                return {
                    entity: child.entity,
                    type: child.type,
                    resolution: child.resolution,
                    startIndex: child.startIndex,
                    endIndex: child.endIndex
                }
            })
        }
    });


  return compositeEntities;
}

//Example Usage
const compositeEntities = getCompositeEntitiesVerbose(luisResponse);
if(compositeEntities.length > 0){
    console.log("Composite Entities:", compositeEntities);
}else{
   console.log("No composite entities found");
}
```
This code demonstrates extraction of composite entities, if your LUIS app uses them. This function navigates the nested structure of `compositeEntities`, showing you how to access the individual component parts, along with their types and resolutions. You can see the parent composite entity object contains the child entities. The structure may vary based on the composite entity definition from LUIS.

**Important Considerations**

*   **Error Handling:** Always include robust null and undefined checks when handling potentially complex JSON structures like these. I’ve often seen production errors arise from naive assumptions about the structure of the response, particularly when experimenting with different LUIS configurations.

*   **Rate Limiting:** Be mindful of your LUIS query limits. You may want to implement caching of the LUIS responses, particularly if you're handling heavy load.

*  **Utterance Processing**: I’ve shown the most basic case of accessing the first utterance but you will need to consider how to handle multiple utterances when they exist. For most cases, this is not necessary but you may need to do some filtering based on score or other criteria.

*  **Data Validation**: Always cross-reference the data you get from LUIS. Do not simply assume that the intent or entities it extracts are correct. Validate against your expected results, or implement more sophisticated techniques to deal with potential errors.

*   **Documentation**: The official Microsoft documentation for LUIS is your go-to reference, but I personally found "Natural Language Processing with Python" by Steven Bird, Ewan Klein, and Edward Loper a very useful background to the broader field, even if the book doesn't specifically address LUIS. For a deeper dive into AI model training, I'd also recommend "Deep Learning with Python" by François Chollet. While they aren't solely dedicated to LUIS, they give you a proper conceptual foundation that will drastically improve your understanding of what’s happening under the hood.

Remember, debugging a complex system like a LUIS-powered application is often an iterative process. You'll often need to experiment with various access patterns, logging, and validations to nail down the exact data you need. Don't be afraid to add breakpoints, use your browser's debugging tools, or add console logs to examine each level of that `utterances` array in detail. Using the `verbose` setting is just the start; truly harnessing the power of the extra information is what makes the difference between a basic implementation and a robust, intelligent application. Good luck!
