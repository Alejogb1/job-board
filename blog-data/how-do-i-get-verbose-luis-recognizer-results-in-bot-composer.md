---
title: "How do I get verbose LUIS recognizer results in Bot Composer?"
date: "2024-12-16"
id: "how-do-i-get-verbose-luis-recognizer-results-in-bot-composer"
---

Alright, let's talk about getting those verbose LUIS recognizer results in Bot Composer – something I've definitely spent time wrestling with in past projects, so I understand the need for granular control. It’s less a matter of wrestling with settings and more about understanding how the underlying LUIS query is formed and how to access its full response structure, rather than just what composer exposes by default.

By default, Bot Composer simplifies the LUIS response to provide what's often needed for conversational flow: the top intent and its entities. But, quite frequently, you need the entire JSON payload returned by LUIS – the scored intents, all entities with their various resolutions, and other metadata that can be incredibly useful for fine-grained control or advanced logic. The challenge, as I’ve seen repeatedly, is that Bot Composer provides an abstraction, making it necessary to dive a bit deeper to extract this richer data.

The primary issue is the built-in LUIS recognizer component in Composer. It's designed for simplicity and assumes the most common use case. So it presents a streamlined response. Accessing the detailed, verbose output requires us to go a step beyond the visual designer. We're essentially peeling back a layer of abstraction and interacting with the underlying data model. This involves a bit of direct scripting using adaptive expressions, which is where the power, and also some of the complexity, lives.

Here's how I’ve typically handled this in practice, breaking it down into key steps and code examples:

**Step 1: Accessing the LUIS Result Directly:**

The core idea here is to bypass the standard Composer variables for intent and entities, and access the raw LUIS result directly. When the LUIS recognizer executes, it populates a variable that contains the full JSON response. We need to find where this information lives. In my experience, this is available as a property on the `turn.recognized.luisResult` object. Specifically, we want the `luisResult.result` property – this is where the complete JSON payload from LUIS is held.

```csharp
// Example 1: Accessing the raw LUIS result in an adaptive expression.
// This expression retrieves the full LUIS result and stores it in the 'luisFullResult' property
{
  "type": "Microsoft.SetProperty",
  "property": "luisFullResult",
  "value": "turn.recognized.luisResult.result"
}
```

The snippet above, when placed within a Bot Composer dialog step as a *Set property* action, will store the complete JSON from the LUIS API into the `luisFullResult` memory variable. You can then access it in further steps.

**Step 2: Parsing the JSON Response:**

Once you have this full result, you'll typically need to parse it. The `json()` function within adaptive expressions is the tool for that. This allows you to navigate the JSON structure and access specific elements. I have frequently found I need to access the various `entities` structure and pick among the various subtypes of entities, their resolution data, and so forth.

```csharp
// Example 2: Parsing the JSON result and extracting specific information.
// This example assumes the result contains a "entities" array and extracts
// all the "datetime" entities with their resolution values.
{
    "type": "Microsoft.SetProperty",
    "property": "datetimeEntities",
    "value": "json(luisFullResult).entities[? type == 'datetimeV2'].resolution[0].value"
  }
```

In this snippet, we’re using a filter expression (the `[? type == 'datetimeV2']` part) to select only the datetime entities and then, further down the tree, picking out the resolved values. This uses the adaptive expression JSON handling capabilities and allows for very refined retrieval of data from the result. `resolution[0]` assumes the simplest resolution is wanted; you might want `resolution` which would return an array of possible resolutions. This will depend on what your specific use case requires.

**Step 3: Utilizing the Data in Your Bot Logic:**

After you've parsed the JSON, you can use this detailed data to drive your bot's behavior, trigger conditional logic, or tailor responses. This is where that granular control comes in. It's important to remember that you are now directly using the data provided by LUIS, not an abstract representation. This will let you inspect a more comprehensive set of options that are not provided by composer out of the box.

```csharp
// Example 3: using the extracted datetime in a conditional.
// This example will set a boolean to true if any datetime entities are available.
{
  "type": "Microsoft.SetProperty",
  "property": "hasDateTimeEntities",
  "value": "if(empty(datetimeEntities), false, true)"
}
```

Here we are checking if the `datetimeEntities` variable is empty and if so, we set a boolean, `hasDateTimeEntities` to false; if it is not empty, it is set to true. This value can now be used in the remainder of the dialog.

**Practical Considerations:**

*   **Error Handling:** It's essential to add robust error handling to your expressions. For example, checking if the `luisFullResult` is actually available before trying to parse it, or adding a try-catch using the `try()` and `catch()` functions in the adaptive expression language. `null` is usually the result when data is not present, and relying on this might introduce issues later.
*   **Understanding the LUIS Schema:** Having a firm grasp on the schema of the LUIS response JSON is crucial. You’ll want to examine your LUIS app's JSON response examples (visible on your LUIS portal) to craft precise expressions. Understanding your LUIS response is the single most important thing when trying to use this technique.
*   **Adaptive Expressions:** Familiarize yourself with the adaptive expressions documentation. They can perform more complex operations than just navigating JSON, like formatting strings, conditional logic, working with collections, and so forth. The entire flexibility of accessing the LUIS result in this way hinges on mastering this tool.
*   **Debugging:** When debugging, I found it useful to temporarily display the `luisFullResult` variable in the bot output using the *Send a response* action to see what LUIS is actually returning. This can greatly assist in debugging your adaptive expressions.

**Recommended Resources:**

*   **Microsoft's Adaptive Expressions Documentation:** The official documentation for adaptive expressions is the absolute best place to start. It covers all the functions, syntax, and capabilities.
*   **"Building Conversational Bots" by Microsoft:** This book provides in-depth knowledge of bot development concepts, including working with the bot framework and related technologies, specifically how the LUIS recognizer is used in a broader setting, and therefore offers a context to this issue.
*   **Microsoft LUIS documentation:** It's important to fully understand how the LUIS service returns data. Review the official documentation which is available online, this documentation includes examples of the entire payload of responses and how to utilize them.

In closing, while Bot Composer is great for rapid bot development, getting verbose LUIS results requires a bit of a deeper dive. Understanding adaptive expressions and how to navigate JSON is key. By accessing the underlying `turn.recognized.luisResult.result`, parsing the JSON, and using the extracted data in your bot logic, you can take full advantage of the rich information available from LUIS and create more sophisticated and accurate conversational experiences. This methodology was essential in several of my past projects, providing the level of control needed to deliver high-quality solutions. And, as with any technical challenge, a solid understanding of the underlying system, and consistent testing will be crucial in achieving your goals.
