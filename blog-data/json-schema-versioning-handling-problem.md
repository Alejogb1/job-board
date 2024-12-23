---
title: "json schema versioning handling problem?"
date: "2024-12-13"
id: "json-schema-versioning-handling-problem"
---

 so json schema versioning yeah been there done that got the t-shirt multiple times trust me it’s a headache but a solvable one Let's unpack this from a purely practical point of view like one engineer talking to another no fluff no marketing jargon just real-world stuff

Right off the bat you need to understand that json schema versioning is not something the schema spec directly provides you Its up to you the developer to figure out a system that works for your specific situation and believe me there is no one size fits all solution I've tried them all and all of them break eventually in some way or another That's just how software development rolls

Let’s tackle the core issue You have a json schema you use it to validate your data fine now your data evolves your schema needs to evolve too But you still have old data lying around that still uses the old schema you can’t just drop that can you Now the problem is how do you handle all of that without tearing your hair out

My first big project dealing with this was back in my early days at a company that made some sort of system for tracking inventory for a warehouse it was a simple thing at first just a few fields for product name id quantity but then the business users started asking for more details like expiry dates batch numbers different storage locations for the same product it was schema change chaos every other week

We didn't have a plan for schema versioning it was like building a house on a quicksand we changed the schema every time there was a new requirement just validated against the new schema and boom we would get errors everywhere whenever we got old data that didn't validate or we needed to implement a ton of ifs and try-catches on our data processing code

First mistake we made we didn't version our schemas period that was a rookie mistake you live and learn I guess we figured that updating the schema meant all the data had to conform to it immediately but then we realized that wasn't realistic data doesn't just change overnight we had to go back and fix that and it wasn't pretty

So what are the options you have and what did I do well first off forget trying to modify existing schemas that's a recipe for disaster treat each schema version as an immutable artifact when you need to make a change create a new schema version

Now how do you actually do that Well there are a few common strategies I've used with varying degrees of success

First the classic version number in the schema itself something simple like adding a `$schemaVersion` property to each of your schemas sounds basic but it works for simple cases

```json
// Schema version 1
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "$schemaVersion": "1.0",
  "type": "object",
  "properties": {
      "productName": { "type": "string" },
      "productId": { "type": "integer" }
  },
  "required": ["productName", "productId"]
}
```

```json
// Schema version 2 with batch number
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "$schemaVersion": "2.0",
  "type": "object",
  "properties": {
    "productName": { "type": "string" },
    "productId": { "type": "integer" },
    "batchNumber": { "type": "string" }
  },
  "required": ["productName", "productId", "batchNumber"]
}
```

```json
// Schema version 3 with optional expiryDate
{
    "$schema": "http://json-schema.org/draft-07/schema#",
    "$schemaVersion": "3.0",
    "type": "object",
    "properties": {
        "productName": { "type": "string" },
        "productId": { "type": "integer" },
        "batchNumber": { "type": "string" },
      "expiryDate": { "type": "string", "format": "date-time"}
    },
  "required": ["productName", "productId", "batchNumber"]
}
```

The idea is when you process incoming json data you always check the `$schemaVersion` property then decide which schema to use for validation and data transformation The code that consumes the data has to check it and act upon it differently it can be implemented using switch case or an object that acts as a lookup table to handle the logic for different schemas this is more flexible than using ifs in my opinion

This approach works surprisingly well for simpler applications the downside is that you have to write the version logic in your code which adds to the maintenance overhead Every time you add a schema you need to remember to update that part of the code otherwise things will break in unpredictable ways

Another thing that can happen when you keep adding more and more schemas is that it can become hard to track and manage all of them I found that it's easier to keep them in separate files and use a naming convention which indicates their version I tend to use this when the application is in an early stage of development so I can modify the schemas in each file and test them individually before making any changes to the main code base this also makes it easy to discard an old schema without messing the code base

For more complex scenarios you may consider using a more structured approach like defining a schema registry A registry is basically a database that stores all your schemas and allows you to look them up by their version or some other identifier like the schema’s type or a unique identifier For this though you will have to build your own or use something that is already available

This can be particularly useful if you're using a message queue or a microservices architecture where multiple services need to validate messages against different schemas This approach is particularly powerful as it centralizes the logic of your schemas and makes the system more maintainable but this one is the most complex and time consuming solution to implement and manage

You may also need to consider data migrations when you need to change an existing field in your schema you will have to write scripts that convert your old data to the new schema this can be particularly challenging if you are modifying a lot of data in place this is why immutable schemas are good you can validate the data with the old schema and then use the transformation methods to create a new data object that is then validated with the new schema this works well for data lake applications

Finally it's essential to have a clear communication strategy within your team every time a schema changes or a new schema is added someone should be notified to ensure that everyone is on the same page this is one area where I've seen multiple projects fail because teams end up using different versions of schemas without realizing it and this brings a lot of inconsistency into the system

Oh one more thing if you are building your own schema registry be prepared to debug a lot I remember one time when I was working on this I thought that the issue was on the schema that was being processed instead I spent days debugging it until I found out that the issue was with the way we configured the routing on our schema registry servers I swear sometimes dealing with network issues is like trying to find a needle in a haystack or as I usually say debugging is like an easter egg hunt the egg can be anywhere haha  i'm done with the joke sorry

As for resources to understand more in detail about schema handling I found that the draft-07 json schema specification is a good place to start also the json schema website has good documentation on the subject and there is a very good book called designing data intensive applications that covers topics related to data modeling schema evolution and many other issues that can help you design a better system

Anyways that's my 2 cents on the topic it's a hard problem but it can be solved with the right approach don't be afraid to experiment and see what works best for your situation and remember versioning everything is key to keeping your sanity so yeah good luck you will probably need it
