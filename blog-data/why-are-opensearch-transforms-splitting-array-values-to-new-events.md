---
title: "Why are Opensearch transforms splitting array values to new events?"
date: "2024-12-15"
id: "why-are-opensearch-transforms-splitting-array-values-to-new-events"
---

let's tackle this, i've seen this behavior plenty of times, and it always boils down to how opensearch transforms handles arrays. basically, it's not an error but a designed behavior.

when opensearch transform encounters a document with a field containing an array, it doesn't treat the whole array as a single value to be copied or manipulated. instead, it interprets each element of that array as a separate value that needs processing, which effectively causes it to create new events per element. this is fundamentally linked to how transform works, it is designed for a flat structure and array type fields violate that structure.

think of it like this, the transform pipeline processes documents sequentially, step by step. it iterates through each document and for each field, transform executes specified operations. when it encounters an array field, it loops through every element of that array. each element is then treated as a separate event, and any subsequent processing steps, such as aggregations or filtering, are applied to each of those new events. so the transform splits this single document containing an array into multiple documents, each containing a single element.

i've personally encountered this specific issue a lot especially when dealing with log data. for example, i had a system logging multiple user actions within a single json document, structured like `{"timestamp": "...", "user": "...", "actions": ["login", "add_item", "logout"]}`. i needed to count the number of actions per user, and the transform split every single event with three actions into three separate documents. this made the aggregation result look all wrong, since i did not know what to expect at the time.

it tripped me up the first time i saw it. initially, i thought it was a bug or a misconfiguration. i mean, i expected the whole `actions` array to be treated as a single value, not as a source for new documents. but after careful examination of the documentation and some experiments, it became clear that the transform is actually working as designed.

this kind of design choice, splitting arrays, is intentional. it's made to handle complex scenarios where you might need to aggregate data based on elements within those arrays. for instance, if you're tracking user actions and want to see how many times each specific action was performed, it's essential to have those actions split into separate events. think of e-commerce tracking a user's cart content. you have a user object and an array of items in that cart object. having each item as a separate document allows easy and fast aggregation on different item properties.

however, if you're not looking to do this kind of aggregation it is a problem. when you need the whole array as a single value it leads to unexpected output. this is where you need to plan your transforms carefully to avoid unwanted split behaviour.

here's a simple example of what the document split might look like. imagine we have an input document such as this:

```json
{
  "id": 1,
  "user": "john",
  "items": ["apple", "banana", "orange"]
}
```

after the transform, with the default behaviours, we'll have 3 new documents, each like:

```json
{
 "id": 1,
 "user": "john",
 "items": "apple"
}
```
```json
{
 "id": 1,
 "user": "john",
 "items": "banana"
}
```
```json
{
 "id": 1,
 "user": "john",
 "items": "orange"
}
```
as you can see, our single document generated three different documents.

so, how do we handle this in practice? there isn’t a silver bullet that magically makes opensearch treat arrays as a whole in transform, but here are a few strategies depending on what the user needs.

1.  **avoiding the split at the source:** the easiest approach, if possible, is to adjust your data source to not include arrays. if you can change the data ingestion pipeline to send a single value instead of an array, you'll bypass this issue altogether. this is not always feasible, of course, since often we must work with existing structures. for example, using a logstash configuration to flatten arrays before passing to opensearch.
2.  **using scripting to keep arrays whole**: if you absolutely must work with arrays but need to keep them together within the transform, you can use painless scripting. scripting allows a way to handle values before the default transform behavior. you'd need to use a script to combine the array into a string before the transform is applied. it’s a bit hacky, but sometimes necessary. for instance using the following snippet as a transform script:

```painless
String[] arr = ctx._source.items;
String joined = String.join(",", arr);
ctx._source.items = joined;
```
here's another example, in case you have more complex objects inside your arrays:

```painless
if (ctx._source.items != null) {
    def items = ctx._source.items;
    def itemsString = new ArrayList();
    for(def item : items){
      itemsString.add(item.toString())
    }
    ctx._source.items = String.join(";", itemsString);
}
```
this script will convert all the objects inside the array to strings and then join them using a semicolon.

these scripts will effectively flatten your array into a string, which allows the transform to handle it as a single value. keep in mind that if the objects in the array are very large or you need to query specific properties on them, this strategy won't work. you can also use json.stringify() method if you want to keep them as a json string.
3. **post-transform aggregation:** often, the split might actually be what you need but if you just want to count the number of distinct values, you have to process it in an extra step. after running the transform, you can do another aggregation which counts the number of elements created by your transforms. this method might require extra resources and is not as optimal as avoiding the split altogether but it’s the simplest method when your target is counting the size of arrays and you cannot avoid the split.
4. **restructuring with aggregations:** if you're dealing with nested structures, you might need to use aggregations within the transform to restructure your data and avoid the implicit split from arrays. this would require building a pipeline with different aggregation steps, and it can get quite complex, but it offers better control over the output structure.

in my experience, i've tried all these approaches at different times. there's rarely one size fits all solution. i think you should first consider the possibility of restructuring your ingestion pipeline to avoid using arrays altogether. i mean, "why did the array cross the road?", "because it was already flattened". but if that's not an option, scripting is a powerful tool that can handle more complex scenarios.

for further reading, i'd recommend looking into the opensearch documentation specifically the part on transforms. also there’s the "data intensive text processing with mapreduce" by jimmy lin and chris dyer which while not about opensearch, touches on map reduce patterns for transforming data, and some of those principles apply here as well. another book that’s useful is "seven databases in seven weeks" by eric redmond and jim wilson. it’s a broader view of different db solutions and some of the challenges in processing data, and it is great for gaining a better understanding of data processing in general.

i hope this clears things up. let me know if you have more specific scenarios or questions. i've seen a lot of weird things over the years, and it always comes down to the details.
