---
title: "How can I compare downloaded JSON data to a GraphQL query result?"
date: "2024-12-23"
id: "how-can-i-compare-downloaded-json-data-to-a-graphql-query-result"
---

,  Funny enough, I remember back in '17, when we were migrating a legacy REST API to GraphQL, we faced a similar challenge daily. We had a suite of integration tests that relied on pre-existing json structures downloaded from the old endpoints, and suddenly we had this new graphql schema returning, well, graphql structured data. We needed a way to consistently validate that the data was semantically equivalent, even if the shape and format had shifted. Simply comparing strings wasn’t going to cut it; it was more nuanced than that. Here's how I approached the problem, and how you can too.

First, we need to acknowledge that comparing downloaded json and graphql results directly, as raw strings, is almost always a recipe for pain. GraphQL is designed to be flexible in response structure depending on the query; order of fields isn’t guaranteed. Also, JSON downloaded from an older REST API might be arbitrarily structured. We're essentially comparing apples to a potential fruit salad, with some of those fruits being slightly mislabeled! So what do we do? We need to normalize the data. Normalization here means transforming both datasets into a consistent, comparable structure *before* comparing the content.

The core principle revolves around reducing both the JSON and GraphQL responses into a canonical, sortable representation. I primarily utilize the following steps.

1.  **Parsing:** Both the downloaded JSON and the GraphQL response need to be parsed into native data structures, typically javascript objects or dictionaries, depending on your language. This is a non-negotiable first step because strings lack the structural awareness we require.
2.  **Filtering and Selection:** Your GraphQL query probably asked for specific fields. Your downloaded JSON might contain a lot more. To compare fairly, we should only consider the data points that correspond to each other, meaning we must filter out any irrelevant data. This might entail walking the resulting javascript object representation based on the fields in the GraphQL query, discarding other parts of the JSON if not requested.
3.  **Sorting:** The order of fields in objects shouldn’t be a factor. Therefore, we’ll need to sort objects based on keys. It's crucial to recursively perform this sort on nested objects to ensure consistency at all levels. Lists will need a slightly more complex handling, which I’ll get to.
4.  **Deep Comparison:** Once both representations are normalized, we compare these data structures recursively. Equality should be structural, not just shallow reference comparison.
5.  **Error Handling:** If a field exists in one data set but not in the other, or if the types don't match, it needs to be explicitly flagged. Not doing this is just asking for silent failures.

Now, to the code examples. Let’s assume we're operating within a javascript environment, for ease of demonstration.

**Example 1: Sorting Objects and Arrays**

This first snippet focuses on creating a generic sort function that handles objects recursively and has a special case for arrays to sort based on object identifiers, when available. In cases where a simple sort isn’t enough, you’d need a bit more domain-specific logic, of course.

```javascript
function deepSort(obj) {
    if (Array.isArray(obj)) {
        //attempt to sort objects by unique identifier, else resort to string sort.
        obj.sort((a, b) => {
          if (typeof a === 'object' && a !== null && typeof b === 'object' && b !== null) {
            if (a.id && b.id) {
              return String(a.id).localeCompare(String(b.id));
            }
          }
            return JSON.stringify(a).localeCompare(JSON.stringify(b));
        });

        return obj.map(deepSort);
    } else if (typeof obj === 'object' && obj !== null) {
        const sortedKeys = Object.keys(obj).sort();
        const sortedObj = {};
        sortedKeys.forEach(key => {
            sortedObj[key] = deepSort(obj[key]);
        });
        return sortedObj;
    }
    return obj;
}
```

This function handles recursive sorting and has a mechanism to attempt id-based sorting of arrays, a very useful real-world necessity when comparing API responses involving lists of objects. The usage is straightforward: just pass any data structure that needs sorting, including objects and arrays, and this will return a sorted representation.

**Example 2: Filtering JSON Data based on GraphQL Query**

Here is a function to simulate selecting only a subset of the downloaded JSON based on the fields you retrieved from the GraphQL query. We would need to extract what to filter from the GraphQL query first (I'll skip that here for clarity). Let’s say we have a list of fields that we care about. Note that this is simplified, and real-world implementations might be more complex.

```javascript
function filterJson(jsonData, fields) {
    if (typeof jsonData !== 'object' || jsonData === null) {
        return jsonData; // not a relevant object, return directly
    }
    if (Array.isArray(jsonData)) {
      return jsonData.map(item => filterJson(item, fields));
    }
    const filtered = {};
    for (const key of fields) {
        if (jsonData.hasOwnProperty(key)) {
            filtered[key] = filterJson(jsonData[key], fields[key] || []);
        }
    }
    return filtered;
}
```

This assumes that `fields` can be a nested structure of keys, mirroring the structure of a nested query, which provides for deeper filtering logic. In an actual implementation, parsing and deriving these "field paths" from the GraphQL query would require dedicated logic. But this illustrates the basic principle: only include what's relevant, given the query.

**Example 3: Deep Comparison with Type Checking**

Finally, here is a function to perform a deep comparison, with a few basic type checks added. I would encourage a more exhaustive approach in a production system.

```javascript
function deepCompare(obj1, obj2) {
    if (typeof obj1 !== typeof obj2) {
      return `Type mismatch: ${typeof obj1} vs ${typeof obj2}`;
    }
    if (typeof obj1 !== 'object' || obj1 === null || obj2 === null) {
       if (obj1 !== obj2) {
         return `Value mismatch: ${String(obj1)} vs ${String(obj2)}`;
       }
        return true;
    }

    if (Array.isArray(obj1) && Array.isArray(obj2)) {
      if (obj1.length !== obj2.length) {
        return `Array length mismatch: ${obj1.length} vs ${obj2.length}`;
      }
      for (let i = 0; i < obj1.length; i++) {
        const result = deepCompare(obj1[i], obj2[i]);
          if (result !== true) return `Array element mismatch at index ${i}: ${result}`;
      }
       return true;
    }

    const keys1 = Object.keys(obj1);
    const keys2 = Object.keys(obj2);

    if (keys1.length !== keys2.length) return `Object key length mismatch: ${keys1.length} vs ${keys2.length}`;

    for (const key of keys1) {
        if (!obj2.hasOwnProperty(key)) {
            return `Key mismatch: ${key} missing in second object`;
        }
         const result = deepCompare(obj1[key], obj2[key]);
         if (result !== true) return `Object value mismatch for key ${key}: ${result}`;
    }

    return true;
}
```

This function will return `true` if the structures are identical and will pinpoint differences with a descriptive string if they are not. Again, for a real-world setup, the error reporting mechanism could be more elaborate.

**Recommendations:**

For a deeper dive into JSON handling, I recommend the ECMAScript specifications on JSON parsing and serializing. Also, the book “Effective JavaScript” by David Herman, while focused on JavaScript, has several invaluable chapters on data structure manipulation. For a thorough understanding of GraphQL itself, the official documentation, of course, should be your first stop. Additionally, for more information on dealing with data structures, "Data Structures and Algorithms in JavaScript" by Michael McMillan may prove valuable. I also highly advise studying common data normalization patterns as described in database design literature. This isn't a trivial exercise and requires a good foundational understanding.

In practice, I integrated these steps into our testing pipeline. We created a test harness that would parse both the JSON from the old APIs and the data from GraphQL, normalize the responses using similar functions, and then perform this deep comparison. By doing so, we drastically reduced the complexity of validation and were able to deploy the new GraphQL API with a higher degree of confidence. It’s a process that, while not perfectly automated, can be made highly reliable when approached methodically. It worked for us, and I believe it will work well for you too.
