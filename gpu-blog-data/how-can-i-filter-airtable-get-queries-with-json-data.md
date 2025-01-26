---
title: "How can I filter Airtable GET queries with JSON data?"
date: "2025-01-26"
id: "how-can-i-filter-airtable-get-queries-with-json-data"
---

Airtable's API, while powerful, doesn't directly interpret complex JSON within its query parameters for filtering. Instead, filtering operations must be expressed through Airtable's own formula language, which is embedded within the `filterByFormula` parameter of GET requests. I encountered this limitation while building an inventory management system where I needed to dynamically filter records based on product attributes stored as nested JSON strings in a single Airtable column. The challenge then became parsing this JSON data and utilizing it to construct a compatible Airtable formula for the `filterByFormula` parameter.

The fundamental approach is to translate JSON filter criteria into Airtable's formula syntax. Consider, for instance, a table of products where each record includes a "Specifications" field containing a JSON string, such as: `{"color": "red", "size": "large", "material": "cotton"}`. To retrieve all products that are red, we cannot pass `{"color": "red"}` directly to Airtable. We need to construct a formula that evaluates this JSON, parses it and compares the result. Airtable's `JSON_VALUE` function becomes essential here. This function allows retrieval of values from within a JSON string using dot notation, allowing you to target a specific key within the JSON data in a record.

Here's a breakdown of how this operates: the basic syntax for the filter would be `JSON_VALUE({Specifications}, '$.color')="red"`.  This formula does the following: first, it accesses the `Specifications` column value; it then extracts the value associated with the `color` key from the JSON string, effectively making it comparable with a string like `"red"`.  This translated string is sent as the `filterByFormula` parameter of the GET request.

The difficulty arises when dealing with more complex filtering needs. For example, when requiring a combination of conditions (red *and* large) or when you need to consider a dynamic number of filter fields. There is no single function to handle dynamic parsing of all the data you might have in your JSON string, you must know what keys to parse before building the filter. You need to parse, and dynamically build the `filterByFormula` string prior to making the GET request.

Let’s look at a few practical code examples.

**Example 1: Basic single-criteria filtering**

```javascript
const apiKey = 'YOUR_AIRTABLE_API_KEY';
const baseId = 'YOUR_AIRTABLE_BASE_ID';
const tableName = 'Products';

const fetchFilteredProducts = async (filterColor) => {
    const formula = `JSON_VALUE({Specifications}, '$.color') = "${filterColor}"`;
    const url = `https://api.airtable.com/v0/${baseId}/${tableName}?filterByFormula=${encodeURIComponent(formula)}`;

    const headers = {
        'Authorization': `Bearer ${apiKey}`,
    };

    try {
        const response = await fetch(url, { headers });
        const data = await response.json();
        if(data.records) return data.records;
        return [];

    } catch (error) {
        console.error('Error fetching data:', error);
        return [];
    }
};

// Example usage
fetchFilteredProducts("blue")
  .then(records => {
        console.log("Filtered Products:", records);
  })
  .catch(error => {
    console.error("Failed to fetch:", error);
  });
```

This example constructs a simple Airtable formula that selects records where the `color` key within the `Specifications` JSON matches the `filterColor` argument, which is passed as a parameter to the `fetchFilteredProducts` function. The `encodeURIComponent` function is crucial because the formula can contain special characters, which must be encoded for a URL. Note how the value for the comparison must be wrapped in quotes if it is a string.

**Example 2: Combining multiple criteria with 'AND'**

```javascript
const apiKey = 'YOUR_AIRTABLE_API_KEY';
const baseId = 'YOUR_AIRTABLE_BASE_ID';
const tableName = 'Products';

const fetchFilteredProducts = async (filters) => {
    const formulaParts = [];
    for (const key in filters) {
        const jsonPath = `$.${key}`;
        const escapedValue = String(filters[key]).replace(/"/g, '\\"');
        formulaParts.push(`JSON_VALUE({Specifications}, '${jsonPath}') = "${escapedValue}"`);
    }
    const formula = formulaParts.join(' AND ');

    const url = `https://api.airtable.com/v0/${baseId}/${tableName}?filterByFormula=${encodeURIComponent(formula)}`;

    const headers = {
        'Authorization': `Bearer ${apiKey}`,
    };

    try {
        const response = await fetch(url, { headers });
         const data = await response.json();
        if(data.records) return data.records;
        return [];

    } catch (error) {
        console.error('Error fetching data:', error);
        return [];
    }
};

// Example usage
fetchFilteredProducts({ color: "red", size: "large" })
  .then(records => {
        console.log("Filtered Products:", records);
    })
  .catch(error => {
        console.error("Failed to fetch:", error);
    });

```

This example demonstrates filtering by multiple criteria. It iterates over the passed `filters` object and constructs individual formula parts for each key-value pair, which are combined with an `AND` operator. This allows the API call to only return records that satisfy all conditions in the object. The example includes a measure for escaping double quotes in the value, using a replace regex, as these may cause problems with the Airtable filter. Note this strategy assumes an 'AND' filter condition between the conditions in the object. You would need to modify this method to handle 'OR' cases, should that be necessary.

**Example 3:  Handling numeric or boolean comparison**

```javascript
const apiKey = 'YOUR_AIRTABLE_API_KEY';
const baseId = 'YOUR_AIRTABLE_BASE_ID';
const tableName = 'Products';

const fetchFilteredProducts = async (filters) => {
    const formulaParts = [];
    for (const key in filters) {
        const jsonPath = `$.${key}`;
        let value = filters[key];
      
      //Check if value is a number or boolean, if so don't wrap it in quotes
        if (typeof value === 'number' || typeof value === 'boolean') {
            formulaParts.push(`JSON_VALUE({Specifications}, '${jsonPath}') = ${value}`);
        } else {
          const escapedValue = String(value).replace(/"/g, '\\"');
            formulaParts.push(`JSON_VALUE({Specifications}, '${jsonPath}') = "${escapedValue}"`);
        }
    }
    const formula = formulaParts.join(' AND ');


    const url = `https://api.airtable.com/v0/${baseId}/${tableName}?filterByFormula=${encodeURIComponent(formula)}`;

    const headers = {
        'Authorization': `Bearer ${apiKey}`,
    };

    try {
        const response = await fetch(url, { headers });
        const data = await response.json();
         if(data.records) return data.records;
        return [];
    } catch (error) {
        console.error('Error fetching data:', error);
        return [];
    }
};

// Example usage with numbers and boolean
fetchFilteredProducts({ price: 25.99, inStock: true})
  .then(records => {
    console.log("Filtered Products:", records);
  })
  .catch(error => {
        console.error("Failed to fetch:", error);
  });
```

This third example extends the previous one to also handle comparison of numeric and boolean values within the JSON. It includes a check on the `typeof` value and does not wrap the value in quotes if the type is a number or a boolean. Without this, the filter will often fail as string comparison will be used for the number. For instance, trying to filter by `price: 25.99` with the previous method would be like comparing the string '25.99' which is unlikely to succeed. Note that `JSON_VALUE` will return a string, which is why a boolean and number must be compared to an un-quoted value to succeed.

It is important to be mindful of the limits of Airtable formulas. Complex logic such as the use of nested AND/OR groupings is not supported. Also, Airtable formulas can be lengthy and may not be suitable for highly complex JSON objects. In my previous inventory project I found that it was often more efficient to query for a larger subset of data than to attempt to parse an overly complex JSON object using Airtable formulas, and to then filter further in client-side code. It all depends on the nature and scale of the data being processed. For very large datasets, it may be advisable to use a separate data store more suitable for complex queries and to sync to Airtable.

For further exploration into this topic, I would recommend reviewing the official Airtable API documentation, particularly the sections pertaining to the `filterByFormula` parameter. Also, I'd suggest reading up on Airtable's specific formula syntax, which is well-documented on their site. Lastly, researching examples of common Airtable formulas via third-party platforms or online communities can provide many practical examples and help inform a tailored solution for your specific use case. Understanding the nuances of JSON parsing within Airtable’s limitations will enable a more effective integration between your data and the Airtable platform.
