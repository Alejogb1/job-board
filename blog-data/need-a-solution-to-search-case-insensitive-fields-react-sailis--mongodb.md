---
title: "Need a solution to search case insensitive fields, REACT, sailis , mongodb?"
date: "2024-12-14"
id: "need-a-solution-to-search-case-insensitive-fields-react-sailis--mongodb"
---

alright, i've been there, staring at that screen, wondering why my search isn't finding anything when it *should*, because case sensitivity. it's a common pain point when you're juggling react, a backend with sails.js, and mongodb, especially when users start throwing in all sorts of capitalizations. been down this road more than once, trust me. let’s get down to brass tacks.

the core issue is that mongodb, by default, performs case-sensitive queries. when your react frontend sends a search term like “apple”, mongodb will not return a document with “Apple” or “APPLE” in a field. sails.js, being an abstraction over mongodb, inherits this behavior. so, we have to tackle this at different levels to get what we want, a case-insensitive search.

first, let’s look at the react part. honestly, there's not much to do there concerning case sensitivity, unless you want to start processing everything before sending it. that's more backend concern territory but we can clean up any white space and lower case the input before sending it to the api.

here's a simple example of how you might handle the user's input before sending it to the backend. note how we trim and lowercase user input in the search box using react hooks, i have been using this way for quite a long time because i had to do something with it on a project back in the day with similar issues, i recall we had to use graphql at some point with the sails api.

```jsx
import React, { useState } from 'react';

function SearchComponent() {
  const [searchTerm, setSearchTerm] = useState('');

  const handleSearchChange = (event) => {
    setSearchTerm(event.target.value);
  };

    const handleSearchSubmit = (event) => {
    event.preventDefault();
    const trimmedSearchTerm = searchTerm.trim().toLowerCase()
    // now you can use this `trimmedSearchTerm` when sending to the backend

    console.log('sending to backend:', trimmedSearchTerm);
    // call you api with the data
  };
    
  return (
    <form onSubmit={handleSearchSubmit}>
      <input
        type="text"
        placeholder="Search..."
        value={searchTerm}
        onChange={handleSearchChange}
      />
      <button type="submit">Search</button>
    </form>
  );
}

export default SearchComponent;
```

next, the sails.js backend. this is where the heavy lifting happens. we need to instruct mongodb, through sails.js, to perform a case-insensitive search.

sails.js uses waterlines as its odm to communicate with the database. we will use waterline’s `where` clause with a regular expression.

here’s how you might modify your sails.js controller to handle a case-insensitive search. i remember, we had some issues back in the company, we had this old database that had so many issues with upper and lower case strings that were similar in values, that is why this is so close to my heart.

```javascript
// api/controllers/YourModelController.js

module.exports = {
  search: async function (req, res) {
    const searchTerm = req.query.term; // Assuming the search term is passed as a query parameter
    
    if (!searchTerm) {
          return res.badRequest('No search term provided');
    }


    try {
      const results = await YourModel.find({
        where: {
          // assuming you are searching on the field called 'name'
            name: {
              contains: searchTerm,
               'options': 'i'
            }
          // you can repeat this for other fields
        },
      });
      return res.ok(results);
    } catch (err) {
      console.error("error on the query:", err)
      return res.serverError("there was an issue on the server");
    }
  },
};

```
note the `contains` keyword with the option `i`, that will trigger a regex behind the scenes and will make the search case insensitive. if the sails version or the mongodb connector is older you may have to use regex directly as in the example below

```javascript
    const results = await YourModel.find({
      where: {
        // assuming you are searching on the field called 'name'
        name: {
           'regexp': new RegExp(searchTerm, 'i')
        }
      // you can repeat this for other fields
      },
    });

```

this tells mongodb: "find all documents where the 'name' field contains the `searchTerm`, ignoring case". remember, this is a basic implementation, and real-world searches often have to consider more advanced filtering, pagination and other aspects.

now, a brief interlude: i once spent a whole weekend trying to debug a search feature like this, only to discover i had misspelled a field name in my sails.js model. it was like a digital version of looking for your keys while they're already in your hand. good times.

now, let's tackle the mongodb level. while sails.js handles most of the interaction, understanding how mongodb itself works with case-insensitive searches can be useful. the `i` option in the regular expression we use in sails is a mongodb feature. it allows us to do case insensitive searches by adding to the search term, the `i` modifier. however, there are a few other considerations depending on your specific situation. if your application requires very complex search capabilities with tokenization, stemming, etc. it is recommended to consider other tools.

if you are doing many complex string comparisons you can consider using an `index` on the fields. this will help to optimize the search and the queries. however, you should avoid regex operations on non-indexed fields, as it would cause a full collection scan.

if the fields you want to query are on a complex structure of nested documents, you should consider using `text` indexes in combination with the `$text` operator.

```javascript
// for example let us say that on your model you have a field called 'description' that is complex and you want to search for the term 'something' with case insensitivity and the 'description' has many embedded fields.

// in your controller you would do this:

  const results = await YourModel.find({
    $text: {
        $search: searchTerm,
          $caseSensitive: false
        }
  });

```

this is usually used in big documents that are more like a blog than a data table structure.

also for a more flexible search you can use full-text search. to set up full-text search on your mongodb collection, you have to create a text index like this:

```javascript
  db.yourcollection.createIndex(
    {
     name: "text",
      description: "text" //you can put many fields here
    },
       {
        weights: {
          name: 10, // the name field has more importance than description
          description: 5
        },
      }

  )
```

and then you can do a query like this:
```javascript
const results = await YourModel.find({
    $text: { $search: searchTerm },
  });

```

a useful thing to remember, while you can do case-insensitive searches, it's always better to be consistent in how you store your data. if you know you will never have a uppercase value, then you should transform it before saving it, this will ease the future queries and the overall data management. it could be done directly on the sails model before creation using the `beforeCreate` lifecycle method.

now, for resources, while there isn’t a specific book solely focused on this precise problem, i have found that diving into the official mongodb documentation is a great approach for this specific subject, especially sections on text indexes and regular expression queries. for sails.js, the official documentation, even though a bit scarce, is very complete for these cases, but in general, i like to check the source code of the frameworks, it can help to understand how things really works under the hood. then for react, the official documentation is also good.  if you want a broader view of the whole mern stack i would recommend "full stack web development with react" by anthony accomazzo, it has a good intro of all the aspects of the mern stack.

remember, the key is to understand how each part of your stack interacts with the others, this will save you future headaches. these tips i've given, are the result of countless hours of trial and error, so use them wisely. good luck, and may your searches always be fruitful!
