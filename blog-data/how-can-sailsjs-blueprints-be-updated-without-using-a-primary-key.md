---
title: "How can Sails.js blueprints be updated without using a primary key?"
date: "2024-12-23"
id: "how-can-sailsjs-blueprints-be-updated-without-using-a-primary-key"
---

Okay, let's tackle this. It’s a question that definitely surfaces once you start pushing Sails.js beyond its initial scaffolding. I've personally bumped into this situation during a project where we were migrating legacy data, where relationships were identified by unique combinations of fields rather than a conventional primary key, and frankly, relying solely on Sails.js’ default blueprint routes felt limiting.

The core issue lies in the way Sails.js blueprints are structured – they inherently expect a primary key to uniquely identify a record for operations like update or delete. By default, this is an auto-incrementing integer id, often named `id`. When you deviate from this model, direct usage of `/model/id` routes won’t work. We need a way to tell Sails.js, "Hey, this set of attributes is *how* I identify this record."

The default blueprint update route, `PUT /:model/:id`, relies on the implicit assumption that you have a single, unambiguous primary key. If you try to use it with, say, a unique combination of fields like `productCode` and `versionNumber` as the key, you’ll encounter a failure or an update to an unintended record (depending on the database setup). This is because the `id` segment in the URL is treated by Sails.js as the single primary key to lookup on. So, how can we update records when our identification mechanism is not a single primary key?

We'll need to bypass or augment Sails.js' built-in blueprints for this. Essentially, we need to create a custom action in our controller that performs the update using the desired set of fields to locate the record. Here’s how we approach it:

**1. Custom Controller Action:**

The most straightforward way to achieve this is to create a custom controller action. I generally prefer this approach because it keeps the logic self-contained and easier to manage. You avoid directly modifying the core framework, which, for most use cases, is the best practice.

Here's a code snippet that illustrates this approach:

```javascript
// api/controllers/ProductController.js

module.exports = {

  updateByProductCodeAndVersion: async function(req, res) {
    const productCode = req.param('productCode');
    const versionNumber = req.param('versionNumber');

    if (!productCode || !versionNumber) {
      return res.badRequest({ error: 'Both productCode and versionNumber are required.' });
    }

    try {
       const updatedRecords = await Product.update({ productCode, versionNumber }, req.body).fetch();

       if(updatedRecords.length === 0) {
           return res.notFound({error: 'Product not found with given details'})
       }
       return res.ok(updatedRecords[0]);
    }
    catch(err){
         return res.serverError(err);
    }
  }
};
```

In this example, we extract `productCode` and `versionNumber` from the request parameters using `req.param()`. We then use these values, combined with data sent in the request body (via `req.body`), to update the record. The `.fetch()` after the update operation is crucial here because it returns the modified record after an update. Without it, you'd be working with a non descriptive update result object. The `if(updatedRecords.length === 0)` handles the case where no record matches the provided criteria, which is important for giving good feedback to the client.

Now, to access this route, you will have to create it in `config/routes.js`, like so:

```javascript
// config/routes.js

module.exports.routes = {
  'PUT /product/:productCode/:versionNumber': 'ProductController.updateByProductCodeAndVersion'
};
```

**2. Using findOne to update:**

An alternate, and sometimes more flexible approach when we need to perform complex lookups, involves utilizing `findOne`. It allows us to incorporate more complex search logic as part of the query before performing the update. This approach shines when you are dealing with more intricate scenarios and where conditions for searching aren’t so straightforward.

Here's an example of updating records based on a combination of fields using `findOne`:

```javascript
// api/controllers/ProductController.js

module.exports = {

  updateByComplexCriteria: async function(req, res) {
    const { manufacturingDate, batchNumber,  ...updateData } = req.body;


    if (!manufacturingDate || !batchNumber) {
      return res.badRequest({ error: 'Both manufacturingDate and batchNumber are required in the body.' });
    }

    try {

       const recordToUpdate = await Product.findOne({
            where: {
                manufacturingDate: manufacturingDate,
                batchNumber: batchNumber,
                }
            });

       if (!recordToUpdate) {
           return res.notFound({error: 'Product not found with given details'})
        }

        const updatedRecord = await Product.update({ id: recordToUpdate.id }, updateData ).fetch()

       return res.ok(updatedRecord[0]);

    }
    catch (err) {
      return res.serverError(err);
    }
  }
};
```

Here, we're assuming the request body contains fields like `manufacturingDate`, `batchNumber`, and the other fields to be updated. We use `findOne` to fetch a specific record, then update it. Note that this requires extracting all parameters from the body, which can be quite different from the previous example. This approach also handles the case when the record isn't found, preventing potential issues.

And as before, we define our route in `config/routes.js`:

```javascript
// config/routes.js

module.exports.routes = {
    'PUT /product/bycriteria': 'ProductController.updateByComplexCriteria'
};

```

**3.  Using Query Builder:**

For highly complex situations, the Query Builder offers a great way to define intricate update logic directly within your code. It bypasses the default Waterline/Sails.js model interaction to provide even lower-level control if needed. Be warned, though: this reduces the abstraction that Sails.js normally gives, and should be used judiciously.

Here’s how you could update a record using the query builder:

```javascript
// api/controllers/ProductController.js

module.exports = {
    updateViaQueryBuilder: async function(req, res) {
       const { partNumber, revision, ...updateData } = req.body;

        if (!partNumber || !revision) {
            return res.badRequest({ error: 'Both partNumber and revision are required in the request body.' });
        }

    try {

        const rawResult = await sails.getDatastore().sendNativeQuery(
          'UPDATE product SET ? WHERE partNumber = ? AND revision = ?', [updateData, partNumber, revision]
           )
          if(rawResult.rowsAffected === 0)
          {
              return res.notFound({error: 'Product not found with given details'})
          }

         const updatedRecords = await Product.find({partNumber: partNumber, revision:revision})
         return res.ok(updatedRecords[0]);

    }
    catch(err){
       return res.serverError(err);
    }
   }
};
```

In this example, we use `sails.getDatastore().sendNativeQuery` to interact directly with the database using a parameterized SQL query. We then use `Product.find()` to pull the record and return it to the user after the update. This method gives you a lot of control over the specific SQL being executed, but is also tightly coupled to the specific database, and may require additional handling depending on your target database. This approach is extremely powerful, but needs to be handled with care.

Here’s the relevant route in `config/routes.js`:

```javascript
// config/routes.js

module.exports.routes = {
    'PUT /product/querybuilder': 'ProductController.updateViaQueryBuilder'
};
```

**Key takeaways and recommendations:**

*   **Prioritize Custom Actions:** For most non-primary key updates, a custom controller action with the Waterline ORM (using `update`, or `findOne` followed by an update), or the raw query builder will offer the best balance between readability, maintainability, and control.
*   **Clear Route Definitions:** Make your routes explicit and understandable to avoid confusion with default Sails.js blueprint routes.
*   **Error Handling:** Handle situations where records are not found with appropriate 404 responses. This ensures robust and clear API interactions.
*   **Data Validation:** Always validate incoming data. Ensure the data types and formats match what your database expects and what you expect to receive.

**Resources:**

*   For a deeper dive into Waterline, I highly recommend the [Waterline ORM documentation](https://sailsjs.com/documentation/concepts/models-and-orm/models).
*   For advanced query scenarios, the book "SQL Antipatterns: Avoiding the Pitfalls of Database Programming" by Bill Karwin provides an in-depth understanding of SQL which is crucial for leveraging the query builder effectively.
*  Additionally, "Database Internals: A Deep Dive into How Databases Work" by Alex Petrov provides a thorough understanding of various database functionalities which could be helpful when dealing with complex queries.

These approaches should provide a good starting point for handling updates with composite or non-traditional keys in Sails.js. The key is to remember that you have complete control over how your API operates and you should use the tools at your disposal to structure your code according to your specific needs. It’s about adapting the framework to your requirements rather than being confined by its defaults.
