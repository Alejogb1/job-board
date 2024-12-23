---
title: "Is custom logic appropriate for GraphQL input types?"
date: "2024-12-23"
id: "is-custom-logic-appropriate-for-graphql-input-types"
---

Alright, let’s tackle this. I've seen this particular conundrum play out in different projects over the years, and it's one that warrants careful consideration. Is custom logic appropriate for GraphQL input types? The short answer, surprisingly perhaps, is: it depends, but generally, proceed with extreme caution. Input types in graphql are primarily designed for data *transport*, not for execution of complex logic. However, the line blurs at the edges, and sometimes, we find ourselves needing to nudge that line just a little. Let's unpack why.

My journey with graphql started, like many others, with a straightforward schema. We had basic types, simple queries, and mutations that mostly mirrored our database operations. It felt elegant, clean. Then, the requirements grew. A particularly challenging case involved a complex registration form. The fields were inter-dependent, some were optional based on other selections, and we needed to enforce a couple of intricate business rules before any data hit the backend. The question became: do we handle all this complexity in the client-side, or do we find a way to embed some of that into our graphql input types? We decided to experiment – and that's where I learned some valuable lessons.

The fundamental principle to grasp is that graphql input types should be focused on defining the structure of the data being submitted. They are, at their core, data containers. Loading them up with custom logic can lead to several problems. The most significant issue, in my experience, is the violation of the separation of concerns. Your graphql schema should primarily describe your data graph. When you start adding validation, transformation, or conditional logic directly into input types, you are essentially mixing your data definition layer with application logic – which invariably leads to more complex debugging, difficult maintainability, and less reusable code.

Think about it: what happens when that specific rule you embedded inside your input type needs to be shared across different mutations or queries? You end up duplicating logic, and that duplication is a maintenance nightmare. Another big concern is that graphql tooling, such as schema validators and code generators, typically assume a degree of "purity" with respect to the data that enters the server via input types. Adding custom logic disrupts this expectation and can lead to unforeseen compatibility issues and unexpected behavior, particularly when moving to different graphql implementations.

Now, this isn't to say that *all* forms of logic in input types are terrible. There's a grey area, and sometimes, certain simple, declarative validations within input types can be effective and convenient. For instance, basic type validation (ensuring a field is a number when it should be), simple length checks, or basic regex validations to enforce data formats are generally safe. However, I advise keeping things strictly at this level. You should never be calling external APIs or altering data based on complex conditional checks directly inside an input type definition.

To illustrate this point, let's look at some examples. The first shows a case where simple validation is used, which I'd generally consider acceptable:

```graphql
input NewUserInput {
    username: String! @constraint(minLength: 5, maxLength: 50)
    email: String! @constraint(format: "email")
    age: Int @constraint(min: 13)
}

type Mutation {
    createUser(input: NewUserInput!): User
}
```

In this snippet, we are using custom directives (which might be implemented using libraries like `@graphql-constraint-directive` or similar) to enforce basic rules related to minimum and maximum length, email format, and age bounds. This falls within the acceptable scope of *declarative* validation on the input data structure. These directives provide a clean and declarative way to express constraints without injecting application logic directly into the schema definition.

Now, let's explore a scenario where things go wrong. Imagine we tried to incorporate conditional validation that depends on other input values directly in the type using a custom directive (which is an anti-pattern):

```graphql
input ProductInput {
  type: String! @allowedValues(values: ["physical", "digital"])
  physicalWeight: Float @conditionalRequired(if: "type == 'physical'")
  digitalSize: Int @conditionalRequired(if: "type == 'digital'")
  name: String!
}

type Mutation {
    createProduct(input: ProductInput!): Product
}
```

Here, the idea is that `physicalWeight` is only required if the product type is 'physical,' and similarly `digitalSize` is needed only if the type is ‘digital’. This approach is dangerous. The `@conditionalRequired` directive now needs to implement business logic and evaluate the input values dynamically. This quickly gets messy, hard to test, difficult to reuse, and tightly couples input data with business logic within the schema itself. If the conditional check changes, we're now potentially messing with the schema, which isn’t the ideal place to do so. I've seen this exact scenario create all kinds of headaches on large projects.

Instead of doing something like the example above, we can keep the input types clean and do validations and conditional logic within the resolver of mutation:

```graphql
input ProductInput {
  type: String! @allowedValues(values: ["physical", "digital"])
  physicalWeight: Float
  digitalSize: Int
  name: String!
}

type Mutation {
    createProduct(input: ProductInput!): Product
}
```

And the resolver logic (using pseudo-code for illustration in node.js):

```javascript
const createProductResolver = async (parent, { input }, context) => {
    const { type, physicalWeight, digitalSize } = input;

    if (type === 'physical' && typeof physicalWeight !== 'number') {
        throw new Error("Physical weight is required for physical products.")
    }

    if (type === 'digital' && typeof digitalSize !== 'number') {
      throw new Error("Digital size is required for digital products.")
    }

    // Logic to create a new product
    const newProduct = await createNewProduct(input);

    return newProduct;
}
```

The resolver, rather than input type, now takes on the responsibility of implementing conditional logic based on the submitted input, adhering to best practices by keeping input types simple, clean and declarative, and placing logic where it belongs: in resolvers and business logic layers.

The key takeaway here is about placement and separation. Keep your input types focused on the data structure, nothing more. Any logic that needs to evaluate values, transform them, or enforce conditional rules based on other fields should reside in your resolver logic or your business layer code.

For further reading, I highly recommend "GraphQL in Action" by Samer Buna, it's an excellent resource on this topic, particularly regarding the separation of concerns in graphql application architecture. Another paper that I frequently go back to is "Designing Evolvable APIs with GraphQL" by Facebook Engineering. Both provide great insight in schema design, and handling complex input validation using patterns that promote flexibility and maintainability. They'll help solidify these concepts and provide a deeper understanding of why separating data definition from execution logic is critical for a sustainable graphql implementation. Don't be tempted to bypass these design principles – you’ll thank yourself later.
