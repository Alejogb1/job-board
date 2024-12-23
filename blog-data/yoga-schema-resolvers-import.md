---
title: "yoga schema resolvers import?"
date: "2024-12-13"
id: "yoga-schema-resolvers-import"
---

 so you're asking about yoga schema resolvers and import setups right I get it been there done that a million times it feels like

Let me break it down for ya because this is one of those things that can bite you in the butt if you're not careful especially when your project gets bigger and more complicated Believe me I've seen it first hand

So first things first yoga schema resolvers are basically functions that tell GraphQL how to fetch data when a specific field in your schema is requested think of them as the data retrieval layer for your GraphQL API Now when it comes to importing these resolvers well that’s where you can get into a bit of a mess if you are not organized

I remember this one project back in the day like 2017 maybe it was this social media thing where we were using GraphQL and yoga and well it was a total disaster the resolvers were everywhere just scattered across like 20 different files it was a nightmare trying to keep track of where everything was and debugging anything was a complete slog every time we needed to add a new feature or modify an old one it was like a giant treasure hunt to find the relevant resolver it felt like searching for a single needle in a haystack of needles a true coder's despair I tell you we did not have any sort of structure for that and importing the resolvers was like playing russian roulette

Anyway back to the point the simplest way to import resolvers is to do it all in one place in your main GraphQL schema setup file it's not the most scalable way but if your app is small and simple then it’s fine no judgements here we all start somewhere

You can just import every resolver individually and plug them into your resolvers object it will be something like this

```javascript
import { createYoga } from 'graphql-yoga'
import { typeDefs } from './schema'
import { userResolver } from './resolvers/user'
import { postResolver } from './resolvers/post'
import { commentResolver } from './resolvers/comment'

const resolvers = {
  Query: {
    user: userResolver.user,
    users: userResolver.users,
    post: postResolver.post,
    posts: postResolver.posts,
    comment: commentResolver.comment,
    comments: commentResolver.comments
  }
}

const yoga = createYoga({
  schema: {
    typeDefs,
    resolvers
  }
})

export default yoga
```

Now this is fine for small projects but what if your resolvers are all split into multiple modules each with their own sets of queries and mutations and things become a little more complex Imagine this a project with user authentication roles and user details and then on top of that posts and comments and the ability to like them and add attachments and user profiles this whole thing can turn into a monster if you are not planning ahead

What you want to do is something similar to what node-js people do with controllers modules and something similar to what is done on backend frameworks with routers This way you avoid putting too much information on a single file and it keeps your code clean and organized You can use something like a separate file that groups all resolvers and then import that single file

Here's how that might look instead

First create `resolvers.js` or `resolvers/index.js` or `resolvers/resolvers.js` whatever floats your boat and keep your resolvers there and you can add sub folders if needed

```javascript
import { userResolvers } from './user'
import { postResolvers } from './post'
import { commentResolvers } from './comment'

const resolvers = {
  Query: {
    ...userResolvers.Query,
    ...postResolvers.Query,
    ...commentResolvers.Query
  },
  Mutation:{
    ...userResolvers.Mutation,
    ...postResolvers.Mutation,
    ...commentResolvers.Mutation
  }
}

export default resolvers
```

And here is how your individual module user resolvers might look like

```javascript
export const userResolvers = {
  Query: {
    user: (parent, { id }, context) => {
        // your logic here
    },
    users: (parent, args, context) => {
      // your logic here
    }
  },
   Mutation: {
    createUser: (parent, { input }, context) => {
      // your logic here
    },
   updateUser: (parent, { id, input }, context) => {
      // your logic here
    }
  }
}
```

And then on your main server file

```javascript
import { createYoga } from 'graphql-yoga'
import { typeDefs } from './schema'
import resolvers from './resolvers'

const yoga = createYoga({
  schema: {
    typeDefs,
    resolvers
  }
})

export default yoga
```

That cleans things up a bit right You are no longer importing 20 different resolvers each time you want to add something new This pattern allows you to keep things organized and easier to find and you can even go as far as having a specific resolver import file for each sub-folder inside of resolvers. This means that you can potentially add a feature by just creating a folder adding index file with the combined resolvers and changing the main resolvers import file

Now this way of combining them into a single import is pretty neat for organization but there's more you could do if you wanted to make things even more scalable especially as your resolvers themselves become more complex

There is a design pattern that is used a lot in the backend world that is called dependency injection basically you are creating functions that have a list of dependencies and those dependencies can be resolved and inject into the function without the need of explicit passing them as parameters this makes it easier to test and work with complex functions that require a lot of context

Now implementing a full dependency injection system for your resolvers might be overkill for many projects but using the basics of it is generally good practice

You could do something like this

```javascript
import { userResolvers } from './user'
import { postResolvers } from './post'
import { commentResolvers } from './comment'

const createResolvers = (dependencies) => ({
  Query: {
    ...userResolvers.Query(dependencies),
    ...postResolvers.Query(dependencies),
    ...commentResolvers.Query(dependencies)
  },
   Mutation: {
    ...userResolvers.Mutation(dependencies),
    ...postResolvers.Mutation(dependencies),
    ...commentResolvers.Mutation(dependencies)
  }
})


export default createResolvers
```

And on your individual resolvers file

```javascript
export const userResolvers = (dependencies) => ({
  Query: {
    user: (parent, { id }, context) => {
        // use dependencies here if you need them
    },
    users: (parent, args, context) => {
       // use dependencies here if you need them
    }
  },
  Mutation:{
    createUser: (parent, { input }, context) => {
       // use dependencies here if you need them
    },
    updateUser: (parent, { id, input }, context) => {
       // use dependencies here if you need them
    }
  }
})
```

And your main file

```javascript
import { createYoga } from 'graphql-yoga'
import { typeDefs } from './schema'
import createResolvers from './resolvers'

const dependencies = {
    // your dependencies here like database connection etc
}

const resolvers = createResolvers(dependencies)

const yoga = createYoga({
  schema: {
    typeDefs,
    resolvers
  }
})

export default yoga
```

Now why do this? Well you might ask yourself why do we need to pass those dependencies?

Well, imagine that you have database connections or authentication logic or some third-party API calls your resolvers would be doing all the time and having those on the global scope might be a bit of a problem you can now easily test your resolvers with mocked dependencies and it also helps keep your resolvers cleaner and more focused on the actual data logic

You can see that the createResolvers function is just a factory that receives the dependencies and returns the resolvers with all the functionalities now all your resolvers will be using the dependencies defined in the create resolvers function and you can pass all the important dependencies like database connections and so on this way

It is a way to improve code maintainability and readability and also helps with code decoupling so this makes it easier to test them in isolation

This is how I've been doing it for years and it works pretty darn well now obviously there are other ways to do it you could use code generation tools and other fancy things but this method works with vanilla nodejs and javascript and it is a solid starting point if you are not sure what to do with your resolvers and you want to avoid the headache

Also as a side note always keep your schema and resolver files separate it makes things easier to reason about and it avoids confusion when you are trying to maintain your code

As for resources on this topic I would avoid going to random blog posts and I suggest you take a deep look into books like "Production Ready GraphQL" or official GraphQL documentation or the Apollo documentation specifically for resolvers those resources will give you a solid theoretical base for using this

Also the "Clean Code" book is a must-read for any developer who wants to write maintainable code it's not strictly related to GraphQL resolvers but it helps you with creating a great codebase that is easy to work with

And remember code is like a joke if you have to explain it then it's bad so try to keep your code clean easy to understand and simple and try not to overcomplicate things I mean you are here trying to understand how to import resolvers so let's not make things unnecessarily complicated eh?

Anyway I hope that this helps and you can see how to import your yoga schema resolvers properly now if you need any other advice about GraphQL let me know I'm always happy to help fellow developers avoid the same headaches that I went through
