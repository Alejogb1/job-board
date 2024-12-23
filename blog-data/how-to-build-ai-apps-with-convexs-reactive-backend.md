---
title: "How to Build AI Apps with Convex's Reactive Backend"
date: "2024-11-16"
id: "how-to-build-ai-apps-with-convexs-reactive-backend"
---

dude so i just watched this crazy talk about this platform called convex and it's totally blowing my mind like seriously it's this whole backend-as-a-service thing but with a seriously cool twist  they're basically trying to automate all the boring backend stuff that nobody wants to deal with  you know  like gluing databases to apis managing data formats dealing with caching  all that jazz  it's all about freeing up devs to focus on the actual app and not get bogged down in plumbing  think of it as a supercharged firebase but way more ambitious


 so the whole spiel started with this guy saying they *accidentally* built an ai platform  which is hilarious  but it's true  they were aiming for a super streamlined backend system and it ended up being perfect for generative ai apps  which is a total happy accident  right  he mentions that half a dev team’s time is spent on backend busywork that users don’t even see – that’s a huge problem that convex aims to solve


one of the key visual cues was this super simple react code snippet he flashed on screen –  something like this


```javascript
const [count, setCount] = useState(0);

return (
  <div>
    <p>You clicked {count} times</p>
    <button onClick={() => setCount(count + 1)}>
      Click me
    </button>
  </div>
);
```

basic right  but the point was how react handles state updates  it’s all about reactive programming where changes in one part automatically update dependent parts  convex takes that idea and extends it to the backend  which is mindblowing


another key moment was when he talked about how traditional server-side stuff breaks this reactive flow  you have to manually pull data invalidate caches set up push notifications – it's a mess  convex aims to completely fix that  it's like react but for the whole application including the backend


the third key thing was the whole "queries and mutations" thing  which sounds familiar if you've used graphql or other similar frameworks  but convex's version is special because it tracks all data dependencies  it’s not just querying data, it's also actively tracking the relationship between data and the backend actions that modify it.  this allows for automatic updates, completely eliminating the need for manual cache invalidation or other workarounds.


he showed an example of an app building workflow – a recipe app that uses convex's backend to generate project plans which involved fetching project names feature requests color palettes and icon ideas concurrently – all happening in the background and seamlessly updating the app as results come in.  this is where the power of convex’s reactive backend shines


then he showed another code snippet (imagine this one on a slide, super stylized):


```typescript
// Convex schema definition
export const myCollection = collection({
  schema: {
    name: string(),
    vector: vector(128), // <-- the vector index!
  },
});

//Adding a vector index:
await myCollection.createIndex({
  fields: ['vector'], // <-- indexing the vector field
  type: 'vector', // <-- explicitly defining it as a vector index
  engine: 'hnsw', // <-- choose your vector index engine - this is super cool
});

```

that's a typescript snippet showcasing how easily you add vector indexes to your data in convex  it's ridiculously simple –  just adding a `vector` field and then specifying a vector index type –  and it enables blazing fast similarity searches – a critical component of many generative ai apps


and the fourth thing was the sheer number of generative ai startups using convex  90% of their projects are generative ai related –  that speaks volumes   clearly  this reactive backend system is incredibly useful for these kinds of applications  because generative ai workflows are inherently asynchronous and involve multiple steps – convex handles the complexity so the developers don't have to


a final key moment was the discussion of convex components  these are pre-built state machines and building blocks, encapsulating sophisticated workflows – allowing developers to integrate complex backend logic easily into their apps.  think of it as a massive library of ready-to-use backend functions, all built around this reactive paradigm – it's like getting a whole team of backend engineers for the price of a subscription


the whole thing culminated with them announcing a startup program with discounts and support –  clearly they are betting big on the generative ai space and want to onboard as many developers as possible


oh and one more thing a final code snippet showing a query function in convex (typescript again)


```typescript
// Convex query function
export const getRecipes = async ({ searchTerm }: { searchTerm?: string }) => {
  const q = query(recipes)
    .where(r => r.name.contains(searchTerm));
  return await q;
};


```

this function shows a simple query that retrieves recipes, optionally filtering by a search term  notice how clean and concise it is – that’s the beauty of convex.  the underlying complexity of data fetching and updating is completely abstracted away.


so yeah that was convex  a platform that promises to revolutionize backend development especially for generative ai and other async workflows –  the whole "accidentally made an ai platform" line is genius marketing.  it's clever,  it's ambitious,  and from what I saw it’s pretty darn effective  definitely worth checking out if you're into building cool things  especially if you're working with ai.  it sounds like a game changer for app development.
