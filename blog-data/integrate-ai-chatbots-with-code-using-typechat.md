---
title: "Integrate AI Chatbots with Code Using TypeChat"
date: "2024-11-16"
id: "integrate-ai-chatbots-with-code-using-typechat"
---

dude, so this talk by daniel roser on typechat—total mind-blow.  it's basically about making ai chatbots play nicely with your regular code, which is way harder than it sounds.  think of it like trying to get a golden retriever to do your taxes—possible, but requires serious training and a whole lotta patience.

the whole point of the video is to bridge the gap between those super-chill chat interfaces like chatgpt—you know, just ask a question and get a rambling answer—and the super-precise world of traditional app development where everything's gotta be just so.  the problem is chatbots spew natural language, which is great for humans but total garbage for code.  you can’t just shove a chatbot’s answer directly into your app.

he starts off with this simple example—a little app showing Seattle venues on a rainy day (because let's be honest, that’s like, every day in Seattle).  he shows this basic app with some user input at the top followed by a list of venues and their descriptions.  totally relatable, right?  i’m picturing the speaker with a slightly weary smile, like he's been battling the Seattle weather for years.  then he hits the real problem: getting a chatbot to build that for you is a total pain.


one of the key visual cues is this screenshot of the app itself—super basic, nothing fancy, just to get the point across. another visual cue is when he's showing the code examples, highlighting those specific type definitions.  he also emphasizes this point verbally, stating that “parsing natural language is extremely hard if not a Fool's erron”. it was funny the way he said it, too. and then he had this totally exasperated expression as he explained why basic parsing methods just won't work. the guy's a genius but relatable too!  


he goes through a few attempts to coax the chatbot into giving nicely formatted data.  first, it's like, "pretty please give me some json"—and sometimes it works.  but that's super unreliable.  sometimes it misses a property, or adds extra ones, or the formatting’s all wonky. it’s like wrestling a greased pig.


here's where the genius comes in: types.  he uses typescript, but the concept applies to other languages too. the key idea here is *schema engineering* not just *prompt engineering*.  imagine defining the *exact* structure of your data using types.  say you need a list of venues, each with a `name` (string) and `description` (string):

```typescript
interface Venue {
  name: string;
  description: string;
}

interface VenueList {
  venues: Venue[];
}
```

now, you give this type definition to the chatbot *along* with the user's request.  the chatbot now knows *exactly* what kind of data to return.  it's like giving the golden retriever a very clear instruction manual.  this is so much better than relying on the chatbot to magically guess the right format.



but that's only half the battle.  even with types, the chatbot could still mess up.  the other key idea is *type validation*.  typescript's compiler is awesome for this.  you can basically tell the compiler, "hey, check if this data matches the `VenueList` type i just defined."  if it does, great.  if not, you get a nice, helpful error message telling you exactly what went wrong.


```typescript
// This is a simplified example. TypeChat handles this for you
const response = await getVenueDataFromChatbot("Rainy day venues in Seattle");

try {
  const validatedResponse: VenueList = response; // Type checking happens here
  console.log(validatedResponse);
} catch (error) {
  console.error("Validation failed:", error);
  //Try to recover the chatbot's response here.
}
```

this error message becomes your recovery mechanism.  you use it to tell the chatbot, "nope, try again, but this time, get it right!"


the resolution? typechat.  it's an npm library that bundles all this together.  it takes your types, your user input, and your favorite language model (openai, anthropic, llama, whatever), and spits out nicely typed data, or a helpful error message if things go south.  it’s essentially a super-powered translator that speaks both chatbot and app.

he shows a demo—a coffee shop app.  you can place orders like “one latte with foam, please,” and it works perfectly.  but then he throws a curveball: “one latte and a medium purple gorilla named Bonsai.”  the app gracefully handles this weird input, showing the error and suggesting a repair.  the speaker is totally deadpan, while the screen is showing some strange output; it is great comedy.


the code for this coffee shop example is surprisingly short:

```typescript
import { createModel, translate } from 'type-chat';

// Coffee shop types (simplified)
interface CoffeeShopOrder {
  item: string;
  options?: string[];
}

const coffeeShopModel = createModel<CoffeeShopOrder>(/*...type definitions here...*/); //typechat handles all of this automatically

const userOrder = "one latte with foam please";
const translatedOrder = await translate(coffeeShopModel, userOrder);

console.log(translatedOrder); // should be well-typed data
```

that's it.  it handles the types, talks to the language model, and validates the response.  that's literally the core magic of the library. it's all about creating a model for the language model. it’s mind-blowing stuff.


he takes it even further—building a mini-calculator app using the same approach.  the types define the available operations (`add`, `subtract`, `divide`, etc.), and typechat makes sure the chatbot only uses those operations.  the cool part is that typechat generates a fake language in JSON to ensure the chatbot doesn't go rogue.

he also demonstrates a slightly more sophisticated example, showcasing how a program can be generated using typechat to handle more complex tasks. this time, he uses a "fake language" in JSON format.  the code is, of course, simplified in the talk but you get the general idea. this enables the safety constraints and sandboxing he talks about.

```typescript
// Example of a simplified program in typechat's "fake language" (JSON)

{
  "program": [
    { "op": "add", "args": [1, 2] }, // adds 1 and 2
    { "op": "multiply", "args": [ { "ref": 0 }, 3 ] } // multiplies the result of the previous step (3) by 3
  ],
  "results": [3,9] // the results of each step
}

//This example shows how the JSON structure allows for the creation of a program
//that ensures no loops or other unsafe code executions are created by the language model.
```


then, he drops this crazy bomb—the same idea works for python too!  he shows a similar example of processing a csv file, using python types to guide the chatbot and validate the results.  it’s just another example to show how general the concept is.  the underlying principle is the same—using types to guide and validate.

overall, the talk was super clear, engaging, and mind-blowing.  i’m not an ai expert, but i understood it perfectly.  the code examples were simple, easy to follow, and illustrative.  and the humor—that totally unexpected “medium purple gorilla named Bonsai”—was the icing on the cake.   the conclusion: types are all you need.  typechat is a game changer.  it’s not perfect yet (he even mentions some experimental python integrations) but the potential is huge.  he ends by encouraging the audience to try typechat, reach out, and collaborate.  basically he's invited everyone to help him build the future of AI programming! and who wouldn’t want to do that?
