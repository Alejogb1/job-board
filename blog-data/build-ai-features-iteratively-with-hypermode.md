---
title: "Build AI Features Iteratively with Hypermode"
date: "2024-11-16"
id: "build-ai-features-iteratively-with-hypermode"
---

dude,  this video was *insane*.  it's like, a hyper-speed crash course on building ai features, but with a hilarious game thrown in for good measure.  the whole point?  to show how easy it is to iterate on ai stuff using their platform, hypermode.  they spent, like, 50 minutes showing how to build and improve ai features in a super iterative way—the kind of thing that usually takes ages.

first, the setup—this wasn't your typical stuffy tech presentation.  they started with this ridiculously fun game called "hyper categories,"  a souped-up version of categories that could handle way more players than the old-school way. i mean,  they actually had us all play along using our phones and a qr code—totally unexpected and engaging. they even had a jacket for the winner, which was awesome.  a visual cue i remember was the leaderboard flashing up on screen as people submitted their answers, showing the bot getting totally schooled.  another was that little improv moment where they threw on some jeopardy music while people were still typing. super casual, super funny.


the key ideas were centered around this notion of "incremental iteration" and modularity.  they hammered home that building ai features isn't some monolithic thing—it's about assembling reusable building blocks.  they used the hyper categories game as an example:


* **building blocks**:  the game wasn't some magical, single piece of code.  it was built from functions like scoring (checking if the answer started with the right letter and fit the category), validation (using a model to verify the answer made sense), clustering (grouping similar answers), and leaderboard updates.  each of these is a distinct unit that could be reused in other projects. this is exactly how you build robust and scalable ai systems

* **model interfaces and abstraction**:  a huge part of the story was how they abstract away the complexities of different ai models.  they use something they call "model interfaces" to handle all the quirks of different APIs.  imagine trying to use openai and hugging face models without these – it would be a nightmare! they showed a snippet that makes calling an openAI model super simple.  it's like magic. they even used assembly script, which compiles to webassembly for speed and security.


here's a taste of what that kind of abstraction looks like. this is a super simplified example, but it gets the idea across:

```typescript
// simplified model interface example

interface ModelInterface {
  predict(input: string): Promise<string>; // a generic prediction method
}

class OpenAIModel implements ModelInterface {
  async predict(input: string): Promise<string> {
    //  Actual OpenAI API call here, abstracted away
    const response = await fetch("openai-api-endpoint", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ input }),
    });
    const data = await response.json();
    return data.output;
  }
}

class HuggingFaceModel implements ModelInterface {
  async predict(input: string): Promise<string> {
    // Actual Hugging Face API call here, abstracted away
    const response = await fetch("huggingface-api-endpoint", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ input }),
    });
    const data = await response.json();
    return data.output;
  }
}

// usage
const openAI = new OpenAIModel();
const huggingFace = new HuggingFaceModel();

const openAIResult = await openAI.predict("some text");
const huggingFaceResult = await huggingFace.predict("some text");

console.log(openAIResult, huggingFaceResult); //  we get the prediction regardless of the model.

```


the second key idea involved their platform itself,  hypermode.  they emphasized how easily you could connect to your databases, apis, and various model hosts.  it all looked incredibly streamlined, and they kept showing the graphical interface to manage all these elements.  one visual cue i noted was the "inferences" tab where you could see a history of model runs, execution times, inputs, and outputs, which is crucial for debugging and monitoring.

after the game, they dove into a more practical example:  triaging github issues using ai.  this involved two functions: `classifyIssue` (determining if an issue is a bug, feature request, etc.) and `trendSummary` (summarizing trends across many issues).


here's a snippet showing a simplified version of the `classifyIssue` function, highlighting how they leverage a pre-trained model from hugging face:

```typescript
import { ClassificationModel } from '@hypermode/models'; // Import from hypermode's model library

export function classifyIssue(title: string, description: string): string {
  const model = new ClassificationModel('bert-mnli-github-issues'); // Specify the pre-trained model
  const input = [title + ' ' + description]; //Combine title and description for classification
  const predictions = model.predict(input); // The model interface handles the API call
  return predictions[0].label; // Return the predicted label (bug, feature request, etc.)
}
```

and here's a  *very* simplified version of the `trendSummary` function, which uses openai:

```typescript
import { OpenAIChatModel } from '@hypermode/models';

export async function trendSummary(issues: any[]): Promise<string> {
  const model = new OpenAIChatModel();
  const prompt = `Summarize the trends from these GitHub issues:\n${issues.map(issue => issue.title + ": " + issue.body).join('\n')}`;
  const summary = await model.predict(prompt); // Again, the model interface simplifies the API call.
  return summary.trim(); // Clean up the output
}

```

the resolution?  they showcased how hypermode simplifies the entire process of building, deploying, and iterating on ai-powered features.  the platform handles everything from model management to deployment, letting developers focus on the logic and not the infrastructure. they basically showed how you can use their system to quickly build a useful ai feature from scratch to production, improving it along the way, using only a few lines of code.  they even teased future features like database integrations—making the whole thing even more powerful.

throughout the whole thing, they used this super friendly, casual tone.  it felt less like a presentation and more like a friend showing you some cool new tech.  this approach makes complex topics approachable. they even admitted that things might break because they’re just launching – which is refreshingly honest. all in all, the video was a great example of how to explain complex technical concepts in a fun and engaging way.  i’m already thinking of all the awesome ai features i could build with hypermode.
