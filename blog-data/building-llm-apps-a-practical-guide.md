---
title: "Building LLM Apps: A Practical Guide"
date: "2024-11-16"
id: "building-llm-apps-a-practical-guide"
---

dude so this keynote was wild it was like six ai ninjas teaming up to drop some serious knowledge bombs on building llm apps  they basically said "forget everything you think you know about the gold rush in ai"  the whole point was showing how building a killer llm app isn't just about the model itself it's the whole shebang  strategy operations and tactics  think of it like a three-legged stool  if one leg's wobbly the whole thing collapses

first they laid down the strategic game plan  basically don't try to out-google google  forget training your own massive language model unless you're google or openai  that's not your competitive advantage  your advantage is your product expertise your niche your understanding of the problem you're solving.  they used the analogy of "steamrolling"  focus on what you do best and leverage existing models as tools.   think of gpt-4 as a really fancy sas product  if something better comes along switch fast.

this is where things got really interesting. they showed a slide with this awesome diagram a virtuous cycle of improvement.  it basically looked like a loop with evals and data at the center  they stressed that constantly evaluating your model's performance and using that data to refine it is key.  it wasn't just about fancy metrics, it was about asking "does this actually solve the user's problem?"  this isn't new, they pointed out.  it's the same iterative process at the heart of mlops devops and even the lean startup methodology.  the toyota production system and kaizen (continuous improvement) were name-dropped as inspiration.  they even used a funny quote "value is only created when metal gets bent" meaning you need to ship something to get real user feedback.  it’s not all theory.


here's a little python snippet illustrating that feedback loop:

```python
import random

# simplified model (replace with your actual llm)
def llm_response(prompt):
  return random.choice(["yes", "no", "maybe"])

# simple evaluation function (replace with your sophisticated evals)
def evaluate_response(response, expected):
  return response == expected


# feedback loop
while True:
  user_prompt = input("Enter prompt: ")
  expected_response = input("Enter expected response: ")
  response = llm_response(user_prompt)
  evaluation = evaluate_response(response, expected_response)
  print(f"Response: {response}, Evaluation: {evaluation}")

  # incorporate feedback (replace with your model training/adjustment logic)
  if not evaluation:
    print("Model needs improvement")
   #add logic to retrain or adjust the model based on the incorrect response


```

that code's super basic, but it shows the idea.  you get user feedback, evaluate the model, and use that to improve it. it's a cycle, not a one-time thing.

then they dove into the operational side things went sideways quickly. jason lou essentially gave a masterclass in how to *screw up* your llm app project.  he joked about buying more shovels (tools) without a clear plan, switching databases constantly, and hiring super expensive ml engineers who fix typescript errors instead of focusing on real-world problems.  his advice was hilarious but deadly serious: focus, don't chase every shiny new tool.

he hammered home the point that vague job titles like "ai ninja" are a recipe for disaster. you need to be specific about skills and expectations.  hamam hussein jumped in to talk about the importance of data literacy and evals even for "ai engineers."  he stressed that you can get really good at evals with just a few weeks of focused effort.  this is a critical skill.  you can't improve without measuring.

next up was the tactical deep dive with shreya and eugene. they talked about creating practical, simple evaluations. don’t try to build some overly complex metric, break it into smaller pieces.  for example, if your llm is supposed to extract product information from text, just check that it gets the price, title, and rating correct.  start simple, then get fancy. they talked about using llms *as evaluators*, a pretty cool concept.   you can prompt the llm to judge the output of another llm or use a fine-tuned model.

here's a little javascript snippet showcasing a simple evaluation using a pre-trained classifier (you'd replace this with an actual classifier):

```javascript
// Simulate a pre-trained sentiment classifier
const sentimentClassifier = (text) => {
  // Replace this with your actual sentiment classification logic
  const positiveWords = ["good", "great", "excellent"];
  const negativeWords = ["bad", "terrible", "awful"];
  let positiveCount = 0;
  let negativeCount = 0;
  text.toLowerCase().split(" ").forEach(word => {
    if (positiveWords.includes(word)) positiveCount++;
    if (negativeWords.includes(word)) negativeCount++;
  });
  return positiveCount > negativeCount ? "positive" : "negative";
};

// Example usage
const llmOutput = "The product is great";
const sentiment = sentimentClassifier(llmOutput);
console.log(`LLM Output: ${llmOutput}, Sentiment: ${sentiment}`);

//Simple evaluation based on sentiment
const isPositive = sentiment === "positive";
console.log(`Is Positive: ${isPositive}`);
```

this code’s super basic but shows how to build on pre-trained models.  you replace the super simple `sentimentClassifier`  with a much more sophisticated model.


then they went into building automated guard rails. think of these as safety nets to catch problematic outputs like toxic text, personal info leaks, or hallucinations.  the key takeaway here was that you have to look at your data regularly – all of it. create dashboards, alerts, and whatever you need to see what’s happening.  don't just build it and forget it.  they emphasized the importance of tracing everything back to the model version, prompt version and the code to make debugging much easier.

finally they hit us with a big-picture wrap-up.  they showed this diagram from a seminal mlops paper that's almost a decade old showing all the things that go into deploying ml models it was a reminder that building real-world llm applications involves far more than just throwing an api call into your code  you’ve got to maintain this entire complex ecosystem. and the last point was that going from demo to actual product takes a long time  seriously don’t rush it.

here’s a bit of simple node.js to show how to log data for later analysis (again, super basic, but illustrative):

```javascript
const fs = require('node:fs');
const { v4: uuidv4 } = require('uuid');

// Function to log data with metadata
function logData(data, modelVersion, promptVersion, codeVersion) {
  const logEntry = {
    id: uuidv4(),
    timestamp: new Date(),
    modelVersion: modelVersion,
    promptVersion: promptVersion,
    codeVersion: codeVersion,
    data: data,
  };
  const logDir = './logs';
  if (!fs.existsSync(logDir)) fs.mkdirSync(logDir);
  const logFile = `${logDir}/log.json`;
  fs.appendFile(logFile, JSON.stringify(logEntry) + '\n', err => {
    if (err) console.error('Error writing to log file:', err);
  });
}

// Example usage
const llmOutput = { text: "This is some LLM output", sentiment: "positive" };
logData(llmOutput, "v1.0", "v2.0", "v3.0");
```

the whole keynote was a hilarious and insightful deep dive into the real-world challenges of building llm apps.   it was less about the “shiny new toy” and more about the nitty-gritty, the sweat and toil of building a successful product.  and yeah, the six of them totally lived up to their "avengers" team-up status.
