---
title: "Building AI-First Products with Anthropic's Claude"
date: "2024-11-16"
id: "building-ai-first-products-with-anthropics-claude"
---

hey dude so i just watched this awesome talk about claude and anthropic's latest ai stuff and man it was wild  it's basically a deep dive into how they're thinking about building ai products not just slapping an ai chatbot onto existing apps which is what a lot of companies are doing right now  think of it like the difference between those old steam-powered factories and the modern electric ones  they talked a lot about that actually

so the whole point of the video was to show how anthropic's building its llms and how devs should be thinking about building *with* llms not just *around* them you know  it's about a paradigm shift man it's like the whole presentation had this "we're changing the game" vibe to it they mentioned this historical parallel with the electrical revolution  super interesting stuff


first off they straight up compared adding llms to existing products to replacing steam engines with electric ones in factories back in the day  remember those old factories  huge steam engines powering everything  a total mess of belts and gears  super inflexible  when electricity came along a lot of factory owners just swapped out the steam engine  but that didn't change the underlying factory design it just gave them some lights  real innovation happened when they redesigned factories from scratch to use electricity efficiently  that’s precisely what's happening now with llms

the speaker dude alex i think his name is  threw in some killer visuals like those old factory diagrams  you could practically smell the coal dust lol  then there were charts showing how claude 3.5 sonnet crushes the competition on benchmarks  and screenshots of the new artifact feature  it looked super slick honestly

one of the big ideas was this concept of "artifacts"  imagine you're building a website with claude  instead of just getting a chat transcript you get a folder full of all the stuff claude generated images code snippets json whatever  it separates the *content* claude made from the *conversation* itself  it's like having a super organized project folder automatically generated  think of it this way:

```python
# a simplified example of managing artifacts
artifacts = {}
chat_history = []

# during a chat session
user_prompt = "create a react component for a button"
claude_response = {"code": "<button>Click Me</button>", "description": "a simple react button"}
artifacts["button_component"] = claude_response
chat_history.append({"user": user_prompt, "claude": claude_response})

# later access the artifact
print(artifacts["button_component"]["code"]) #prints the react code
```

that's super handy because it lets you actually use claude's output without all the messy chat logs cluttering things up  you can grab those react components or svgs and drop them straight into your project  no more copy-pasting and fixing formatting it's pure magic

another major point was "projects"  a feature that lets you ground claude's output in your existing knowledge base  so say you've got a style guide for your company or a massive code repo  you upload it into projects and now claude can base its responses on your internal stuff  it's like giving claude context supercharging its performance  it's kind of like  building a hyper-personalized knowledge graph  this bit totally blew my mind


```javascript
// example of leveraging project data (conceptual)
const projectData = {
  styleGuide: {
    colors: {
      primary: "#007bff",
      secondary: "#6c757d"
    }
  },
  codebase: {
    // existing code structure
  }
};

// claude accesses projectData to create styled code
const claudeResponse = claude.generateCode({
  prompt: "create a button using the primary color from the style guide",
  context: projectData
});

console.log(claudeResponse); // output includes a button with the primary color
```

they also talked about the new claude 3.5 sonnet model  it's like claude 3 but on steroids  faster cheaper and way smarter  they showed benchmarks showing it obliterating previous models on various tasks especially coding and reasoning  it's got this crazy 200k context window which means it can remember a ton of stuff which really matters for complicated tasks

check out this code example of how they measured its pull request problem-solving abilities:

```python
# simplified representation of evaluating pull request resolution
def evaluate_pull_request(model, pull_request_description):
  """Simulates a model solving a pull request."""
  solution_steps = []
  current_state = pull_request_description
  for i in range(5):  # max 5 attempts
    code_suggestion = model.generate_code(current_state)
    test_result = run_tests(code_suggestion)  # hypothetical test function
    if test_result == "success":
      solution_steps.append(code_suggestion)
      break
    else:
      current_state += test_result + "\n" + code_suggestion
      solution_steps.append(code_suggestion)
  return "success" if test_result == "success" else "failure", solution_steps


# hypothetical test runner
def run_tests(code):
    # run unit tests or similar
    return "success" # or error message
```

the cool thing is that the model is available on multiple platforms  aws bedrock google vertex ai you name it  they’re really going for accessibility.  they also mentioned  pricing  sonnet is five times cheaper than the previous model which is massive news for developers

the resolution is that we're at the dawn of a new era in ai development  it's not just about adding ai features but about building ai-first products  that's the key takeaway  and tools like artifacts and projects are making it easier than ever to do that  also claude 3.5 sonnet is a beast of a model and i'm stoked to see what people build with it.  they’re also working on things like a steering api  which lets you fine tune model behavior which opens up a whole other level of customization. seriously wild stuff man you should check out the whole talk


anyway  that's the gist of it  it was super insightful and pretty funny too  alex the speaker is a really engaging guy  the whole thing had me hooked  i'm already brainstorming ai-first product ideas  let me know what you think dude
