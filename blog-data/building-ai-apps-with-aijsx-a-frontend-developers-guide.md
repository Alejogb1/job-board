---
title: "Building AI Apps with AIJSX: A Frontend Developer's Guide"
date: "2024-11-16"
id: "building-ai-apps-with-aijsx-a-frontend-developers-guide"
---

dude so i just saw this killer presentation on ai jsx and i gotta tell you it blew my mind like seriously  the whole thing was about making ai app development way easier especially for us frontend folks who usually get stuck watching the backend python devs have all the fun


the setup was basically this the guy's like look building awesome llm apps is a total nightmare you gotta wrangle vector databases context windows rag stacks the whole shebang  most of us here love that kinda stuff but let's be real most devs just wanna build cool things not wrestle with infrastructure  so fixie's mission is to make that easier their claim to fame is ai jsx which they're calling the future of ai app development they even had that 2001 space odyssey music playing during the intro haha


one of the first things he pointed out was that the "bear staring into the window" problem.  lots of frontend and full stack devs—mostly javascript peeps—are all like hey we wanna build ai stuff but it's all in python and we're stuck using whatever tools exist... which are not great.   ai jsx tries to solve this using typescript.  he showed a hello world app where you basically write a jsx-like component that calls an llm to write a shakespearean sonnet about llms.   super simple but cool


the key concepts were pretty mind blowing. one was the whole ai jsx framework itself it's designed to be like react but instead of rendering to the dom you're rendering to an llm.  you define components with  `<...> </...>` tags and everything   this is a simple component:

```typescript
const MyComponent = ({message}: {message: string}) => (
  <ChatCompletion>
    <SystemMessage>Write a story about {message}</SystemMessage>
    <UserMessage>Tell me a tale</UserMessage>
  </ChatCompletion>
);

// usage
<MyComponent message="a brave knight"/>
```

this renders a tree of components asynchronously, calling the llm at each step which is wicked cool, as it allows for parallel processing and cool stuff like streaming.  imagine calling three different llm functions concurrently—one for the character, one for the setting, and one for the plot—and combining them into a single story output in real time.  that’s the power of async and parallel rendering in ai jsx.

another big idea was composability. the guy showed how you could wrap components within other components to add constraints or functionality. for example, he showed a `kidSafe` component that sanitized the output of any component wrapped inside it  he had a simple example:

```typescript
const KidSafeComponent = ({ children }) => (
  <ChatCompletion>
    <SystemMessage>Rewrite the following text to be safe for children:</SystemMessage>
    <UserMessage>{children}</UserMessage>
  </ChatCompletion>
);

// Usage
<KidSafeComponent>
  This content might be inappropriate for kids.
</KidSafeComponent>
```

this is where things got really interesting.   he showed how to use  `AIJSX` to easily hook into tools and apis using a `useTools` component. this basically lets your llm access external services, creating a rag (retrieval augmented generation) pipeline which  you could write in just a few lines— seriously  


here's a snippet showing how to create a tool that calls the github graphql api

```typescript
const githubTool = {
  name: "GitHub API",
  description: "Access GitHub data using GraphQL.",
  handler: async (query: string) => {
    const response = await fetch("https://api.github.com/graphql", {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${process.env.GITHUB_TOKEN}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ query }),
    });
    const data = await response.json();
    return data;
  },
};

const MyGitHubApp = () => (
  <UseTools tools={[githubTool]}>
    {/* Your app that uses the GitHub tool */}
  </UseTools>
);

```
  pretty straightforward and the component handled all the integration


the resolution was that ai jsx aims to make building ai apps way easier and more accessible to frontend devs using typescript  they showed a real-time voice interaction demo— seriously mindblowing—with almost zero latency, which was totally slick.  plus they've built a cloud platform fixie to handle all the backend stuff  so you can just focus on building cool ai powered stuff  this is big because it simplifies RAG pipelines allowing you to hook in your own vector databases and manage your llm interactions   the entire thing was a pretty compelling argument for their approach to make llm app development simpler and faster


the presentation was really well done, and i especially liked how they emphasized the visual cues like the "bear at the window" analogy to represent frontend devs wanting to get involved in ai dev, and the use of 2001 space odyssey music which was unexpected but super memorable. the live demo they showed was also impactful; even a recording showed off the impressive low-latency voice interaction. it’s a pretty convincing package overall.  definitely check it out if you’re into building ai apps
