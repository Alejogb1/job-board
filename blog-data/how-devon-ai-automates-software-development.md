---
title: "How Devon AI Automates Software Development"
date: "2024-11-16"
id: "how-devon-ai-automates-software-development"
---

dude so this video totally blew my mind it's scott from cognition ai talking about devon their like super-duper early-stage ai software engineer  think of it as an ai that writes code not just spits out code but actually plans designs debugs and deploys it's wild  the whole point is showing how they're building this thing and hinting at how it'll change the game for software engineers

first off the demo was killer he asked devon to build this name game website on the fly  like literally he throws a tsv file of names and faces at it says "yo devon make a mobile-friendly site let me play a name-matching game with these peeps"  it's insane how he just speaks naturally totally conversational no coding needed


the visuals were great too you see devon churning out a project plan in real time  it's not some static thing it's constantly changing based on scott's feedback it's like watching a software engineer think  then it's making directories firing up a react app  the whole thing just happens all in front of your eyes. another super cool visual cue was when devon actually created a pull request in git  that's not just some simulated output its showing devon integrating with standard dev tools

one key idea that stood out was devon's ability to plan it's not just some brute-force coding monkey  it actually strategizes creates a plan it dynamically updates this plan and adapts it based on feedback and changes  like imagine building a house you wouldn't just start slapping bricks together right you'd make blueprints and adjust them as needed devon's like that but for code


another crucial concept is this iterative feedback loop  scott gives devon instructions devon spits out code scott gives feedback devon improves the code  it's almost like a conversation where devon's learning and getting better with each interaction  it's not just fire and forget its this elegant cycle of improvement that's so important for software development


let's get into some code snippets because that's where the real magic happens


snippet 1: basic react component creation

```javascript
// devon probably generated something like this to display a name
function NameCard({ name, image }) {
  return (
    <div className="name-card">
      <img src={image} alt={name} />
      <h3>{name}</h3>
    </div>
  );
}
// simple easy to understand and exactly what you'd need to display info from a tsv file
```

that's literally the kind of thing you'd see devon crank out to display the names and faces from that tsv file super straightforward

snippet 2: handling user interaction

```javascript
// devon probably did something like this to handle clicks and update the score
function handleGuess(isCorrect) {
  if (isCorrect) {
    setScore(score + 1);
  } else {
    setScore(0);
  }
}
// this showcases how devon manages user interaction  updating scores and reacting to player choices.
```

this bit deals with the core gameplay  handling the user's guess updating the score and the streak  again dead simple but shows the logical flow

snippet 3: integrating with external services

```javascript
// hypothetical example of devon interacting with a third party api for image fetching

async function fetchImage(name) {
  const response = await fetch(`https://some-api.com/images/${name}`); // replace with actual api endpoint
  const data = await response.json();
  return data.imageUrl; // assumes api returns url for the image
}
// this bit is about devon's ability to grab images perhaps from a cloud storage bucket or a third party image api  it's showing the flexibility to interact with systems beyond just its own code.
```

this highlights devon's ability to interact with the outside world  fetching images maybe from a cloud storage bucket or an image api


the resolution is pretty clear devon is not going to replace human software engineers  it's not going to take over jobs it's more like a superpower for them  scott says it frees up devs to focus on the "thinking part" the architecture the problem solving  leaving devon to handle the grunt work the implementation  it's about amplifying human capabilities not replacing them

it's pretty mind-blowing right you get to spend way more time thinking about the problem and way less time dealing with the boilerplate  which is a huge deal  plus it's still very early days for devon so the possibilities are literally limitless

i know it's a lot to take in but that's just the tip of the iceberg this devon thing is a massive shift in how software gets built  and it's super fun to watch it evolve
