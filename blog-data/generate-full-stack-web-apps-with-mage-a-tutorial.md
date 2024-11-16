---
title: "Generate Full-Stack Web Apps with Mage: A Tutorial"
date: "2024-11-16"
id: "generate-full-stack-web-apps-with-mage-a-tutorial"
---

dude so i just saw this amazeballs video about this thing called mage  it's basically this ai-powered web app generator that's like totally bonkers  it pumps out full-stack react/node.js apps in minutes and they built over 10k of them in a single month  insane right

the whole setup is ridiculously simple  you go to this webpage it’s like some kinda magical portal and you just chuck in some info  first you name your app  like "todo-app-supreme" or whatever ridiculous name you can think of then you give it a tiny description like "a super-duper todo app where you can add delete and complete tasks with an optional sprinkle of unicorn glitter"  then you tweak a "creativity" slider this is basically the gpt temperature so higher means more features but potentially more glitches it’s like telling gpt "go wild but don't break anything please"  then boom you click "generate"  and poof a whole web app appears

one of the coolest visual cues is seeing the app generate right before your eyes  it's like watching a digital god build your app from scratch it's mesmerizing  another cool visual is the database inspector that pops up they showed the database updating live as they added todo items super neat.  and finally, the "bam" sound effect when the app generates is just chef’s kiss pure comedic genius.

the core ideas here are wasp and the clever way they're using gpt  wasp is a full-stack framework that's all about this super-clean configuration file it's basically a high-level description of your entire app in a single file  think of it like the blueprints for your app but way cooler it’s all declarative  you say *what* you want, not *how* to do it and wasp figures out the rest this makes things super easy for gpt to understand and generate code


here’s a little snippet showing a basic wasp config file:


```javascript
{
  "name": "my-awesome-app",
  "routes": [
    {
      "path": "/",
      "page": "HomePage"
    },
    {
      "path": "/about",
      "page": "AboutPage"
    }
  ],
  "database": {
    "provider": "prisma",
    "models": [
      {
        "name": "TodoItem",
        "fields": {
          "id": { "type": "Int", "primaryKey": true },
          "text": { "type": "String" },
          "done": { "type": "Boolean" }
        }
      }
    ]
  }
}
```

this config file dictates everything the routes pages data model it's all there  it makes the whole app generation process way simpler than trying to generate individual files

and then there's how they use gpt   it's a three-stage process  "step zero" is where wasp does its magic generating basic config files authentication logic and other boilerplate  then the gpt-powered "code agent" takes over it has three phases: planning where it figures out what needs to be built code generation which is where the actual code is written and finally error fixing  it tries to squash bugs on its own  it's like having a little robotic programmer that’s weirdly good at fixing its own errors.


here's a bit of example react code the video implied gpt would generate (i’m simplifying of course):

```javascript
import React, { useState, useEffect } from 'react';
import axios from 'axios';

function TodoList() {
  const [todos, setTodos] = useState([]);

  useEffect(() => {
    const fetchTodos = async () => {
      const response = await axios.get('/api/todos');
      setTodos(response.data);
    };
    fetchTodos();
  }, []);

  // ...rest of the component to add, edit, and delete todos
}
```

this is a simplified representation but you get the idea  gpt's role is to flesh out the functionality defined in wasp’s config


and finally a bit of node.js backend code:

```javascript
const express = require('express');
const { PrismaClient } = require('@prisma/client');

const app = express();
const prisma = new PrismaClient();

app.get('/api/todos', async (req, res) => {
  const todos = await prisma.todoItem.findMany();
  res.json(todos);
});

// ... other API endpoints for adding, updating, and deleting todos

app.listen(3000, () => console.log('Server listening on port 3000'));

```


they cleverly use gpt-3.5 for most of the coding because it's faster and cheaper using gpt-4 only for the initial planning phase which requires more creative problem-solving  this is a huge cost optimization  it's like using a fancy sports car for the tricky parts of a race and then a reliable sedan for the rest this keeps costs down massively  without this trick it would have been 10x more expensive


the resolution is pretty clear mage is awesome for generating full-stack web apps super fast and cheaply  but it's not a magic bullet you still need to understand what you're building  it’s a great starting point not a replacement for a developer  they’re even looking to add live debugging and maybe even fine-tune an llm specifically for wasp which would make things even more efficient.  it is very definitely the future of sas starters

so yeah  that's mage  it's mind-blowing and kind of terrifying at the same time.  but mostly mind-blowing  go check it out  if nothing else, the video alone is worth watching for the sheer entertainment value.  the “bam” is iconic
