---
title: "bubble chart in d3.js?"
date: "2024-12-13"
id: "bubble-chart-in-d3js"
---

Okay so bubble chart in D3js you say yeah I've tangled with that beast a few times its a classic visualization but theres always some little detail that throws you off right? Alright lets break it down like we’re debuggin some gnarly code together

First off lets be super clear D3js is powerful you can pretty much make anything you want but that also means you gotta build it up almost from scratch which is why I like to say its like a lego kit for data viz not a pre-built playmobil set.

So we are doing a bubble chart right which fundamentally is just a bunch of circles on a plane each circle's size corresponds to a value and maybe its position or color also means something. So let's get some basic structure down. You have your data you've likely structured it something like an array of objects each object with say an ID a value and maybe a label

```javascript
const myData = [
  {id: "a", value: 10, label: "Category A"},
  {id: "b", value: 25, label: "Category B"},
  {id: "c", value: 15, label: "Category C"},
  {id: "d", value: 30, label: "Category D"},
  {id: "e", value: 5, label: "Category E"}
]
```

Thats standard stuff Nothing complicated there. Now comes the D3js magic. First you setup your SVG canvas where everything will live right you create it and append it to your webpage.

```javascript
const width = 600
const height = 400
const svg = d3.select("body") // Or whichever element you want to append to
                .append("svg")
                .attr("width", width)
                .attr("height", height)
```

Cool SVG ready for some circles Now the core logic is binding your data to SVG circles D3 does this with `.data()` and `.join()` its crucial to get this right.

```javascript
const circles = svg.selectAll("circle")
                 .data(myData)
                 .join("circle")
                 .attr("cx", (d, i) => (i+1) * width / (myData.length+1)) // Basic equidistant placement
                 .attr("cy", height/2) // Center vertically for now
                 .attr("r", d => Math.sqrt(d.value) * 5) // Radius based on value
                 .attr("fill", "steelblue") // A default color can change it
                 .attr("stroke", "black") // Give them a border
```

Notice the radius calculation I am using `Math.sqrt(d.value) * 5`. Why square root? Well if you just scale the radius directly to the value your big bubbles will take over the entire space because areas grow quadratically with radius. Square root helps keep it a bit balanced for the eye. Also that 5 is a scaling factor play with that number to get a bubble size you like.

I remember this one time I was building a bubble chart for some marketing data I was just starting with D3 then I forgot this square root detail and the largest bubble covered most of the screen and looked terribly misbalanced I had to rewrite it from scratch that took me an evening of painful debugging.

Okay but what if you want more interesting bubble placement? Right now I am just spacing them out evenly horizontally. This is where D3's force simulation becomes relevant but not in this case I want to be clear I wont go into full force layout details because it's a bit advanced for a starting point I will just touch on it briefly in the next steps.

For now let’s try something basic. A little bit of random y variation so they are not aligned on a straight line. I mean that will make it feel a bit less static right.

```javascript
const circles = svg.selectAll("circle")
                 .data(myData)
                 .join("circle")
                 .attr("cx", (d, i) => (i+1) * width / (myData.length+1))
                 .attr("cy", () => Math.random() * height) // Random y placement
                 .attr("r", d => Math.sqrt(d.value) * 5)
                 .attr("fill", "steelblue")
                 .attr("stroke", "black")
```
Good this creates a more organic layout rather than a line of bubbles. Now I would like to address the other critical piece namely labeling. If you have your labels you need to display them along the circles. D3 makes this quite easy actually and very practical so.
```javascript
const labels = svg.selectAll("text")
                 .data(myData)
                 .join("text")
                 .attr("x", (d,i) => (i+1) * width / (myData.length+1))
                 .attr("y", d => Math.random() * height + Math.sqrt(d.value) * 5 + 10) // Slightly below the circle
                 .attr("text-anchor", "middle") // Center labels
                 .text(d => d.label)
                 .attr("font-size", "12px") // Size
                 .attr("fill", "black");
```

Here I am setting the x positions based on the circle's x also the y position will follow the circle y and then you just push the label a bit more down to make it look more visually clear and tidy and then the label will appear under it.

This is also a practical place where you can incorporate interactivity right You may want the circles to change their color on hover or show a tooltip if the user hovers the mouse over it. Lets create an hover effect for the circle this can be accomplished through the `.on("mouseover",...)` function in D3 that attaches specific functions on hover of the mouse and mouseout event for the opposite effect:
```javascript
circles
  .on("mouseover", function(event, d) {
      d3.select(this).transition()
          .duration('100')
          .attr('fill', 'red'); // change to red when the mouse is over
  })
  .on("mouseout", function(event,d) {
       d3.select(this).transition()
         .duration('100')
         .attr('fill', 'steelblue')// change to steelblue when the mouse is out
  });
```

So there you have it. A simple and functional bubble chart using D3js. This is of course just the beginning it is up to you to make it beautiful and more interactive. You can use d3 scale functions for color gradients you can add legends and also make the whole thing responsive. Its all in your hand now.

Now if you wanna go deep like really deep you need to understand the math behind layouts and that's no joke that involves a bit of linear algebra and optimization theory. It is not really that difficult if you understand it in a visual way but it takes time to digest everything.

If you are interested you can check books like "Data Visualization Handbook" by Claus O. Wilke. That's my go-to resource and it goes well in depth when dealing with data visualization techniques. Also for the math stuff "Linear Algebra Done Right" by Sheldon Axler is a good introduction to linear algebra in a more visual and intuitive way. Don't get intimidated by the math it is a tool like anything else it helps you to create better and more efficient code.

Now there is more but the code examples given in this answer will likely get you started with what you need. Remember this is an endless journey right you learn by doing and failing and fixing so go out there start coding and have fun because that is how you learn in the end.

Oh yeah and you might have problems with d3 at first but hey nobody got it right the first time. It’s like trying to catch a greased pig sometimes. You just have to keep at it and not give up.
