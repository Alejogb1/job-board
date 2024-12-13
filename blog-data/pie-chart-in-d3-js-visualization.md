---
title: "pie chart in d3 js visualization?"
date: "2024-12-13"
id: "pie-chart-in-d3-js-visualization"
---

Okay so you want to build a pie chart in d3js right been there done that a million times its a classic visualization task nothing too fancy but yeah there are some gotchas if you are not careful lets get into it

First off let's address the data format you'll need to have your data in a usable format for d3 usually an array of objects where each object has a key representing the label and a key representing the value something like this:

```javascript
const data = [
    {label: "Apples", value: 30},
    {label: "Bananas", value: 50},
    {label: "Cherries", value: 20}
];
```

This is what d3 likes for most charts if your data is different lets say its in json or a csv well youll need to load it and parse it into this array of objects format that will not be a problem at all if you use `d3.json` or `d3.csv` functions but I will assume that you already got this part done

Now the crucial part of any pie chart is the `d3.pie()` function this is what transform the array of objects into something that is ready for a pie shape the function needs to tell the pie what the value part of the object you want to create a pie from looks like it uses an accessor function like this

```javascript
const pie = d3.pie()
  .value(d => d.value); // specify the value accessor
```

This `pie` is a layout function it is not really d3s drawing function it takes the data and it returns an array of objects but those objects have a startAngle endAngle data and others its the main part of how d3 knows what to draw as an arc and its crucial to get it right

After you create the `pie` layout it's time to create the arc generator its d3's way of saying okay now lets transform the angles from the layout into a path a `d` attribute in an SVG which is what it will be rendered as think of it as the final step to getting the arc path strings we will use `d3.arc()` and we will configure its inner and outer radius like this:

```javascript
const arc = d3.arc()
  .innerRadius(0)    // change if you want a donut chart
  .outerRadius(100); // adjust the radius to your needs
```

The `innerRadius` will give you a donut chart if you set it to something else than `0` so if that is what you want make sure you change it and the `outerRadius` sets how big the pie is overall again very important to set this right and the units are pixels so if you set `100` it means that the radius will be 100 pixels so pay attention to the size of your visualization element container

Now lets create a svg element or find the existing one in your html code and select it via d3 like this:

```javascript
const svg = d3.select("body") // or wherever your svg is
  .append("svg")
  .attr("width", 250)
  .attr("height", 250);
```

this creates a svg element inside of the body tag with a width and height of 250 by 250 and now for the last step which is actually drawing the shapes which will use the pie layout and the arc function

```javascript
// center the pie
const g = svg.append("g")
    .attr("transform", `translate(${250/2}, ${250/2})`); // centering via translate

// this next code makes the magic happen
g.selectAll("path")
  .data(pie(data))     // this passes the data array to the pie function and links them via d3
  .enter()
  .append("path")      // each part of the pie gets a path element
  .attr("d", arc)      // the arc function generates the final path
  .attr("fill", (d, i) => d3.schemeCategory10[i]); // assigning a color based on the index to each pie part using schemeCategory10 its colors that already look good
```

This code does a few things

first we create a `g` element which is a grouping of SVG elements its common practice to create this when dealing with visualizations this group tag is where the pie is going to be placed after that we use `selectAll` to get all paths but there arent any so it is just selects an empty collection the data function binds the result of the `pie` layout function and then enter creates a new `path` for each piece in the layout then the magic happens the `d` attribute is created using the arc function which turns the angle data into a string that can be used by svg and finally the color is assigned from the d3's color schemes

The `d3.schemeCategory10` gives you a nice default color scale its a good start but you can change to other color scales or create your own in any way that you want

In my past experience with pie charts I remember that one time I was really confused about the angle calculations because I started doing some calculations by myself instead of using d3 I ended up having this ugly overlapping pie where some parts were going outside of the SVG I had to spend almost 3 hours debugging the angles to realize that using `d3.pie()` and `d3.arc()` was the correct and efficient way to do this lesson learned from that time is dont reinvent the wheel especially if it involves geometric calculations.

Another problem that people tend to face is the labeling part a pie without a label is not a great pie you need the text right?

The solution is again quite straightforward but there are a few tricks you might want to use let's create a function that places labels around the pie for you:

```javascript
function placeLabels(selection){
  selection.append("text")
      .attr("transform", function(d) {
          const centroid = arc.centroid(d);
          return `translate(${centroid})`;
        })
      .attr("dy", ".35em")
      .style("text-anchor", "middle")
      .text(d => d.data.label);
}
```

This function appends a text to the center of each pie part and it takes the layout information and generates the text by using `arc.centroid` which gives you the exact position of the center of a pie slice and uses the original label from the data and also styles a little the text making it more presentable

And now the way we use this in our pie chart is by adding:

```javascript
g.selectAll("g.slice")
  .data(pie(data)) // binds data
  .enter()
  .append("g")
    .attr("class", "slice") // classes
    .call(placeLabels)  // here is where the magic happens
    .append("path") // here is the path
    .attr("d", arc)
    .attr("fill", (d, i) => d3.schemeCategory10[i]);
```

As you see we created a group element `g` for each of the pie parts and assigned a class and then called the function to do the label placing inside that group and as you see we put the label group before the path element so they are placed on the correct z index layer in the svg document

Finally if your data changes you will need to rebind the data to the chart in this case you can create a function to update the visualization:

```javascript
function updatePieChart(newData){
  g.selectAll(".slice")
      .data(pie(newData))
    .join(
    function(enter){
      const gEnter = enter.append("g").attr("class", "slice")
            .call(placeLabels)

      gEnter.append("path")
          .attr("fill", (d, i) => d3.schemeCategory10[i])
          .attr("d", arc);
      return gEnter;
      },
    function(update){
      update.selectAll("path")
        .transition()
        .duration(300)
        .attr("d", arc)
        .attr("fill", (d, i) => d3.schemeCategory10[i]);

      update.selectAll("text")
           .text(d => d.data.label)
           .attr("transform", d => `translate(${arc.centroid(d)})`);
      return update
    }
    ,
    function(exit){
      exit.transition().duration(300).remove();
    }
  )
}
```

This is a little bit more complex it uses the `join` method which is a newish method in d3 to handle the data updates this `join` does three things an `enter` method for when there are more data items than currently shown a `update` for when data has been updated and an `exit` for when you removed one element from the data

This is the common pattern that d3 does for updates and animation

One important thing to keep in mind is performance when dealing with a large number of pie slices this can become slow there are certain optimization techniques that can help reduce jankiness but the gist is dont render 1000 slices if you dont have to aggregate data as much as you can

Now lets talk about resources on d3 there are plenty but a good start is the original book by Mike Bostock the creator of d3 called "Interactive Data Visualization for the Web" this is kind of old but the core principles of d3 are all there its a very very solid starting point there are other good books like "Fullstack Data Visualization with D3" which is a bit more up to date

Also some really good resources are the official D3 API documentation it is really well written and full of examples it explains really well all d3's modules and also check some articles from "Observablehq" which is basically a notebook environment where many famous d3 creators work in there you can learn some new tricks and also see how pros deal with complicated visualizations

And finally the last advice I can give you is to practice a lot. Seriously that is where you will learn the most by building stuff and also by debugging stuff after all its the way that we all learn in this industry right

Oh and also this code its not just a pie in the sky idea it really works trust me on that unless your browser doesn't render SVG which would be a real head scratcher right? anyway good luck with your charts
