---
title: "recharts cartesian grid different style for vertical and horizontal lines?"
date: "2024-12-13"
id: "recharts-cartesian-grid-different-style-for-vertical-and-horizontal-lines"
---

Alright so you're trying to get recharts cartesian grid to behave right like different styles for vertical and horizontal lines I've been there man believe me this one's a classic pain point I remember way back when I first started tinkering with data viz and yeah recharts was a go-to for me it looked clean and simple which is always a trap you know

So the default grid in recharts right it's all one style all lines the same color width dash whatever and you're sitting there like no no I want the x-axis lines one way and y-axis lines another way and the thing is recharts doesn't give you a direct prop to split them up like that its approach is more like build your own damn it you know but thankfully not too hard let's walk through it

First off you gotta understand that the `<CartesianGrid>` component is really just drawing a single set of lines based on the props you pass to it no separate x and y logic hidden inside it so we have to take a different route here

The secret weapon is actually to use multiple `<CartesianGrid>` components layering them on top of each other one for the horizontal lines and another for the vertical lines and that's pretty much it

I remember back in my earlier days I tried to hack into the source code of recharts like directly modify the rendering logic you know what I'm talking about like injecting custom functions into the drawing calls I spent an entire weekend on it trying to bypass that "single grid component" limitation It was madness like trying to change the engine of a car while it's running you know needless to say that didn't work it was a lesson in not trying to reinvent the wheel but instead learning to use the tools properly and that's what we're doing here using the tools

So here's the first code snippet this is your basic setup to get those separated lines

```jsx
import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';

const data = [
  {name: 'Page A', uv: 4000, pv: 2400, amt: 2400},
  {name: 'Page B', uv: 3000, pv: 1398, amt: 2210},
  {name: 'Page C', uv: 2000, pv: 9800, amt: 2290},
  {name: 'Page D', uv: 2780, pv: 3908, amt: 2000},
  {name: 'Page E', uv: 1890, pv: 4800, amt: 2181},
  {name: 'Page F', uv: 2390, pv: 3800, amt: 2500},
  {name: 'Page G', uv: 3490, pv: 4300, amt: 2100},
];

const MyChart = () => {
  return (
    <LineChart width={700} height={300} data={data}>
        <CartesianGrid vertical={false} strokeDasharray="3 3" stroke="#ccc" />
        <CartesianGrid horizontal={false} stroke="#ddd" />
        <XAxis dataKey="name"/>
        <YAxis />
        <Tooltip/>
        <Legend />
        <Line type="monotone" dataKey="pv" stroke="#8884d8" activeDot={{r: 8}}/>
        <Line type="monotone" dataKey="uv" stroke="#82ca9d" />
    </LineChart>
  );
};

export default MyChart;
```

See how we have two `<CartesianGrid>` elements one with `vertical={false}` and another with `horizontal={false}`? The first one only draws the horizontal lines because it disables vertical ones and you can see the difference in `stroke` and `strokeDasharray` props for different look The other one only draws the vertical lines So now you have two independently style-able sets of grid lines.

Now you might be thinking ok great but what if I want to control the spacing or the number of grid lines you know not a default one you know well you can't directly manipulate these with the `<CartesianGrid>` component you would need to customize more directly you know This is why recharts gives the option to get more granular with `<XAxis>` and `<YAxis>` props.

Ok that previous example is good and works perfectly but we want to customize spacing a little so this snippet will be more useful to many people

```jsx
import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';

const data = [
  {name: 'Page A', uv: 4000, pv: 2400, amt: 2400},
  {name: 'Page B', uv: 3000, pv: 1398, amt: 2210},
  {name: 'Page C', uv: 2000, pv: 9800, amt: 2290},
  {name: 'Page D', uv: 2780, pv: 3908, amt: 2000},
  {name: 'Page E', uv: 1890, pv: 4800, amt: 2181},
  {name: 'Page F', uv: 2390, pv: 3800, amt: 2500},
  {name: 'Page G', uv: 3490, pv: 4300, amt: 2100},
];

const MyChart = () => {
  return (
      <LineChart width={700} height={300} data={data}>
          <XAxis dataKey="name" tickLine={false}/>
          <YAxis tickLine={false} />
            <CartesianGrid
              horizontal={false}
              strokeDasharray="3 3"
              stroke="#ccc"
              vertical={false}
             />
          <CartesianGrid
            horizontal={false}
            stroke="#ddd"
            vertical={false}
            />
          <CartesianGrid
            horizontal={true}
            stroke="#ccc"
            strokeDasharray="5 5"
            vertical={false}
            />
          <Tooltip/>
          <Legend />
          <Line type="monotone" dataKey="pv" stroke="#8884d8" activeDot={{r: 8}}/>
          <Line type="monotone" dataKey="uv" stroke="#82ca9d" />
      </LineChart>
  );
};

export default MyChart;
```

Look now at this code you can see there's 3 grid components the last one with horizontal lines and first two with vertical lines the first two are for spacing purposes and the last one is to customize the horizontal lines you see the tickLine in XAxis and YAxis is a flag that you can use to decide wether to show the tick lines or not but the line that is on the grid it self so we set it to false to hide the default tick line

By now you might be thinking what if I want more control over the number of lines and where they appear on the chart you know for instance grid lines that represent multiples of 5 well the solution is to use the `<XAxis>` and `<YAxis>` again but this time use the `ticks` prop instead of the `<CartesianGrid>` we are not gonna use the component at all and we will manage everything manually

Here we are the last code snippet that will solve almost everything and I think it's a good example to help many people in a similar situation

```jsx
import React from 'react';
import { LineChart, Line, XAxis, YAxis, Tooltip, Legend } from 'recharts';

const data = [
    {name: 'Page A', uv: 4000, pv: 2400, amt: 2400},
    {name: 'Page B', uv: 3000, pv: 1398, amt: 2210},
    {name: 'Page C', uv: 2000, pv: 9800, amt: 2290},
    {name: 'Page D', uv: 2780, pv: 3908, amt: 2000},
    {name: 'Page E', uv: 1890, pv: 4800, amt: 2181},
    {name: 'Page F', uv: 2390, pv: 3800, amt: 2500},
    {name: 'Page G', uv: 3490, pv: 4300, amt: 2100},
];

const MyChart = () => {
    return (
        <LineChart width={700} height={300} data={data}>
            <XAxis dataKey="name"
            ticks={[ 'Page A', 'Page C', 'Page E', 'Page G']}
            axisLine={{ stroke: '#ccc' }}
            tickLine={{ stroke: '#ccc' }}
        />
            <YAxis
            ticks={[0,2000,4000,6000,8000, 10000]}
             axisLine={{ stroke: '#ddd' }}
             tickLine={{ stroke: '#ddd' }}
           />
            <Tooltip/>
            <Legend />
            <Line type="monotone" dataKey="pv" stroke="#8884d8" activeDot={{r: 8}}/>
            <Line type="monotone" dataKey="uv" stroke="#82ca9d" />
        </LineChart>
    );
};

export default MyChart;
```

Ok so what is happening here well we are setting `ticks` for `<XAxis>` and `<YAxis>` to be the explicit values we want and since we set the `tickLine` props in the `XAxis` and `YAxis` those tick lines will act as the horizontal and vertical grid lines you just set the color `stroke` and you are done the beauty of it is that you have full control over the data that will show up as grid lines and you can use any range or interval you desire it is not only that you can also change the style of the horizontal and vertical lines with the axisLine stroke just check it out.

I once spent an afternoon trying to get this to work and I was frustrated by all the attempts I made until I found out that recharts handles that for you in these props it's like you go through the entire forest only to find out there's a path you could just take all along you know but the experience of trying to solve it is important it makes you a better programmer I guess I had to make that mistake so you don't make it

And as a small joke why do programmers prefer dark mode because light attracts bugs you know like you wouldn't debug in a sunny park so yeah we keep it dark.

If you're looking to dig deeper into data visualization principles and why certain choices are made you know it's not just about code then check out “The Visual Display of Quantitative Information” by Edward Tufte and “Information Visualization: Perception for Design” by Colin Ware these books go into much depth about design principles and theory behind effective data visualization it's really worth a read if you are looking to become an expert in data visualization.

Anyway that's how I'd approach this problem I hope this helps you and good luck on your data viz journey feel free to ask if there's more you know.
