---
title: "How to Build Interactive Web Pages with JavaScript"
date: "2024-11-16"
id: "how-to-build-interactive-web-pages-with-javascript"
---

dude so you won't believe this video i watched it's like a total mind-melt on how to build these super cool interactive web pages using javascript and all that jazz the whole thing is about making web pages that react to what you do like if you click a button something happens or if you hover over something it changes color it's basically magic but with code  it's all about making things dynamic and engaging not just those boring static pages everyone hates


 so the setup is they start by showing this super basic html page it's like the bare bones of a website just a heading and a paragraph  remember those days of learning basic html? this was way simpler than my first attempt which was a complete mess and looked like a digital train wreck haha you know it was like `<html><head><title>my first page</title></head><body><h1>hello world</h1><p>this is my epic page</p></body></html>` yeah it was a masterpiece of uninspired web design.  the video then builds on that super simple foundation, adding javascript step-by-step which is what makes it awesome. i'm talking about adding event listeners and stuff to make it interactive


one key moment was when they showed how to add an event listener to a button you know those things you click on websites. basically you tell javascript hey when someone clicks this button  do something  like change the text on the page or make an image appear or even play a sound. it's totally insane how much you can do with it  they showed a visual cue of the button changing color when clicked.  i'm talking about vibrant explosions of color right there on the screen so satisfying! another key moment was when they explained how to use `addEventListener`  it's like this magical function that lets you listen for all sorts of events clicks mouseovers key presses even things like the window resizing  i spent way too long looking at this function and scratching my head initially but trust me once you get it it makes so much sense.


then there was this whole section on manipulating the dom  the dom is like the structure of your web page it's like the skeleton the  bones of everything you see on the screen. so manipulating it means changing things like text colors  or making elements appear and disappear or even rearranging the whole layout on the fly.  it's like playing with lego but in digital form.  they showed this really cool example where a paragraph would change its text content when a button was clicked it was actually very funny because they used the example text: “i’ve been clicked!”


the other key concept is this thing called  `innerHTML`. it’s like a super cool superpower for changing content on your webpage. say you have a paragraph with some initial text you can totally change that text dynamically using javascript by targeting the paragraph's `innerHTML` property. and the best part is you don’t have to reload the entire page, which is what makes it so dynamic.


here's a little code snippet to illustrate this  imagine you have a button with the id "myButton" and a paragraph with the id "myParagraph"

```javascript
document.getElementById("myButton").addEventListener("click", function() {
  document.getElementById("myParagraph").innerHTML = "you clicked me";
});
```

see how simple that is?  you grab the button using its id then you attach an event listener that says hey when this button is clicked run this function and inside the function you change the `innerHTML` of the paragraph element


another cool thing they showed was how to use css to style things up. you know it's not just about functionality it's about making it look good too and that's where css comes in  they even showed how you could dynamically change css properties using javascript so you could make things glow or fade or whatever you want  i mean it's all about that user experience we're talking about here


another snippet. this time let's change the background color of the entire page when that same button is clicked

```javascript
document.getElementById("myButton").addEventListener("click", function() {
  document.body.style.backgroundColor = "purple"; // i chose purple, you can choose anything
  document.getElementById("myParagraph").innerHTML = "i've changed the background to purple you happy now?";
});
```


now that last one was epic because i had to change my background to purple every time i clicked it. not gonna lie i hated it but hey it showed me how powerful this simple snippet was. you can change it to any color you want, even make it change randomly on each click if you’re feeling fancy and adventurous, and i mean incredibly adventurous. also i added a sassy message just because i could


finally they even showed how to use javascript to fetch data from an api. an api is basically a way for different applications to talk to each other.  so you could use javascript to fetch weather data or stock prices or whatever  it's like accessing a massive database of information from the internet.  they showed a simple example of fetching some json data and displaying it on the page  that really blew my mind  i mean you're getting real-time information from a server into your web page it's like seriously advanced stuff


and here's a tiny taste of fetching data. this example isn't fetching real data but it gives you a sense of how to do it

```javascript
fetch('data.json') // replace data.json with your api endpoint
  .then(response => response.json())
  .then(data => {
    // update your page with the data received from the api
    console.log(data);
    document.getElementById('myParagraph').innerHTML = JSON.stringify(data);
  })
  .catch(error => console.error('Error:', error));
```

this bit is a little more complex you need to understand promises and asynchronous javascript to really get it but the basic idea is that fetch gets the data and the then functions process it then you can stick it into your page in a more dynamic way!


so the resolution basically is that javascript lets you build super interactive and dynamic web pages it's not just about static content anymore you can make websites that react to user input and fetch data from various sources and all sorts of fun stuff. it's like adding a brain to your website and that brain does some pretty amazing things honestly.  the video showed the power of javascript in making those really engaging interactive experiences that everyone loves and i was just blown away by how much i learned in such a short time.  i definitely felt like i leveled up my web dev skills  now i gotta practice and get even better!
