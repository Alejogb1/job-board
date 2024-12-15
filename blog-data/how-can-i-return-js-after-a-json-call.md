---
title: "How can I return JS after a JSON Call?"
date: "2024-12-15"
id: "how-can-i-return-js-after-a-json-call"
---

alright, so you're having trouble getting javascript code back after a json call, i've been there. it sounds like you're expecting a json endpoint to just magically return javascript code for you to execute, and that's... not how it typically works. let me tell you, i learned this the hard way back in '08 when i was building this horrendous flash-based web app. i had this server-side api that was spitting out data in some weird custom format, and i tried, god help me, to `eval` it directly on the client side. it was a security nightmare and also incredibly brittle, don't ever do that. please.

first things first, json (javascript object notation) is specifically designed for data serialization. it's meant to be a lightweight format for transmitting data between a server and a client, and that's it. it's not meant to contain executable code. the browser, by design, will not just execute javascript strings pulled out of json. that would be a massive security vulnerability. imagine if you could just inject javascript code through some api endpoint. chaos. utter chaos.

think of it this way: json is like a neatly labeled box of ingredients. it tells you what things are, what their properties are, but it doesn't include any instructions or recipes on what to *do* with those things. javascript, on the other hand, is the recipe—it's the actual set of instructions for how to make something happen in the browser.

so, when you're making an ajax call (which is what i'm assuming you are doing), and getting a json response, you're just getting that nicely formatted box of data. it’s up to *your* javascript code to interpret this data and then use it to manipulate the page.

the key thing you need to understand is that the javascript code that you want to execute *must* be on the client side *already*. your javascript code is what makes the ajax call and after that your javascript code handles the response. the server simply responds with the information, the data, the ingredients, but never the instructions.

so how do you actually achieve what you want? well, there are several standard approaches. usually what i would do, is have my client side javascript code, that, based on some data from the json response decides what to do. it’s something like a data-driven approach. for example, you may have a json response like this:

```json
{
  "action": "show_modal",
  "modal_id": "my-cool-modal",
  "modal_content": "this is the content for the modal!"
}
```

and then, in your javascript client side, you would do something like this:

```javascript
fetch('/api/get_modal_data')
  .then(response => response.json())
  .then(data => {
    if (data.action === 'show_modal') {
      const modal = document.getElementById(data.modal_id);
      modal.querySelector('.modal-content').textContent = data.modal_content;
      modal.style.display = 'block';
    } else if(data.action === 'update_div') {
      const div = document.getElementById('target-div')
      div.innerHTML = data.div_content
    } else {
       console.log("unknown action!")
    }
  })
  .catch(error => console.error('error fetching data:', error));
```

in this case, the server is sending instructions to the client in the form of *data*, not executable javascript code. we are saying to ourselves, if you get an action type called `show_modal`, you will show the modal, if you get an action type called `update_div` you will update a certain div, if not you will output to the console, the important thing is the javascript client code is what makes the decision of what to do.

this example is pretty basic, but it illustrates the point. you use the data from the json response to control the behavior of your existing javascript code.

let's say, as a variation, you want to dynamically create a javascript object with a function, after receiving the json data. that could also be achieved easily using the same approach. for example, suppose you have the following json data:

```json
{
  "object_name": "myDynamicObject",
    "function_name": "myDynamicFunction",
  "property1": "value1",
    "property2": "value2"
}
```

you could then parse this information, and dynamically create an object with a new function:

```javascript
fetch('/api/get_object_config')
  .then(response => response.json())
  .then(data => {
    const obj = {};
    obj[data.function_name] = function() {
        console.log(`properties are: ${data.property1} and ${data.property2}`);
    }
    window[data.object_name] = obj;
    window[data.object_name][data.function_name]();
  })
  .catch(error => console.error('error fetching config:', error));
```

in this snippet, the server provides the property name and the method name that will be added into the object, but, what the function does is predefined in the javascript. the point is, you can orchestrate or parameterize your javascript behavior by sending appropriate data, not by sending actual javascript code to execute.

now, another fairly common use case is when you are rendering complex parts of a page, let’s suppose that you have a server that does the rendering of an html element, and you want to retrieve this rendering after the page is loaded. in that case, you will have your server to prepare the rendering of that specific part of the page, in html, and then you would set the innerhtml of the target container. something like this:

```javascript
fetch('/api/get_html_snippet')
  .then(response => response.json())
  .then(data => {
      document.getElementById('target-container').innerHTML = data.html
    })
  .catch(error => console.error('error fetching snippet:', error));
```
in the server you could be rendering a specific html element, or a section of the page with its own complex logic, and returning the pure html. notice that there is no javascript code involved. this html can also include javascript events, so if you have interactive elements like buttons they will still work. you will be effectively sending the `html` recipe, but the code that is going to handle those actions is still predefined.

but what if you *really* need to execute some javascript dynamically? well, you are usually approaching this problem the wrong way. there is this function `eval`, it can evaluate javascript code that you send as a string, but, again, *do not use it*. the internet has plenty of articles and warnings about it. `eval` is a security risk and can introduce hard-to-debug issues, and you should usually, never ever use it, and if you find yourself in the position to use it, you are most likely doing something terribly wrong. using `eval` is like using a rocket launcher to swat a fly, and a fly in this case, could be a potential hacker. why would you risk that?

if, for some insane reason, you need code evaluated on the client, you should use code compilation on the server, you can send a string to your server, your server compiles it and sends the compilation results back to the client. but you should *never* compile that code on the client. there are better ways for almost every situation, so using `eval` is just bad praxis.

if you really want to dive deeper into these concepts, i recommend looking into books and papers on topics like web security best practices, data-driven design, and the principles of api design. look up "restful web apis" by leonard richardson and sam ruby it is a classic. also, "javascript: the definitive guide" by david flanagan is a must-have reference for any serious javascript developer.

in summary, the golden rule when working with json is to treat it as a means to an end—a way to transfer data, not to transfer execution instructions. keep your client-side javascript logic on the client, and use the json data to control that logic, not to inject code, and your web app will be more secure and easier to maintain. oh and by the way, i once had a json endpoint returning a html with a javascript inside, it took me 3 days to debug it (i was starting to think it was cursed) until i realized i was doing that, so just don't ever do that. it's like trying to drive a car using a map of a bicycle, it just doesn't work.
