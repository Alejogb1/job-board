---
title: "too many active webgl contexts problem?"
date: "2024-12-13"
id: "too-many-active-webgl-contexts-problem"
---

Okay so you've got the classic "too many active WebGL contexts" blues eh I've been there trust me I've seen the depths of that particular rabbit hole and come back with a few battle scars and a whole lot of wisdom to share This isn't a fun place to be it can make your app choke like it's trying to swallow a whole pizza whole and leave your users staring at a blank screen or worse a frozen one

First off what's usually happening here is a simple case of overenthusiasm You're creating WebGL contexts like they're going out of style and you aren't properly cleaning up after yourself It's like leaving a pile of dirty dishes after every meal eventually the kitchen becomes unusable your browser is the kitchen and WebGL contexts are the dirty dishes it's a messy analogy but you get the point right

I once had a project where we were trying to render a bunch of 3D models in a user-controlled viewport It was this cool data viz thing where each model represented a data point you know fancy stuff Problem was every time the user zoomed or panned the viewport we were creating a new WebGL context for rendering the new frame because I used the wrong mindset which was an amateur one I believed I need to render everything from scratch every time a dumb error it is This was a nightmare It led to memory leaks terrible performance and users sending me angry emails with subject lines like "My browser is melting" and one of them "are you trying to destroy the world with your program" haha I kid you not

So what did I do We had to do a full rewrite and we did it using a few tricks

**Tip 1: Context Sharing/Reusing**

The core of the issue you're dealing with is not understanding when to reuse your webgl contexts. Instead of creating a new context every time you need one try reusing an existing one if the rendering context allows it If you have canvas elements for multiple objects consider using the same WebGL context across multiple canvas and objects instead of creating a new one per canvas

Here's a simplified example showing how you might approach context sharing:

```javascript
let gl; // Global variable to store the WebGL context

function getWebGLContext(canvas) {
    if (!gl) {
        gl = canvas.getContext("webgl") || canvas.getContext("experimental-webgl");
        if (!gl) {
          console.error("WebGL not supported");
          return null
        }
      console.log("creating a context")
    }
  console.log("using an already created context")
    return gl;
}


function renderScene(canvas) {
  const context = getWebGLContext(canvas)
    if (!context) return;
  // Now render here using the 'context'
   context.clearColor(0.0, 0.0, 0.0, 1.0);
  context.clear(context.COLOR_BUFFER_BIT);

  // Add your rendering logic here
}

const canvas1 = document.getElementById('canvas1');
const canvas2 = document.getElementById('canvas2');
if (canvas1 && canvas2){
renderScene(canvas1);
renderScene(canvas2)
}
```

In this snippet we first create the context inside a function called `getWebGLContext` and before creating it checks if we have already created it before using the global variable `gl` if we haven't created it the function creates it and sets the value of `gl` otherwise we just return the stored one We then use this function in the `renderScene` function and for any canvas we need to render to We can reuse our context like this.

**Tip 2: Context Loss Management**

WebGL contexts are not invincible you know they can get lost especially when users switch tabs or minimize the browser window It's important to handle `webglcontextlost` and `webglcontextrestored` events properly.

Here's a basic way to handle context loss:

```javascript
const canvas = document.getElementById('myCanvas');
const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');

if (!gl) {
    console.error("WebGL not supported");
}

let rendering = true

function renderLoop() {
    if (rendering) {
        // Your rendering code here
        gl.clearColor(0.0, 0.0, 0.0, 1.0);
        gl.clear(gl.COLOR_BUFFER_BIT);

        requestAnimationFrame(renderLoop);
    }
}

renderLoop() // Initial call

canvas.addEventListener('webglcontextlost', function(event) {
    event.preventDefault();
    console.log('WebGL context lost');
    rendering = false
}, false);

canvas.addEventListener('webglcontextrestored', function() {
    console.log('WebGL context restored');
    rendering = true
    renderLoop()
}, false);
```

Here what I have done is adding an event listener for `webglcontextlost` which makes sure the render loop is stopped when the webgl context is lost which prevents our program from running into errors. The `webglcontextrestored` adds an event handler for when it has been restored which will then make the program continue to work. This is crucial and will save you from tons of headaches.

**Tip 3: Explicitly Destroying Contexts**

When you're absolutely sure you don't need a WebGL context anymore destroy it explicitly This releases resources back to the browser. Remember that not every context we create should be shared we need to have our own way to destroy them safely

```javascript
let gl;

function createWebGLContext(canvas) {
    gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
    if (!gl) {
        console.error("WebGL not supported");
        return null;
    }
    console.log("created a context")
    return gl;
}

function destroyWebGLContext(canvas) {
    if (gl) {
        const extension = gl.getExtension("WEBGL_lose_context");
         if (extension)
         {
              extension.loseContext();
              gl = null
             console.log("destroyed the context")
         }else {
            console.error("could not get extension WEBGL_lose_context")
         }
    }
}

const canvas3 = document.getElementById('canvas3');
const context3 = createWebGLContext(canvas3)

if (context3) {
    // do some rendering
    context3.clearColor(0.5,0.5,0.5,1)
    context3.clear(context3.COLOR_BUFFER_BIT)
    // now destroy it
     destroyWebGLContext(canvas3)

}
```

In this example we have made a function `destroyWebGLContext` that when called will explicitly destroy a webgl context created by the `createWebGLContext` This is very useful when you are doing something like switching between scenes for example or when you know for a fact you no longer need the context. You might need to re-render a scene using the same context but then you will need to use a shared context.

**Further Reading**

There are some good resources on the WebGL spec that can help with this. I highly recommend reading Khronos Group's official WebGL specification. It is available online and a good understanding will help you a great deal. There's also the WebGL Programming Guide which can be found in any of the major book vendors which will give you a lot more practical knowledge. This book has code snippets that actually works which I find the most useful when I need to know how something works. There are also various blogs and articles but most of them miss some very important details and only focus on a small part of the problem at hand. If you want to really understand how to properly handle webgl and its contexts you should be following standards from Khronos itself. There is also a good article by Mozilla that covers webgl in general and it will help you in the long run.

**Final thoughts**

Don't create webgl context like its a habit you should only create one when it's absolutely necessary remember to share them and destroy them when needed also remember to handle context loss that's not everything but that is the core of solving this particular issue which will save you a lot of headaches.

Good luck and may your frame rates be high and your context counts be low!
