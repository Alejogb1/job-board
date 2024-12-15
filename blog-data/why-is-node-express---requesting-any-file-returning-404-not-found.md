---
title: "Why is Node Express - requesting any file returning 404 not found?"
date: "2024-12-15"
id: "why-is-node-express---requesting-any-file-returning-404-not-found"
---

ah, 404s with node express when trying to serve files, a classic. i've definitely been there, staring at the screen wondering why my perfectly named file is suddenly invisible to the world. let's break this down, because usually it’s something pretty straightforward, yet incredibly annoying to miss.

from what i'm getting from the question, you're expecting express to simply hand over files when asked, but instead, you’re being met with that dreaded 404 not found. this means express isn't able to locate the files, and that points to a few common culprits.

first off, let's talk about the express setup. it's not magic; express needs explicit instructions on where to look for those files. by default it does not serve static files unless you tell it to. a very common mistake is assuming that it will automatically grab files from your project directory. it will not. you need to utilize a thing called middleware which is something that runs in between the request of the user and your response to that request.

the most used middleware for this specific task is `express.static()`. this middleware tells express to serve static files from a specific directory. this is your first place to double-check. is `express.static()` even being used in your code? and if it is, is it pointing to the actual directory where your files are?

here's what a basic setup using `express.static()` looks like:

```javascript
const express = require('express');
const path = require('path');

const app = express();
const port = 3000;

// this line is key!
app.use(express.static(path.join(__dirname, 'public')));

app.get('/', (req, res) => {
  res.send('hello, world!');
});


app.listen(port, () => {
  console.log(`server running on port ${port}`);
});
```

in this example, all the files inside the 'public' folder (which should be in the same directory as your `server.js` or whatever you named your main server file) become accessible. if your `index.html` is in `public/index.html`, then it becomes accessible through `http://localhost:3000/index.html`. note that the folder 'public' it's not included in the url and it's like a virtual root for the browser. also, important to mention, when serving a `index.html` file, you do not need to specify the file's name, the web browser will serve it automatically so you would access it via `http://localhost:3000/`.

if you have that line in place and still get 404s, it’s time to look at the `path.join(__dirname, 'public')`. `__dirname` is a node.js magic variable that points to the directory your script is in. `path.join()` is a node.js method that combines paths correctly. it protects you from errors depending on which operative system the server is running on. now, check that the `'public'` directory is indeed in your folder and that's where all the files you are trying to serve are. i've been bitten by a silly typo in the folder name several times. seriously, look twice at the folder name, it's always that.

another thing to keep in mind is the order of your middleware. express executes middleware in the order it's added. that is fundamental. if you have a route handler (`app.get('/something', ... )`) that matches the request before `express.static()`, the static file serving will never be triggered.

here's an example of that:

```javascript
const express = require('express');
const path = require('path');

const app = express();
const port = 3000;

app.get('/index.html', (req, res) => {
  res.send('this route comes first, so no file will be served');
});

app.use(express.static(path.join(__dirname, 'public')));

app.listen(port, () => {
  console.log(`server running on port ${port}`);
});
```

in the above example, any request to `/index.html` will always trigger the route handler sending `"this route comes first, so no file will be served"` and the static file will never be returned.

i remember one time, i spent a good couple of hours debugging this, convinced that the path was the culprit. turns out, i had accidentally named my folder `publics` (notice the 's'). i swear, i wanted to throw my keyboard across the room. it's always the simplest things... sometimes the computer is not wrong it's me the one that's stupid, and that hurts more.

also, if you're dealing with more complex setups where you want specific files to be served only under certain conditions, you might be using the wrong middleware, you can use `app.use` for middleware that applies to all requests or you can use middleware specific to a certain route, like so `app.get('/something', someMiddleware, (req, res) => ... )`. but, from the question it does not seem to be the case so i'll try to stick to the most basic issue.

now, let’s imagine that you want only certain types of file to be accessible under specific paths, you could try something like this:

```javascript
const express = require('express');
const path = require('path');

const app = express();
const port = 3000;

app.use('/images', express.static(path.join(__dirname, 'public', 'images'), { extensions: ['jpg', 'png', 'gif'] }));
app.use('/css', express.static(path.join(__dirname, 'public', 'css'), { extensions: ['css'] }));

app.get('/', (req, res) => {
  res.send('hello, world!');
});

app.listen(port, () => {
  console.log(`server running on port ${port}`);
});
```

in this example, files in the `public/images` folder will only be accessible under `/images` path and only if they are a `.jpg`, `.png`, or `.gif`. similar behavior is happening with the `public/css` folder for `.css` files under the path `/css`. the folder `public` should be in the same level as the server file, and `images` and `css` should be folders within `public`. you can use the `extensions` option to provide an array of allowed extensions so express will only serve those files. this is a good practice if you want to filter which static assets you want to serve and can protect your server from unwanted file access.

sometimes it can also be that you have weird characters in the filenames, which can cause issues. it's best practice to avoid spaces and any kind of special character in your filenames and filepaths.

as a last piece of advice, if you're dealing with more advanced use cases, i recommend diving deep into the official express documentation, they have specific sections on serving static files and middleware in general, but it should not be needed for such a basic setup. also you can check "node.js design patterns" by mario casciaro and luciano mammino that book goes in-depth about node.js architecture and will provide you a ton of understanding about it.

in summary, triple-check your `express.static()` setup, verify your paths, be mindful of your middleware order, and just in case you have a very strange issue, verify that you don't have any weird filenames. the most probable issue it's the path you are sending to `express.static()` or its position in the middleware chain. if everything else fails, close your editor, take a walk and look at the problem again. sometimes a fresh pair of eyes can help you spot what is going wrong. trust me, this has saved me from so many headaches.
