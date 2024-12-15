---
title: "How to use a Socket.io server side implementation in Sails JS?"
date: "2024-12-15"
id: "how-to-use-a-socketio-server-side-implementation-in-sails-js"
---

alright, so you're looking to get socket.io running smoothly within a sails.js application. i've been there, trust me, it's a common stumbling block for folks new to sails and real-time stuff. i remember my first time, i was building this collaborative whiteboard app back in '16, or was it '17? anyway, things got real messy trying to shoehorn socket.io in without a proper setup. sails kinda handles some of the low-level http stuff, and figuring out where to inject the socket.io magic can be a bit confusing at first. but, let me break it down for you based on my past experience.

basically, sails doesn't automatically come with socket.io baked in. you’ve got to bring it in yourself. that's not a bad thing, it keeps the framework lightweight. you’ll first need to install the necessary package. it’s pretty standard node stuff. fire up your terminal, and within your sails project directory run:

```bash
npm install socket.io --save
```

that will add socket.io to your project's dependencies. once that’s done, the crucial part is hooking socket.io into sails. this usually involves a small bit of boilerplate. this bit, in my experience, is where many people go wrong initially. the key is to get to the raw http server sails uses under the hood. you need to modify the `config/http.js` file, you know that one, it is where the main configuration of the http server lives. you will need to grab the http server instance and then start the socket.io server from that instance. here’s the change i usually apply:

```javascript
// config/http.js
module.exports.http = {
  middleware: {
    order: [
      'cookieParser',
      'session',
      'bodyParser',
      'compress',
      'poweredBy',
      '$custom',
      'router',
      'www',
      'favicon',
    ],
    $custom: (function () {
      return function (req, res, next) {

       //the following part is crucial, we access the main server
        if (sails.hooks.http && sails.hooks.http.server) {
            if(!sails.io){
              sails.io = require('socket.io')(sails.hooks.http.server);
              sails.io.on('connection', function(socket) {
                console.log('a user connected');
                socket.on('disconnect', function(){
                   console.log('user disconnected');
                });
              //we can add more handlers here
              });
          }
        }

        return next();
      };
    })(),
  },
};
```

now, let’s get into this bit of code. the `config/http.js` file holds the middleware configuration. i’m injecting a `$custom` middleware, and this middleware runs after sails has fired up the http server. it checks if sails has hooks available and if an http server instance exists. if so, it initializes socket.io, attaches the instance to `sails.io` so we can use it elsewhere, and sets up a basic connection event listener, just to see if it works. this is the minimum you need, of course, in real projects you want to add all kind of business rules. remember to check for the proper order of middleware, i learned that one the hard way.

after that, you will start seeing ‘a user connected’ and ‘user disconnected’ messages in your terminal when connecting and disconnecting from the socket.io server. you can test it using a simple javascript client.

now that the server-side is set up, you can integrate it to your javascript code client side, the way to get to this part is using a client that can speak the socket.io protocol. you have to be aware that the socket.io implementation in sails won't automatically set up static file routing, which, for a test, is just fine. i'm going to create a basic javascript client, and then, we can see if this works. you can set it inside any html file that will be delivered by any action on the server.

```html
<!DOCTYPE html>
<html>
<head>
    <title>Socket.IO Test Client</title>
</head>
<body>
  <h1>Socket.io test</h1>
  <script src="/socket.io/socket.io.js"></script>
  <script>
   const socket = io();
   socket.on('connect', function(){
        console.log("connected to server")
   });
   socket.on('disconnect', function(){
      console.log("disconnected to server")
  });
 </script>
</body>
</html>
```

here you have a simple html client file, that will connect to the socket.io server you've created before. you have to be aware that the path `/socket.io/socket.io.js` will be set by default by socket.io and sails is able to serve it by default because it hooks itself to the http server.

one of the common places where i find myself using sockets in sails is within controllers. for example, imagine a chat application where you want to broadcast new messages to all connected users. in the controller action that handles the new message you have to access to the sails.io instance and send messages to clients. for example:

```javascript
// api/controllers/ChatController.js

module.exports = {
  newMessage: async function (req, res) {
    const message = req.param('message');
    sails.io.sockets.emit('newMessage', {
      message: message,
      sender: req.session.userId
    });
    return res.ok();
  },
};
```

this controller receives a new message and then using the sails.io.sockets.emit function it broadcasts to every connected client an event named ‘newMessage’ and sends the message, and the user id that originated that message. the client can then listen for this message and render it on the page.

as you can see, there are some caveats when integrating socket.io with sails but following these basic steps you should be able to start connecting clients to the server, now, if you want to have more complex behaviors or rooms you have to dig deeper into the socket.io documentation.

now, a few points i’ve learned the hard way:

*   **namespacing:** if your app grows, don't cram everything into the default namespace. socket.io's namespaces are your friends, use them. it’s much easier to manage different streams of data when you have it all separated logically.
*   **error handling:** socket.io connection/disconnection issues, lost messages, handle those gracefully, client and server side. never assume the network is going to be perfect.
*   **authentication:** if you're dealing with user-specific data over websockets, secure those connections. use something like jwt to identify users and use it on socket.io connection.
*   **scaling:** if you are expecting a lot of concurrent users you will have to scale your socket.io server, for this you need a message broker for example, rabbitmq or redis.
*   **debugging:** i remember once spending a whole night figuring out a websocket issue. check your browser's dev tools, check your server logs, it is tedious, i know, but it is the only way.

also, i recommend a book called 'realtime web' from ashley davis, a really good resource that was used back in the days when websockets started getting popular. it is a great resource and goes beyond socket.io. also, the official documentation of socket.io is great. it has examples and explanations of every part of the library. i also would recommend the book 'node.js design patterns' by mario casciaro and luciano mammino, which, although it does not focus specifically on websockets, will give you a general great view of backend patterns in node.

so, there you have it, a basic rundown of integrating socket.io with sails.js. it’s not too bad once you get the hang of it, remember the http server access is key. the good news is that you can easily create a simple chat app following these guidelines. and you may ask, why did the web developer start carrying around a ladder? because he wanted to take his code to the next level.
