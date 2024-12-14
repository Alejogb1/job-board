---
title: "Why am I being able to log in my web app when I have jwt set up?"
date: "2024-12-14"
id: "why-am-i-being-able-to-log-in-my-web-app-when-i-have-jwt-set-up"
---

alright, let's unpack this. the fact that you can log in even with jwt in place is a classic head-scratcher, and i've been there more times than i care to remember. it's usually one of a few culprits, and trust me, i’ve chased my tail on this particular issue more than once. from my early days building web apps with nothing but vanilla php and a healthy dose of sql injections (don't judge, we all started somewhere), to more modern frameworks, jwt issues like this one pop up. back then, debugging was mostly print statements and a lot of caffeine. good times, sort of.

so, the core idea of jwt is to avoid needing a session for authentication. the server gives you a jwt when you log in, which is basically a signed token proving that you are, well, you. your client then sends this token with each request, and the server validates it. no need to keep a session id around. if you're logging in, then you’ve effectively circumvented the expected auth flow.

let's go through the usual suspects i’ve encountered. first, double-check where your jwt authentication middleware is placed in the request pipeline. it needs to run *before* any endpoint that requires authentication. i’ve personally made the mistake of placing it *after* the login route, which, believe it or not, defeats the entire purpose. if you have a login route that does not enforce the jwt authentication flow then its bypassed. think of it like this, the login process, when it succeeds should give you a jwt, subsequent calls require the jwt be verified for valid access to resources.

here's a simplified example of a possible expressjs middleware setup (assuming node.js here):

```javascript
const express = require('express');
const jwt = require('jsonwebtoken');
const app = express();

const secretKey = 'your-secret-key'; // ideally from env variables

const authMiddleware = (req, res, next) => {
  const token = req.headers.authorization?.split(' ')[1]; // expecting: Bearer <token>

  if (!token) {
    return res.status(401).send('no token provided');
  }

  jwt.verify(token, secretKey, (err, decoded) => {
    if (err) {
      return res.status(403).send('invalid token');
    }

    req.user = decoded;
    next();
  });
};

app.post('/login', (req, res) => {
   // assuming some logic to create user here with valid username and password.
   const user = { id: 123, username: "user" };
  const token = jwt.sign(user, secretKey, { expiresIn: '1h' });
  res.json({ token: token });
});

app.get('/protected', authMiddleware, (req, res) => {
  res.send('protected route access granted!');
});

app.listen(3000, () => console.log('server started'));
```

notice that `authMiddleware` is applied only to `/protected` route. the login `/login` is not.

if your middleware isn't set up like this, then you have a potential issue there.

another thing i’ve seen countless times is inconsistent secret key usage. i remember once spending a good part of a day wondering why my tokens were always invalid on the backend only to find out i had a typo in the secret key variable on the server. both the client and the server must use the exact same secret key for signing and verifying jwt tokens. if the keys differ the server cannot properly validate your token hence it will reject the request. if you are using different secrets then the validation logic will fail and your authentication flow will not work as expected. if you are logging in then most likely your authentication logic is not happening as expected.

also, let's talk about the token itself. is your client even sending the token after the initial login? i've seen people forget to implement that part. after you get the token from the `/login` endpoint, you need to store that somewhere (local storage, cookies, in memory) and send it back with the `authorization` header for every subsequent request to protected routes. the typical format is:

```
authorization: bearer <your-jwt-token>
```

here is an example of how this would look in javascript using `fetch`:

```javascript
const login = async () => {
    const response = await fetch('/login', {
        method: 'post',
        headers: {
            'content-type': 'application/json'
        },
         body: JSON.stringify({
             username: 'user',
            password: 'password'
        })
    });

    const data = await response.json();
    localStorage.setItem('token', data.token); // store the token locally.
}

const fetchProtectedData = async () => {
  const token = localStorage.getItem('token');
  const response = await fetch('/protected', {
    headers: {
      authorization: `bearer ${token}`,
    }
  });
    const data = await response.text();
    console.log(data); // protected route access granted!
}
```

if the token isn't being added to the header correctly, the server will not receive it and your auth middleware will never run, in turn your routes will remain unprotected.

a bit more advanced case could be related to token expiration. jwt tokens can expire, and you have to handle this on the client side. it is important to verify both the expiration of the jwt in your auth middleware as well as have the client side handle when a token expires. i once got myself into a situation where the user could successfully make a request despite an expired token, because i wasn’t handling the expiration properly. a good way to check this on server is:

```javascript
jwt.verify(token, secretKey, (err, decoded) => {
    if (err) {
      if(err.name === 'TokenExpiredError'){
        return res.status(401).send('token expired');
      }
      return res.status(403).send('invalid token');
    }
    req.user = decoded;
    next();
  });
```

the client will have to handle the token expiration as well, usually by requesting a refresh token. which i won't go into detail here, but worth looking into. but if your token is expired and your logic fails to catch it. the user may log in without it even being valid.

now, i know you mentioned jwt is set up, but sometimes 'set up' can mean different things. you might have all the libraries in place, but there can be a fundamental mistake in how your auth flow is implemented. and believe me, it happens. this might not be your case specifically, but i've seen cases where even the person who wrote the login logic didn't fully understand the jwt logic itself. which creates cases like the one you are facing. i do recommend brushing up on the fundamentals, perhaps looking at papers on jwt security best practices, and the security aspect in detail. it helped me a lot to have this background knowledge as i've encountered a lot of weird edge cases along the way. i would recommend *understanding jwt: how json web tokens work* by alexandra edwards it covers a lot of concepts you can look into and be more familiar with jwt. also, look for papers about oauth2 and jwt since they go hand in hand in modern web apps.

in short, double-check: your middleware placement, secret key usage, if the client is even sending the token, and token expiration. i know it might seem obvious but it's usually one of those things that is messing the whole thing up. it can be frustrating when you know you have set things up, but it’s always the small stuff that gives you the most headaches. and hey, if all else fails, start printing stuff. console.log statements can be your best friend when debugging auth problems. in fact i've had cases where i spent hours chasing a bug only to find out there was a missing semi-colon. it was so bad that i considered programming with semicolons illegal (but that was a joke). anyway, i hope this helps narrow things down and good luck.
